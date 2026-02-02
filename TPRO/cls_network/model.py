import pickle as pkl
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_

from cls_network import mix_transformer
from cls_network.attention import Block
from cls_network.engram import EMACodebook, EngramVision, EngramInject
from cls_network.knowledge_encoders import load_knowledge_features
from cls_network.cam_refine import CAMGuidedResidual, create_cam_mask



class AdaptiveLayer(nn.Module):
    def __init__(self, in_dim, n_ratio, out_dim):
        super().__init__()
        hidden_dim = int(in_dim * n_ratio)
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.relu = nn.ReLU()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


class ClsNetwork(nn.Module):
    def __init__(self,
                 backbone='mit_b1',
                 cls_num_classes=4,
                 stride=[4, 2, 2, 1],
                 pretrained=True,
                 n_ratio=0.5,
                 k_fea_path=None,
                 l_fea_path=None,
                 knowledge_encoder='clinical_bert',
                 dataset_name=None,
                 knowledge_features_base_dir=None,
                 bio_model_name='emilyalsentzer/Bio_ClinicalBERT',
                 engram_confidence_mode='norm',
                 engram_reweight_pseudo=True,
                 **kwargs):
        super().__init__()
        self.cls_num_classes = cls_num_classes
        self.stride = stride
        self._engram_reweight_pseudo = engram_reweight_pseudo
        self._confidence_mode = engram_confidence_mode

        self.encoder = getattr(mix_transformer, backbone)(stride=self.stride)
        self.in_channels = self.encoder.embed_dims

        if pretrained:
            state_dict = torch.load('./pretrained/'+backbone+'.pth', map_location="cpu")
            state_dict.pop('head.weight', None)
            state_dict.pop('head.bias', None)
            state_dict = {k: v for k, v in state_dict.items() if k in self.encoder.state_dict().keys()}
            self.encoder.load_state_dict(state_dict, strict=False)

        self.pooling = F.adaptive_avg_pool2d

        ## label features (medclip)
        self.l_fc1 = AdaptiveLayer(512, n_ratio, self.in_channels[0])
        self.l_fc2 = AdaptiveLayer(512, n_ratio, self.in_channels[1])
        self.l_fc3 = AdaptiveLayer(512, n_ratio, self.in_channels[2])
        self.l_fc4 = AdaptiveLayer(512, n_ratio, self.in_channels[3])
        with open("./text&features/text_features/{}.pkl".format(l_fea_path), "rb") as lf:
            self.l_fea = pkl.load(lf).cpu()
        self.logit_scale1 = nn.parameter.Parameter(torch.ones([1]) * 1 / 0.07)
        self.logit_scale2 = nn.parameter.Parameter(torch.ones([1]) * 1 / 0.07)
        self.logit_scale3 = nn.parameter.Parameter(torch.ones([1]) * 1 / 0.07)
        self.logit_scale4 = nn.parameter.Parameter(torch.ones([1]) * 1 / 0.07)

        ## knowledge features (clinical_bert or bio_clinical_bert)
        self.k_fea = load_knowledge_features(
            encoder_type=knowledge_encoder,
            feature_path=k_fea_path,
            knowledge_features_base_dir=knowledge_features_base_dir,
            bio_model_name=bio_model_name,
            dataset_name=dataset_name,
        ).cpu()
        self.k_fc4 = AdaptiveLayer(self.k_fea.shape[-1], n_ratio, self.in_channels[3])
        self.ka4 = nn.ModuleList([Block(self.in_channels[3], 8, mlp_ratio=4, qkv_bias=True, qk_scale=None, drop=0,
                                        attn_drop=0, drop_path=0.1, norm_layer=partial(nn.LayerNorm, eps=1e-6)) for _ in range(2)])

        ## EMA Codebook + Engram (stage 4)
        self._codebook_K = 64
        C4 = self.in_channels[3]
        self.ema_codebook = EMACodebook(K=self._codebook_K, C=C4, momentum=0.99, eps=1e-6)
        engram_vocab_size = max(self._codebook_K * 5, 512)
        self.engram4 = EngramVision(dim=C4, vocab_size=engram_vocab_size, num_heads=4)
        self.engram_inject = EngramInject(init_alpha=-2.0)

        ## CAM-Guided Residual Refinement for Stage 3
        self.cam_refine_enabled = kwargs.get('cam_refine_enabled', True)
        self.cam_refine_init_gamma = kwargs.get('cam_refine_init_gamma', 0.0)
        if self.cam_refine_enabled:
            C3 = self.in_channels[2]
            self.cam_refine = CAMGuidedResidual(init_gamma=self.cam_refine_init_gamma)

    def get_param_groups(self):
        regularized = []
        not_regularized = []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if name.endswith(".bias") or len(param.shape) == 1:
                not_regularized.append(param)
            elif "log_alpha" in name or "engram_inject" in name or "cam_refine.gamma" in name:
                not_regularized.append(param)
            else:
                regularized.append(param)
        return [{'params': regularized}, {'params': not_regularized, 'weight_decay': 0.}]

    def _forward_encoder_with_refinement(self, x):
        """
        Forward encoder with CAM-guided refinement for Stage 3.
        Refines F3 before feeding to Stage 4.
        """
        B = x.shape[0]
        outs = []
        attns = []
        
        # Stage 1
        x, H, W = self.encoder.patch_embed1(x)
        for i, blk in enumerate(self.encoder.block1):
            x, attn = blk(x, H, W)
            attns.append(attn)
        x = self.encoder.norm1(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)
        
        # Stage 2
        x, H, W = self.encoder.patch_embed2(x)
        for i, blk in enumerate(self.encoder.block2):
            x, attn = blk(x, H, W)
            attns.append(attn)
        x = self.encoder.norm2(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)
        
        # Stage 3
        x, H, W = self.encoder.patch_embed3(x)
        for i, blk in enumerate(self.encoder.block3):
            x, attn = blk(x, H, W)
            attns.append(attn)
        x = self.encoder.norm3(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        F3_raw = x
        outs.append(F3_raw)
        
        # CAM-Guided Refinement for Stage 3 (if enabled)
        if self.cam_refine_enabled and hasattr(self, 'cam_refine'):
            # Compute CAM3 from F3_raw (before refinement) - MUST use F3_raw, not F3_refined
            imshape_3 = F3_raw.shape
            _x3_flat = F3_raw.permute(0, 2, 3, 1).reshape(-1, F3_raw.shape[1])
            _x3_flat = _x3_flat / _x3_flat.norm(dim=-1, keepdim=True)
            l_fea3 = self.l_fc3(self.l_fea.to(x.device))
            logits_per_image3 = self.logit_scale3 * _x3_flat @ l_fea3.t().float()
            out3_raw = logits_per_image3.view(imshape_3[0], imshape_3[2], imshape_3[3], -1).permute(0, 3, 1, 2)
            cam3 = out3_raw.clone().detach()  # Detach to avoid backprop through CAM
            
            # Store CAM3 for later use (will be used in main forward, not recomputed)
            self._cam3_from_raw = cam3
            
            # Create CAM mask and refine F3
            cam3_mask = create_cam_mask(cam3, detach=True)  # (B, 1, H, W)
            F3_refined = self.cam_refine(F3_raw, cam3_mask)
            x = F3_refined
        else:
            x = F3_raw
            self._cam3_from_raw = None
        
        # Stage 4 (with refined F3 if enabled)
        x, H, W = self.encoder.patch_embed4(x)
        for i, blk in enumerate(self.encoder.block4):
            x, attn = blk(x, H, W)
            attns.append(attn)
        x = self.encoder.norm4(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)
        
        return outs, attns

    def forward(self, x, ema_update_enabled=True, **kwargs):
        # Use refinement-aware forward if enabled
        if self.cam_refine_enabled and hasattr(self, 'cam_refine'):
            _x, _attns = self._forward_encoder_with_refinement(x)
        else:
            _x, _attns = self.encoder(x) 

        logit_scale1 = self.logit_scale1
        logit_scale2 = self.logit_scale2
        logit_scale3 = self.logit_scale3
        logit_scale4 = self.logit_scale4

        imshape = [_.shape for _ in _x]
        image_features = [_.permute(0, 2, 3, 1).reshape(-1, _.shape[1]) for _ in _x]   
        _x1, _x2, _x3, _x4 = image_features
        l_fea = self.l_fea.to(x.device)
        l_fea1 = self.l_fc1(l_fea)
        l_fea2 = self.l_fc2(l_fea)
        l_fea3 = self.l_fc3(l_fea)
        l_fea4 = self.l_fc4(l_fea)
        _x1 = _x1 / _x1.norm(dim=-1, keepdim=True)
        logits_per_image1 = logit_scale1 * _x1 @ l_fea1.t().float() 
        out1 = logits_per_image1.view(imshape[0][0], imshape[0][2], imshape[0][3], -1).permute(0, 3, 1, 2) 
        cam1 = out1.clone().detach()
        cls1 = self.pooling(out1, (1, 1)).view(-1, l_fea1.shape[0]) 

        _x2 = _x2 / _x2.norm(dim=-1, keepdim=True)
        logits_per_image2 = logit_scale2 * _x2 @ l_fea2.t().float() 
        out2 = logits_per_image2.view(imshape[1][0], imshape[1][2], imshape[1][3], -1).permute(0, 3, 1, 2) 
        cam2 = out2.clone().detach()
        cls2 = self.pooling(out2, (1, 1)).view(-1, l_fea2.shape[0]) 

        _x3 = _x3 / _x3.norm(dim=-1, keepdim=True)
        logits_per_image3 = logit_scale3 * _x3 @ l_fea3.t().float() 
        out3 = logits_per_image3.view(imshape[2][0], imshape[2][2], imshape[2][3], -1).permute(0, 3, 1, 2)
        
        # Use CAM3 computed from F3_raw if refinement is enabled (to avoid self-feedback loop)
        # CAM3 MUST be computed from F3_raw, not from F3_refined
        if hasattr(self, '_cam3_from_raw') and self._cam3_from_raw is not None:
            cam3 = self._cam3_from_raw  # Already detached, computed from F3_raw
            # Still need out3 for cls3 computation (computed from refined features is OK for classification)
        else:
            cam3 = out3.clone().detach()
        cls3 = self.pooling(out3, (1, 1)).view(-1, l_fea3.shape[0]) 

        k_fea = self.k_fea.to(x.device)
        k_fea4 = self.k_fc4(k_fea)
        k_fea4 = k_fea4.reshape(1, k_fea4.shape[0], k_fea4.shape[1])
        k_fea4 = k_fea4.repeat(imshape[3][0], 1, 1)
        _x4 = _x4.reshape(imshape[3][0], -1, imshape[3][1])
        patch_feat_raw_4 = _x4

        cluster_ids = self.ema_codebook(_x4, update=ema_update_enabled)
        if self.training:
            self._last_cluster_ids = cluster_ids
        knowledge_tokens = self.engram4(_x4, position_ids=cluster_ids)
        enhanced_patches = self.engram_inject(_x4, knowledge_tokens)

        _z4 = torch.cat((enhanced_patches, k_fea4), dim=1)
        for blk in self.ka4:
            _z4, attn = blk(_z4, imshape[3][2], imshape[3][3])
        _x4 = _z4[:, :imshape[3][2] * imshape[3][3], :]
        patch_feat_4 = _x4
        _x4 = _x4.reshape(-1, imshape[3][1])
        _x4 = _x4 / _x4.norm(dim=-1, keepdim=True)
        logits_per_image4 = logit_scale4 * _x4 @ l_fea4.t().float()
        out4 = logits_per_image4.view(imshape[3][0], imshape[3][2], imshape[3][3], -1).permute(0, 3, 1, 2)

        if self._engram_reweight_pseudo:
            E = enhanced_patches - patch_feat_raw_4
            if self._confidence_mode == "cos":
                w = torch.sigmoid(F.cosine_similarity(patch_feat_raw_4, enhanced_patches, dim=-1))
            else:
                w = torch.sigmoid(E.norm(dim=-1))
            w_spatial = w.view(imshape[3][0], 1, imshape[3][2], imshape[3][3])
            cam4 = (w_spatial * out4).clone().detach()
        else:
            cam4 = out4.clone().detach()
        cls4 = self.pooling(out4, (1, 1)).view(-1, l_fea4.shape[0])
        class_tokens_4 = l_fea4

        return cls1, cam1, cls2, cam2, cls3, cam3, cls4, cam4, _attns, patch_feat_4, class_tokens_4
