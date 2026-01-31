# Flow: Từ ảnh đầu vào → Encoder → CAM / pseudo mask → Validation IOU và Training

Tài liệu mô tả luồng theo **plan** và **đối chiếu với code hiện tại** (repo gốc [zhangst431/TPRO](https://github.com/zhangst431/TPRO) + chỉnh sửa trong chat).

---

## 1. Pipeline tổng thể

```
Input image (B, 3, H, W)
    → Encoder (SegFormer/MiT backbone, 4 stages)
    → Multi-scale features _x1, _x2, _x3, _x4
    → Stage 1–3: pixel features × text embeddings (l_fea) → CAM 1–3, cls 1–3
    → Stage 4: (plan) _x4 → EMA Codebook → cluster_ids
                         → Engram(position_ids=cluster_ids) → knowledge_tokens
                         → EngramInject: enhanced = _x4 + alpha * knowledge_tokens
                         → Knowledge Attention (ka4) với k_fea
                         → pixel × l_fea4 → CAM4, cls4
              (code hiện tại) _x4 → concat k_fea → Knowledge Attention (ka4) → l_fea4 → CAM4, cls4
    → CAM 1–4: (1) classification loss; (2) pseudo mask (get_seg_label); (3) LIR (chỉ L_cre, optional)
    → Pseudo mask: sim_only (normalize + bg); (plan) optional Engram reweight S' = w⊙S; (code) không có Engram reweight
    → (Optional) shortest-path refine sau fuse CAM
    → Validation: TTA → fuse234 → [optional refine] → argmax → IOU
```

**Trạng thái code hiện tại**

- **Encoder + Stage 1–3**: Giống plan. File `cls_network/model.py`, `cls_network/mix_transformer.py`.
- **Stage 4**: Đã khôi phục: _x4 → **EMA Codebook** → cluster_ids → **Engram** (position_ids=cluster_ids) → **EngramInject** → concat k_fea → KA (ka4) → CAM4, cls4. Có **Engram reweight** (S' = w⊙S) khi `engram_reweight_pseudo: true`. Model trả về `patch_feat_4`, `class_tokens_4` cho LIR.
- **LIR**: `utils/lir_utils.py` đã implement L_cre (confident patch–class contrast).

---

## 2. Encoder và CAM (stage 1–3)

**Plan & code (khớp)**

- **Encoder**: `mix_transformer` (SegFormer/MiT), stride `[4, 2, 2, 1]` → 4 cấp feature _x1, _x2, _x3, _x4.
- **Text**: `l_fea` load từ pkl (MedCLIP label 512-d), project qua `l_fc1`–`l_fc4` xuống channel từng stage.
- **Stage 1–3**:
  - `image_features` = flatten spatial (_xi) → (B×Hi×Wi, C).
  - L2 normalize, `logit_scale_i * feat @ l_fei.t()` → logits.
  - Reshape (B, num_classes, Hi, Wi) → **CAM**; global average pooling → **cls** logits.

**File**: `cls_network/model.py` (dòng ~107–131).

---

## 3. Stage 4: (Plan) EMA Codebook + Engram vs (Code) Knowledge Attention

### 3.1 Plan (chưa có trong repo)

- **EMA Codebook** (`cls_network/engram.py`): hidden_states → cluster_ids (content-aware), dùng làm position_ids cho Engram.
- **Engram**: hash theo cluster_id → knowledge_tokens.
- **EngramInject**: `enhanced_patches = _x4 + alpha * knowledge_tokens` (alpha learnable, init ≈ 0.135).
- **Knowledge Attention (ka4)**: concat(enhanced_patches, k_fea) → Block × 2 → patch features → × l_fea4 → CAM4, cls4.

### 3.2 Code hiện tại (đã khôi phục Engram)

- **Có** EMA Codebook (`cls_network/engram.py`), Engram, EngramInject.
- Luồng: `_x4` → `cluster_ids = ema_codebook(_x4, update=ema_update_enabled)` → `knowledge_tokens = engram4(_x4, position_ids=cluster_ids)` → `enhanced_patches = engram_inject(_x4, knowledge_tokens)` → concat với `k_fea4` → `ka4` (2 Block) → patch_feat_4 → × l_fea4 → out4. Nếu `engram_reweight_pseudo`: w = sigmoid(‖E‖) hoặc cos, **cam4** = w_spatial ⊙ out4; ngược lại cam4 = out4.
- `forward()` trả về `(cls1, cam1, ..., cls4, cam4, _attns, patch_feat_4, class_tokens_4)`.

**File**: `cls_network/model.py`, `cls_network/engram.py`, `cls_network/knowledge_encoders.py`.

---

## 4. Pseudo mask (get_seg_label)

**Plan**: sim_only (normalize + bg). Optional: Engram reweight (S' = w⊙S) cho CAM4; optional shortest-path refine sau fuse.

**Code hiện tại**

- **get_seg_label** (`utils/cam_utils.py`): giống plan “sim_only”:
  - CAM normalize (min-max theo channel), nhân với label (chỉ class có trong ảnh).
  - Interpolate lên (H, W).
  - `bg_cam = (1 - max_cam)^10`, concat → output (B, C+1, H, W).
- **Engram reweight**: không có (model không trả về enhanced_patches / confidence w).
- **Shortest-path refine**: có trong **validation** khi `train.pseudo_refine_enabled: true` (`utils/pseudo_refine.py`): sau khi fuse CAM (0.3·cam2 + 0.3·cam3 + 0.4·cam4), refine bằng Dijkstra từ confident → uncertain, rồi argmax → so sánh GT → IOU.

**File**: `utils/cam_utils.py`, `utils/pseudo_refine.py`; gọi trong `train_cls.py` `validate()`.

---

## 5. Localization-informed Regularization (LIR)

**Plan**: Ma (multi-threshold từ CAM) → L_cre (confident relation enhancement). L_ure đã bỏ.

**Code hiện tại**

- Model trả về `patch_feat_4`, `class_tokens_4` (sau KA stage 4). `utils/lir_utils.py` tính **L_cre**: Ma (multi-threshold từ CAM) → vùng confident → contrastive loss (patch–class token).

**File**: `utils/lir_utils.py`, `train_cls.py` (đoạn loss LIR).

---

## 6. Training (train_cls.py)

**Luồng mỗi iteration**

1. `inputs` → **ClsNetwork** → cls1..cls4, cam1..cam4, patch_feat_4, class_tokens_4 (hai cái sau = None với model hiện tại).
2. **Classification loss**: l1·loss1 + l2·loss2 + l3·loss3 + l4·loss4 (BCE).
3. **LIR** (nếu patch_feat_4 không None): lir_weight * L_cre; hiện tại = 0.
4. **Consistency loss** (nếu `consist_weight > 0`): weak aug view2 → forward → mask1, mask2 từ get_seg_label(cam4) → MSE(mask1, mask2).
5. Backward, optimizer step.
6. (Optional) Log cluster / codebook nếu có; hiện tại model không có.

**Phase warmup/ramp** (config): warmup_iters, ramp_iters để bật dần LIR/consist/EMA; với model repo chỉ ảnh hưởng consistency và logic phase (ema_update_enabled).

**File**: `train_cls.py` (vòng lặp train, đoạn loss).

---

## 7. Validation

**Luồng**

1. Với mỗi batch: forward 1 lần → cls loss (trên cls4).
2. **TTA**: HorizontalFlip × Multiply [0.9, 1.0, 1.1]. Với mỗi biến thể TTA:
   - Forward → cam2, cam3, cam4 (input 5D được flatten thành 4D; nếu CAM shape không hợp lệ thì bỏ qua TTA đó).
   - Interpolate cam về (h, w) → fuse234 = 0.3·cam2 + 0.3·cam3 + 0.4·cam4.
   - Nếu **pseudo_refine_enabled**: refine pseudo bằng shortest-path (lam_low, lam_high, max_cost, neighbor) → soft mask → deaugment.
   - Nếu không refine: argmax(fuse234) → fuse_label234.
3. Gộp theo TTA (average), rồi argmax → so sánh với GT mask → **IOU** (fuse234_score).
4. Nếu mọi TTA đều lỗi (CAM shape invalid): fallback dùng CAM gốc (không TTA) để tính fuse_label234.

**File**: `train_cls.py` → `validate()`; `utils/pseudo_refine.py` (refine_pseudo_by_shortest_path, refined_to_soft_mask).

---

## 8. Tóm tắt đối chiếu plan vs code

| Thành phần | Plan | Code hiện tại |
|------------|------|----------------|
| Encoder + Stage 1–3 | Có | Có, giống |
| Stage 4: EMA Codebook | Có | Có (engram.py) |
| Stage 4: Engram + EngramInject | Có | Có (engram.py, model.py) |
| Stage 4: Knowledge Attention (k_fea) | Có | Có |
| get_seg_label (sim_only) | Có | Có (utils/cam_utils.py) |
| Engram reweight CAM4 (S'=w⊙S) | Optional | Có (model, engram_reweight_pseudo) |
| Shortest-path refine | Optional | Có (validation, pseudo_refine.py) |
| LIR (L_cre) | Optional | Có (utils/lir_utils.py) |
| Consistency loss | Có | Có |
| Validation TTA + fuse234 + IOU | Có | Có (+ 5D guard + fallback) |
| Log file (work_dir.log_file) | — | Có (config + setup_logger) |

---

## 9. File tham chiếu nhanh

| Luồng | File |
|-------|------|
| Encoder 4 stage | `cls_network/mix_transformer.py` |
| CAM 1–4, Knowledge Attention stage 4 | `cls_network/model.py` |
| Pseudo mask (sim_only) | `utils/cam_utils.py` → `get_seg_label` |
| Shortest-path refine | `utils/pseudo_refine.py` |
| LIR (hiện tại no-op) | `utils/lir_utils.py` |
| Train + validation | `train_cls.py` |
| Config (path, LIR, pseudo_refine, log_file) | `work_dirs/bcss/classification/config.yaml` |
