# TPRO – Khôi phục sau khi xóa nhầm

Đã lấy lại code từ repo gốc [zhangst431/TPRO](https://github.com/zhangst431/TPRO) và giữ các chỉnh sửa từ chat.

## Đã có từ repo gốc

- **cls_network/** (model, mix_transformer, attention, conv_head)
- **datasets/** (bcss, luad_histoseg)
- **seg_network/**, **figures/**, **text&features/** (clinical_bert, medclip)
- **utils/** (trainutils, optimizer, evaluate, pyutils, imutils, tta_wrapper)
- **utils/cam_utils.py** – `get_seg_label` (lấy từ repo train_cls inline)
- **evaluate_cls.py**, **evaluate_seg.py**, **train_seg.py**
- **work_dirs/** (luad, bcss/segmentation), **requirements.txt**, **README.md**

## Đã chỉnh / thêm từ chat (giữ nguyên)

- **train_cls.py** – log_file trong config, TTA 5D guard, pseudo_refine fallback, LIR/consist (LIR no-op khi dùng model repo)
- **work_dirs/bcss/classification/config.yaml** – log_file, warmup/ramp, lir, consist, pseudo_refine, đường dẫn dataset
- **utils/pseudo_refine.py** – shortest-path refine + refined_to_soft_mask
- **utils/lir_utils.py** – no-op khi patch_feat_4/class_tokens_4 là None (model repo không trả về)
- **run_background.sh** – nohup / tmux

## Cần thêm tay

- **pretrained/mit_b1.pth** – tải từ [SegFormer](https://github.com/NVlabs/SegFormer) (theo README repo).
- Chỉnh `dataset.train_root` và `dataset.val_root` trong config nếu đường dẫn máy bạn khác.

## Chạy

```bash
cd TPRO
# BCSS (1 GPU)
CUDA_VISIBLE_DEVICES=0 python -u train_cls.py --config work_dirs/bcss/classification/config.yaml
# Chạy nền
./run_background.sh tmux work_dirs/bcss/classification/config.yaml
```
