CUDA_VISIBLE_DEVICES=2 python train.py \
  --config checkpoint/pose3d/PoseMamba_B/config.yaml \
  --evaluate checkpoint/pose3d/PoseMamba_B/best_epoch.bin \
  --checkpoint eval/checkpoint
# CUDA_VISIBLE_DEVICES=2 python train.py \
#   --config checkpoint/pose3d/PoseMamba_S/config.yaml \
#   --evaluate checkpoint/pose3d/PoseMamba_S/best_epoch.bin \
#   --checkpoint eval/checkpoint
# CUDA_VISIBLE_DEVICES=2 python train.py \
#   --config checkpoint/pose3d/PoseMamba_L/config.yaml \
#   --evaluate checkpoint/pose3d/PoseMamba_L/best_epoch.bin \
#   --checkpoint eval/checkpoint