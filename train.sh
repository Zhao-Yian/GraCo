# Note: The abbreviation ''fmg'' stands for Fine-grained Mask Generator, which is the same as the mask engine in previous AGG.
# GraCo w/ MMT+FMG ViT-B
CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py models/plainvit_base448_graco_mmt_fmg.py --part_path /path/to/fmg_proposal.pkl --enable_lora --weights weights/simpleclick/sbd_vit_base.pth --gpus 0,1,2,3

# GraCo w/ MMT+FMG ViT-L
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python train.py models/plainvit_large448_graco_mmt_fmg.py --part_path /path/to/fmg_proposal.pkl --enable_lora --weights weights/simpleclick/sbd_vit_large.pth --gpus 0,1,2,3,4,5,6,7