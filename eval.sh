# Instance-level benchmarks
CUDA_VISIBLE_DEVICES=0 python evaluate.py NoBRS --datasets GrabCut,Berkeley,SBD,DAVIS,PascalVOC --checkpoint weights/simpleclick/sbd_vit_base.pth --lora_checkpoint weights/graco/GraCo_base_lora.pth --graco
CUDA_VISIBLE_DEVICES=0 python evaluate.py NoBRS --datasets GrabCut,Berkeley,SBD,DAVIS,PascalVOC --checkpoint weights/simpleclick/sbd_vit_large.pth --lora_checkpoint weights/graco/GraCo_large_lora.pth --graco


# Part-level benchmarks
CUDA_VISIBLE_DEVICES=0 python evaluate.py NoBRS --datasets PascalPart,PartImageNet,SA1B --checkpoint weights/simpleclick/sbd_vit_base.pth --lora_checkpoint weights/graco/GraCo_base_lora.pth --graco
for c in "head" "wheel" "torso"; do CUDA_VISIBLE_DEVICES=0 python evaluate.py NoBRS --datasets PascalPart --checkpoint weights/simpleclick/sbd_vit_base.pth --lora_checkpoint weights/graco/GraCo_base_lora.pth --phrase $c --part-name $c; done
for c in "head" "body" "foot"; do CUDA_VISIBLE_DEVICES=0 python evaluate.py NoBRS --datasets PartImageNet --checkpoint weights/simpleclick/sbd_vit_base.pth --lora_checkpoint weights/graco/GraCo_base_lora.pth --phrase $c --part-name $c; done

CUDA_VISIBLE_DEVICES=0 python evaluate.py NoBRS --datasets PascalPart,PartImageNet,SA1B --checkpoint weights/simpleclick/sbd_vit_large.pth --lora_checkpoint weights/graco/GraCo_large_lora.pth --graco
for c in "head" "wheel" "torso"; do CUDA_VISIBLE_DEVICES=0 python evaluate.py NoBRS --datasets PascalPart --checkpoint weights/simpleclick/sbd_vit_large.pth --lora_checkpoint weights/graco/GraCo_large_lora.pth --phrase $c --part-name $c; done
for c in "head" "body" "foot"; do CUDA_VISIBLE_DEVICES=0 python evaluate.py NoBRS --datasets PartImageNet --checkpoint weights/simpleclick/sbd_vit_large.pth --lora_checkpoint weights/graco/GraCo_large_lora.pth --phrase $c --part-name $c; done


# Out-of-domain benchmarks
CUDA_VISIBLE_DEVICES=0 python evaluate.py NoBRS --datasets ssTEM,BraTS,OAIZIB --checkpoint weights/simpleclick/sbd_vit_base.pth --lora_checkpoint weights/graco/GraCo_base_lora.pth --graco
CUDA_VISIBLE_DEVICES=0 python evaluate.py NoBRS --datasets ssTEM,BraTS,OAIZIB --checkpoint weights/simpleclick/sbd_vit_large.pth --lora_checkpoint weights/graco/GraCo_large_lora.pth --graco