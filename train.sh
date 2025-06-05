#CUDA_VISIBLE_DEVICES=0 torchrun --master_port=7777 --nproc_per_node=1 train.py -c configs/deim_dfine/deim_hgnetv2_s_visdrone.yml --seed=0

CUDA_VISIBLE_DEVICES=0 torchrun --master_port=7777 --nproc_per_node=1 train.py -c configs/dfine/dfine_hgnetv2_n_mal_custom.yml --seed=0