```
NODE_RANK=0
NUM_GPUS=4
CUDA_VISIBLE_DEVICES='0,1,2,3' python -m torch.distributed.launch \
    --nproc_per_node=${NUM_GPUS} --node_rank $NODE_RANK \
    pretrain_src/main_r2r.py --world_size ${NUM_GPUS} \
    --model_config pretrain_src/config/r2r_model_config.json \
    --config pretrain_src/config/pretrain_r2r.json \
    --output_dir datasets/R2R/exprs/pretrain/cmt-vitbase-6tasks
    
NODE_RANK=0
NUM_GPUS=1
CUDA_VISIBLE_DEVICES='0' python pretrain_src/main_r2r.py --world_size ${NUM_GPUS} \
    --model_config pretrain_src/config/r2r_model_large_config.json \
    --config pretrain_src/config/pretrain_r2r_vc_l.json \
    --output_dir datasets/R2R/exprs/pretrain/cmt-vitbase-6tasks-vc-l
    
NODE_RANK=0
NUM_GPUS=1
CUDA_VISIBLE_DEVICES='0' python pretrain_src/main_r2r.py --world_size ${NUM_GPUS} \
    --model_config pretrain_src/config/r2r_model_small_config.json \
    --config pretrain_src/config/pretrain_r2r_vit_16.json \
    --output_dir datasets/R2R/exprs/pretrain/cmt-vitbase-6tasks-vit-16
    
NUM_GPUS=1    
CUDA_VISIBLE_DEVICES='0' python pretrain_src/main_r2r.py --world_size ${NUM_GPUS} \
    --model_config pretrain_src/config/r2r_model_config.json \
    --config pretrain_src/config/pretrain_r2r_vc_b.json \
    --output_dir datasets/R2R/exprs/pretrain/cmt-vitbase-6tasks-vc-b-lxmert > $LOG
    
    
NODE_RANK=0
NUM_GPUS=4
CUDA_VISIBLE_DEVICES='0,1,2,3' python -m torch.distributed.launch \
    --nproc_per_node=${NUM_GPUS} --node_rank $NODE_RANK \
    pretrain_src/main_r2r_image.py --world_size ${NUM_GPUS} \
    --model_config pretrain_src/config/r2r_model_config.json \
    --config pretrain_src/config/pretrain_r2r_e2e.json \
    --output_dir datasets/R2R/exprs/pretrain/cmt-vitbase-6tasks-e2e
    
NODE_RANK=0
NUM_GPUS=1
OMP_NUM_THREADS=12 CUDA_VISIBLE_DEVICES='0' python -m torch.distributed.launch \
    --nproc_per_node=${NUM_GPUS} --master_port=25342 --node_rank $NODE_RANK \
    pretrain_src/main_r2r_image.py --world_size ${NUM_GPUS} \
    --model_config pretrain_src/config/r2r_model_config.json \
    --config pretrain_src/config/pretrain_r2r_e2e.json \
    --output_dir datasets/R2R/exprs/pretrain/cmt-vitbase-6tasks-e2e
    
python pretrain_src/main_r2r_image.py --world_size ${NUM_GPUS} \
    --model_config pretrain_src/config/r2r_model_config.json \
    --config pretrain_src/config/pretrain_r2r_e2e.json \
    --output_dir datasets/R2R/exprs/pretrain/cmt-vitbase-6tasks-e2e
    
/n/fs/kl-project/conda_env/envs/vlnhamt/bin/pip install timm==0.4.12
/n/fs/kl-project/conda_env/envs/vlnhamt/bin/pip install vc_models    
/n/fs/kl-project/conda_env/envs/eai-vc/bin/pip install h5py
/n/fs/kl-project/conda_env/envs/eai-vc/bin/pip install progressbar2
    
salloc --gres=gpu:rtx_3090:1 --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=32G -A visualai

salloc --nodelist node205 --gres=gpu:1 -ntasks=1 --cpus-per-task=8 --mem=80G -A allcs --time 12:00:00


CUDA_VISIBLE_DEVICES='0' python r2r/main.py $flag --eval_first \
      --aug ../datasets/R2R/annotations/prevalent_aug_train_enc.json \
      --bert_ckpt_file ../datasets/R2R/trained_models/vitbase-6tasks-pretrain/model_step_130000.pt --test
      
CUDA_VISIBLE_DEVICES='0' python r2r/main.py $flag \
      --aug ../datasets/R2R/annotations/prevalent_aug_train_enc.json \
      --bert_ckpt_file ../datasets/R2R/trained_models/vitbase-6tasks-pretrain-e2e/model_step_22000.pt

CUDA_VISIBLE_DEVICES='0' python r2r/main.py $flag --eval_first \
      --aug ../datasets/R2R/annotations/prevalent_aug_train_enc.json \
      --bert_ckpt_file ../datasets/R2R/exprs/pretrain/cmt-vitbase-6tasks/ckpts/model_step_115000.pt --test
      
CUDA_VISIBLE_DEVICES='0' python r2r/main.py $flag --eval_first \
      --aug ../datasets/R2R/annotations/prevalent_aug_train_enc.json \
      --bert_ckpt_file ../datasets/R2R/exprs/pretrain/cmt-vitbase-6tasks/ckpts/model_step_120000.pt --test    
            
CUDA_VISIBLE_DEVICES='0' python r2r/main.py $flag --eval_first \
      --aug ../datasets/R2R/annotations/prevalent_aug_train_enc.json \
      --bert_ckpt_file ../datasets/R2R/exprs/pretrain/cmt-vitbase-6tasks-vc-b/ckpts/model_step_325000.pt --test  

CUDA_VISIBLE_DEVICES='0' python r2r/main.py $flag --eval_first \
      --aug ../datasets/R2R/annotations/prevalent_aug_train_enc.json \
      --bert_ckpt_file ../datasets/R2R/exprs/pretrain/cmt-vitbase-6tasks-vc-b/ckpts/model_step_370000.pt --test  
      
CUDA_VISIBLE_DEVICES='0' python r2r/main.py $flag --eval_first \
      --aug ../datasets/R2R/annotations/prevalent_aug_train_enc.json \
      --bert_ckpt_file ../datasets/R2R/exprs/pretrain/cmt-vitbase-6tasks-vc-l/ckpts/model_step_335000.pt --test 
      
CUDA_VISIBLE_DEVICES='0' python r2r/main.py $flag --eval_first \
      --aug ../datasets/R2R/annotations/prevalent_aug_train_enc.json \
      --bert_ckpt_file ../datasets/R2R/exprs/pretrain/cmt-vitbase-6tasks-vc-l/ckpts/model_step_400000.pt --test 
      
CUDA_VISIBLE_DEVICES=0 python precompute_img_features_vc.py \
    --num_workers 1 \
    --output_file ../datasets/R2R/features/test.hdf5 \
    --vit base
    
    

      
```