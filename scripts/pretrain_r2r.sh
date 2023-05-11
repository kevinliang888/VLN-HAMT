#!/bin/sh
#SBATCH --account=visualai    # Specify VisualAI
#SBATCH --nodes=1             # nodes requested
#SBATCH --ntasks=1            # tasks requested
#SBATCH --cpus-per-task=8     # Specify the number of CPUs your task will need.
#SBATCH --gres=gpu:rtx_3090:4          # the number of GPUs requested
#SBATCH --mem=60G             # memory
#SBATCH -o outfile            # send stdout to outfile
#SBATCH -e errfile            # send stderr to errfile
#SBATCH -t 120:00:00           # time requested in hour:minute:second
#SBATCH --mail-type=all       # choice between begin, end, all to notify you via email
#SBATCH --mail-user=kl2471@princeton.edu

BDIR=/n/fs/kl-project/VLN-HAMT
JOBID=$SLURM_JOB_ID
LOG=$BDIR/logs/log_pretrain_r2r.txt

NODE_RANK=0
NUM_GPUS=4
CUDA_VISIBLE_DEVICES='0,1,2,3' python -m torch.distributed.launch \
    --nproc_per_node=${NUM_GPUS} --node_rank $NODE_RANK \
    pretrain_src/main_r2r.py --world_size ${NUM_GPUS} \
    --model_config pretrain_src/config/r2r_model_config.json \
    --config pretrain_src/config/pretrain_r2r.json \
    --output_dir datasets/R2R/exprs/pretrain/cmt-vitbase-6tasks > $LOG