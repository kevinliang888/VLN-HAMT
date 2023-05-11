#!/bin/sh
#SBATCH --account=visualai    # Specify VisualAI
#SBATCH --nodes=1             # nodes requested
#SBATCH --ntasks=1            # tasks requested
#SBATCH --cpus-per-task=8     # Specify the number of CPUs your task will need.
#SBATCH --gres=gpu:rtx_3090:1          # the number of GPUs requested
#SBATCH --mem=50G             # memory
#SBATCH -o ./outs/outfile_base_vc            # send stdout to outfile
#SBATCH -e ./outs/errfile_base_vc            # send stderr to errfile
#SBATCH -t 120:00:00           # time requested in hour:minute:second
#SBATCH --mail-type=all       # choice between begin, end, all to notify you via email
#SBATCH --mail-user=kl2471@princeton.edu

BDIR=/n/fs/kl-project/VLN-HAMT
JOBID=$SLURM_JOB_ID
LOG=$BDIR/logs/log_preprocess_vc_b.txt

echo -e "JobID: $JOBID\n======" > $LOG
echo "Time: `date`" >> $LOG
echo "Running on master node: `hostname`" >> $LOG

#module load anaconda
#source activate <your-virtual-environment>
#module load cudatoolkit
#module load cudann/cuda-8.0/5.1
CUDA_VISIBLE_DEVICES=0 python precompute_img_features_vc.py \
    --num_workers 4 \
    --vit base \
    --output_file ../datasets/R2R/features/pth_vc1_base.hdf5 > $LOG

echo "Time: `date`" >> $LOG