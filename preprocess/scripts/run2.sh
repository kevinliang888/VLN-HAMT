#!/bin/sh
#SBATCH --account=allcs    # Specify VisualAI
#SBATCH --nodes=1             # nodes requested
#SBATCH --ntasks=1            # tasks requested
#SBATCH --cpus-per-task=8     # Specify the number of CPUs your task will need.
#SBATCH --gres=gpu:a6000:1          # the number of GPUs requested
#SBATCH --mem=50G             # memory
#SBATCH -o outfile222            # send stdout to outfile
#SBATCH -e errfile222            # send stderr to errfile
#SBATCH -t 48:00:00           # time requested in hour:minute:second
#SBATCH --mail-type=all       # choice between begin, end, all to notify you via email
#SBATCH --mail-user=kl2471@princeton.edu

BDIR=/n/fs/kl-project/VLN-HAMT
JOBID=$SLURM_JOB_ID
LOG=$BDIR/logs/log1.txt

echo -e "JobID: $JOBID\n======" > $LOG
echo "Time: `date`" >> $LOG
echo "Running on master node: `hostname`" >> $LOG

#module load anaconda
#source activate <your-virtual-environment>
#module load cudatoolkit
#module load cudann/cuda-8.0/5.1
python precompute_img.py > $LOG
echo "Time: `date`" >> $LOG