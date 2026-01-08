#!/bin/bash
#SBATCH --job-name=seqjepa-pls-stl10
#SBATCH --account=rrg-shahabkb
#SBATCH --export=ALL,DISABLE_DCGM=1
##SBATCH --array=0-1  
#SBATCH --mem=64G       
#SBATCH --time=0-23:59:59                 # Time limit (D-HH:MM:SS)
#SBATCH --nodes=1
#SBATCH --ntasks=1                    # Number of tasks (processes)
#SBATCH --gpus-per-node=1                     # Total number of GPUs
#SBATCH --cpus-per-task=8             # Number of CPU cores per task (num_workers)


#SBATCH --output=/home/hafezgh/projects/rrg-emuller/hafezgh/slurm_out/%x.%j.out
#SBATCH --error=/home/hafezgh/projects/rrg-emuller/hafezgh/slurm_out/%x.%j.err

##SBATCH --output=/home/hafezgh/projects/rrg-emuller/hafezgh/slurm_out/%x_%A_%a.out
##BATCH --error=/home/hafezgh/projects/rrg-emuller/hafezgh/slurm_out/%x_%A_%a.err


nvidia-smi


module load StdEnv/2020
module load python/3.10.2
module load cuda/11.8.0
module load cudacore/.12.2.2
module load nccl/2.18.5 

source ~/environments/TORCH-ENV/bin/activate

cd /home/hafezgh/projects/rrg-shahabkb/hafezgh/seq-jepa


# Set MASTER_ADDR and MASTER_PORT
export MASTER_ADDR=$(hostname)
export MASTER_PORT=12345

srun python src/main_pls.py \
 --num-workers $SLURM_CPUS_PER_TASK\
 --data-path-img /home/hafezgh/projects/rrg-emuller/hafezgh/datasets\
 --data-path-sal /home/hafezgh/projects/rrg-emuller/hafezgh/datasets/salmap-stl10\
 --output-folder /home/hafezgh/projects/rrg-emuller/hafezgh/seq-jepa-outputs\
 --cifar-resnet\
 --fovea-size 32\
 --num-saccades 5\
 --use-sal 1\
 --ior 1\
 --num-classes 10\
 --n-channels 3\
 --backbone "resnet18"\
 --dataset "stl10"\
 --img-size 96\
 --epochs 2000\
 --save-freq 100\
 --seed 42\
 --batch-size 512\
 --run-id $SLURM_JOB_NAME\
 --num-heads 4\
 --num-enc-layers 3\
 --act-cond 1\
 --learn-act-emb 1\
 --ema\
 --ema-decay 0.996\
 --pred-hidden 1024\
 --offline-wandb\
 --distributed\
 --wandb\
 --method "seqjepa"\
 --act-projdim 128\
 --lr 0.0004\
 --initial-lr 1e-5\
 --warmup 20\
 --scheduler\
 --weight-decay 0.001\
 --optimizer AdamW

