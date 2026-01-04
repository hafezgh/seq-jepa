#!/bin/bash
#SBATCH --job-name=seqj_tr1_neurips_rebuttal_narval_3db
#SBATCH --account=rrg-shahabkb
##SBATCH --account=rrg-bengioy-ad
#SBATCH --export=ALL,DISABLE_DCGM=1
##SBATCH --array=0-1  
#SBATCH --mem=64G       
#SBATCH --time=0-23:59:59                 # Time limit (D-HH:MM:SS)
#SBATCH --nodes=1
#SBATCH --ntasks=4                    # Number of tasks (processes)
#SBATCH --gpus-per-node=4                      # Total number of GPUs
#SBATCH --cpus-per-task=8             # Number of CPU cores per task (num_workers)


#SBATCH --output=/home/hafezgh/projects/rrg-emuller/hafezgh/slurm_out/%x.%j.out
#SBATCH --error=/home/hafezgh/projects/rrg-emuller/hafezgh/slurm_out/%x.%j.err

##SBATCH --output=/home/hafezgh/projects/rrg-emuller/hafezgh/slurm_out/%x_%A_%a.out
##BATCH --error=/home/hafezgh/projects/rrg-emuller/hafezgh/slurm_out/%x_%A_%a.err


nvidia-smi

unset CUDA_VISIBLE_DEVICES


module load StdEnv/2020
module load python/3.10.2
module load cuda/11.8.0
module load cudacore/.12.2.2
module load nccl/2.18.5 

source ~/environments/TORCH-ENV/bin/activate

PYTHON_SCRIPT="/home/hafezgh/projects/rrg-emuller/hafezgh/seq-jepa-dev/src/main_aug.py"
WANDB_API_KEY="e953ffe81cc613c07e7030e1e66829658b82b514"

## CIFAR10, CIFAR100, and STL10 (DATA_ROOT will be set inside the python script)
# DATA_ROOT=$SLURM_TMPDIR

## TinyImagenet
# echo "Creating directory $SLURM_TMPDIR/tiny-imagenet-200"
# mkdir $SLURM_TMPDIR/tiny-imagenet-200
# echo "Extracting dataset to $SLURM_TMPDIR/tiny-imagenet-200"
# unzip -q /home/hafezgh/projects/rrg-emuller/hafezgh/datasets/tiny-imagenet-200.zip -d $SLURM_TMPDIR/tiny-imagenet-200
# DATA_ROOT=$SLURM_TMPDIR/tiny-imagenet-200/tiny-imagenet-200
# echo "Contents of $DATA_ROOT:"
# ls -l $DATA_ROOT

### 3DIEBench
echo "Creating directory $SLURM_TMPDIR/3diebench"
mkdir $SLURM_TMPDIR/3diebench
echo "Extracting dataset to $SLURM_TMPDIR/3diebench"
tar xf ~/projects/rrg-emuller/hafezgh/datasets/3DIEBench.tar.gz -C $SLURM_TMPDIR/3diebench
DATA_ROOT=$SLURM_TMPDIR/3diebench/3DIEBench



###STL-10-OLD
## echo "Creating directory $SLURM_TMPDIR/stl10_binary"
## mkdir $SLURM_TMPDIR/stl10_binary

## SOURCE_DIR="/home/hafezgh/projects/rrg-emuller/hafezgh/datasets/stl10_binary"
## DEST_DIR="$SLURM_TMPDIR/stl10_binary"

## echo "Copying dataset to $DEST_DIR"
## cp -r "$SOURCE_DIR"/* "$DEST_DIR"
## DATA_ROOT=$SLURM_TMPDIR

##echo "Copied data to $SLURM_TMPDIR"



export WANDB_MODE=offline



# Set MASTER_ADDR and MASTER_PORT
export MASTER_ADDR=$(hostname)
export MASTER_PORT=12345



srun python $PYTHON_SCRIPT \
 --num-workers $SLURM_CPUS_PER_TASK\
 --wandb-key $WANDB_API_KEY\
 --data-root $DATA_ROOT\
 --backbone "resnet18"\
 --dataset "3diebench"\
 --latent-type "rot"\
 --eval-type "rot"\
 --img-size 256\
 --epochs 2000\
 --save-freq 50\
 --seed 42\
 --batch-size 256\
 --run-id $SLURM_JOB_NAME\
 --num-heads 4\
 --num-enc-layers 3\
 --act-cond 1\
 --rel-act 1\
 --learn-act-emb 1\
 --ema\
 --ema-decay 0.996\
 --pred-hidden 1024\
 --offline-wandb\
 --narval\
 --distributed\
 --wandb\
 --model "seqjepa"\
 --alpha 0.01\
 --seq-len 1\
 --act-projdim 128\
 --lr 0.0004\
 --warmup 20\
 --scheduler\
 --weight-decay 0.001\
 --optimizer AdamW


## --alpha 0.1\
##--scheduler\
##--plus-projector
##--cifar-resnet\
##--data-path-img\
##--data-path-sal\
##--ckpt-wandb-id "36h022z0"\
##--ckpt-epoch 1000\
##--ckpt-folder "seq-jepa_stl10pls_leanract_numSac-5_edgesnew-1_pred1024"\
##--narval\
##--cedar\
##--offline-wandb\
##--permute-imgs
##--include-edges\
##--scheduler\
##--plot\
##--wandb\
##--no-rnn\
##--use-knn\
##--tsne\
##--mlp-proj\
##--no-pred\
##--use-posenc\
##--learn-posenc\
##--distributed\
##--online-linprobe\
