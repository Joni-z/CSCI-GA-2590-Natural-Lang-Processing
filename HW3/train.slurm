#! /bin/bash
#SBATCH --job-name=jit
#SBATCH --nodes=1
#SBATCH --ntasks=1

#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=48:00:00

#SBATCH --gres=gpu:h100:1

#SBATCH --account=torch_pr_63_tandon_advanced

#SBATCH --mail-user=zz5070@nyu.edu
#SBATCH --mail-type=ALL
#SBATCH --output=/scratch/zz5070/ODE_FFN/jit-codebase/logs/jit-%j.out
#SBATCH --error=/scratch/zz5070/ODE_FFN/jit-codebase/logs/jit-%j.err

set -euo pipefail

# 作业里 Slurm/cgroup 通常只把「分到的那块卡」暴露成 cuda:0。登录 shell 里带的
# CUDA_VISIBLE_DEVICES 会指错或和 SLURM_JOB_GPUS 混用导致 PyTorch 报 No CUDA GPUs。
unset CUDA_VISIBLE_DEVICES

# Conda 激活
CONDA_SH="${CONDA_SH:-/scratch/zz5070/miniconda3/etc/profile.d/conda.sh}"
source "$CONDA_SH"
conda activate VLM
export PATH="$CONDA_PREFIX/bin:$PATH"

cd /scratch/zz5070/ODE_FFN/jit-codebase

CONFIG_NAME=${1:-jit_b16_in256_imagenet}
CONFIG_PATH="./configs/${CONFIG_NAME}.yaml"
mkdir -p logs

PRECISION=${PRECISION:-bf16}
GPUS_PER_NODE=${GPUS_PER_NODE:-1}
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-localhost}
MASTER_PORT=${MASTER_PORT:-12017}
WORLD_SIZE=$(($GPUS_PER_NODE * $NNODES))

echo "Config: $CONFIG_PATH | Log: logs/${CONFIG_NAME}.log | World size: $WORLD_SIZE"

accelerate launch \
  --main_process_ip "$MASTER_ADDR" \
  --main_process_port "$MASTER_PORT" \
  --machine_rank "$NODE_RANK" \
  --num_processes "$WORLD_SIZE" \
  --num_machines "$NNODES" \
  --mixed_precision $PRECISION \
  src/main_jit.py --config "$CONFIG_PATH" 2>&1 | tee "logs/${CONFIG_NAME}.log"
