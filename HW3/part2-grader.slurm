#! /bin/bash
#SBATCH --job-name=hw3-p2-grade
#SBATCH --nodes=1
#SBATCH --ntasks=1
#
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#
#SBATCH --gres=gpu:1
#
#SBATCH --account=torch_pr_63_tandon_advanced
#
#SBATCH --mail-user=zz5070@nyu.edu
#SBATCH --mail-type=ALL
#SBATCH --output=/scratch/zz5070/CSCI-GA-2590-Natural-Lang-Processing/HW3/logs/hw3-p2-grade-%j.out
#SBATCH --error=/scratch/zz5070/CSCI-GA-2590-Natural-Lang-Processing/HW3/logs/hw3-p2-grade-%j.err

set -euo pipefail

# 避免登录环境里残留的 CUDA_VISIBLE_DEVICES 干扰 Slurm 分配的 GPU。
unset CUDA_VISIBLE_DEVICES

CONDA_SH="${CONDA_SH:-/scratch/zz5070/miniconda3/etc/profile.d/conda.sh}"
source "$CONDA_SH"
conda activate "${CONDA_ENV:-py312}"
export PATH="$CONDA_PREFIX/bin:$PATH"

cd /scratch/zz5070/CSCI-GA-2590-Natural-Lang-Processing/HW3/part2
mkdir -p /scratch/zz5070/CSCI-GA-2590-Natural-Lang-Processing/HW3/logs
mkdir -p results

python run_tests.py
