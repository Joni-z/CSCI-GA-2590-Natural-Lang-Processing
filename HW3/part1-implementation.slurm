#! /bin/bash
#SBATCH --job-name=hw3-p1-impl
#SBATCH --nodes=1
#SBATCH --ntasks=1
#
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:04:00
#
#SBATCH --account=torch_pr_63_tandon_advanced
#
#SBATCH --mail-user=zz5070@nyu.edu
#SBATCH --mail-type=ALL
#SBATCH --output=/scratch/zz5070/CSCI-GA-2590-Natural-Lang-Processing/HW3/logs/hw3-p1-impl-%j.out
#SBATCH --error=/scratch/zz5070/CSCI-GA-2590-Natural-Lang-Processing/HW3/logs/hw3-p1-impl-%j.err

set -euo pipefail

CONDA_SH="${CONDA_SH:-/scratch/zz5070/miniconda3/etc/profile.d/conda.sh}"
source "$CONDA_SH"
conda activate "${CONDA_ENV:-NLP}"
export PATH="$CONDA_PREFIX/bin:$PATH"

cd /scratch/zz5070/CSCI-GA-2590-Natural-Lang-Processing/HW3
mkdir -p logs

cd part1
PYTHONPATH="$(pwd)" python -m tests.test_tokenizer implementation
