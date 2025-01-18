#!/usr/bin/env bash
#SBATCH --job-name=fine_tuning_tacred
#SBATCH --cpus-per-task=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=64GB
#SBATCH --gres=gpu:2
#SBATCH --output=LOG_FITXATEGIKO_PATH.log
#SBATCH --error=ERRORE_FITXATEGIKO_PATH.err
#SBATCH --mail-type=REQUEUE
#SBATCH --mail-user=italavera002@ikasle.ehu.eus

# Activate our required virtual environment
source ../.venv/bin/activate

# Specify common cache for Transformers library
export TRANSFORMERS_CACHE="/ncache/hub"

# # Specify common cache for Transformers library
srun python fine_tune_roberta_l_in_tacred.py
