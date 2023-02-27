#!/bin/bash -l
#SBATCH --job-name=reproducible_run
#SBATCH --account=def-glass # adjust this to match the accounting group you are using to submit jobs
#SBATCH --time=0-1:00:00         # adjust this to match the walltime of your job
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4      # adjust this if you are using parallel commands
#SBATCH --mem=32000M             # adjust this according to the memory requirement per node you need
#SBATCH --mail-user=thomas.bury@mcgill.ca # adjust this to match your email address
#SBATCH --mail-type=END
#SBATCH --output=stdout/job-%j.out

echo Job $SLURM_JOB_ID released

# Load modules
echo Load modules
module load gcc/9.3.0 arrow python scipy-stack cuda cudnn

# Create virtual env
echo Create virtual environemnt
virtualenv --no-download $SLURM_TMPDIR/venv
source $SLURM_TMPDIR/venv/bin/activate

# Install packages
echo Install packages
pip install --no-index --upgrade pip

pip install tensorflow
pip install scikit-learn
pip install ewstools
pip install matplotlib
pip install kaleido


# Number of test model simulations 
MODEL_SIMS=2500

echo -e "-----\n Test DL classifier and EWS on Kot model \n-----"
mkdir -p output
python test_kot.py --model_sims $MODEL_SIMS
python compute_roc.py
