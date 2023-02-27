#!/bin/bash -l
#SBATCH --job-name=reproducible_run
#SBATCH --account=def-glass # adjust this to match the accounting group you are using to submit jobs
#SBATCH --time=0-10:00:00         # adjust this to match the walltime of your job
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

# Number of trainng simulations of each class
NSIMS=10000

# Number of epochs with which to train DL classifiers
NEPOCHS=200

# Number of test model simulations 
MODEL_SIMS=2500

# Time increment between DL predictions in chick heart data
INC=5

echo -e "-----\n Generate training data \n-----"
cd training_data
mkdir -p output
python gen_training_data.py --nsims $NSIMS --verbose 0

echo -e "-----\n Train classifier of type 1 \n-----"
cd ../dl_train
mkdir -p output
python dl_train.py --model_type 1 --num_epochs $NEPOCHS

echo -e "-----\n Train classifier of type 2 \n-----"
python dl_train.py --model_type 2 --num_epochs $NEPOCHS

echo -e "-----\n Get F1 scores on test data \n-----"
python dl_test.py

echo -e "-----\n Make Sup Fig 2 - example training simulations \n-----"
cd ../figure_s2
python make_fig.py

echo -e "-----\n Make Sup Fig 3 - example model test simulations \n-----"
cd ../figure_s3
python make_fig.py

echo -e "-----\n Test DL classifier and EWS on Fox model \n-----"
cd ../test_fox
mkdir -p output
python test_fox.py --model_sims $MODEL_SIMS
python compute_roc.py

echo -e "-----\n Test DL classifier and EWS on Westerhoff model \n-----"
cd ../test_westerhoff
mkdir -p output
python test_westerhoff.py --model_sims $MODEL_SIMS
python compute_roc.py

echo -e "-----\n Test DL classifier and EWS on Ricker model \n-----"
cd ../test_ricker
mkdir -p output
python test_ricker.py --model_sims $MODEL_SIMS
python compute_roc.py

echo -e "-----\n Test DL classifier and EWS on Kot model \n-----"
cd ../test_kot
mkdir -p output
python test_kot.py --model_sims $MODEL_SIMS
python compute_roc.py

echo -e "-----\n Test DL classifier and EWS on Lorenz model \n-----"
cd ../test_lorenz
mkdir -p output
python test_lorenz.py --model_sims $MODEL_SIMS
python compute_roc.py

echo -e "-----\n Make Figure 2 - EWS and DL predictions for sample model simulations \n-----"
cd ../figure_2
mkdir -p output
python generate_data.py
python make_fig.py

echo -e "-----\n Make Sup Fig 4 - AUC scores across rof and sigma \n-----"
cd ../figure_s4
python make_fig.py

echo -e "-----\n Make Sup Fig 5 - DL favourite bifurcation prop. correct \n-----"
cd ../figure_s5
python make_fig.py

echo -e "-----\n Find transition times in chick heart data \n-----"
cd ../test_chick_heart
mkdir -p output
python find_transition_times.py

echo -e "-----\n Compute EWS in chick heart data \n-----"
python compute_ews.py --inc $INC

echo -e "-----\n Test EWS and DL in chick heart data \n-----"
python test_chick_heart.py
python compute_roc.py

echo -e "-----\n Make figure 3 - sample EWS and DL preds in chick heart data \n-----"
cd ../figure_3
python make_fig.py

echo -e "-----\n Make Fig S6 S7 - EWS in chick heart period-doubling \n-----"
cd ../figure_s6_s7
python make_fig.py

echo -e "-----\n Make Fig S8 S9 - EWS in chick heart null \n-----"
cd ../figure_s8_s9
python make_fig.py

echo -e "-----\n Make Figure 4 - ROC curves \n-----"
cd ../figure_4
mkdir -p output
python make_fig.py


