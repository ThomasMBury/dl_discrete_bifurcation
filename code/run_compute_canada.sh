#!/bin/bash -l
#SBATCH --job-name=reproducible_run
#SBATCH --account=def-glass # adjust this to match the accounting group you are using to submit jobs
#SBATCH --time=0-8:00:00         # adjust this to match the walltime of your job
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
pip install arch==4.19
pip install ewstools
pip install matplotlib
pip install seaborn
pip install kaleido

echo List all packages
pip list


GEN_TRAINING_DATA=false # generate the training data from scratch
TRAIN_CLASSIFIER=false # train a new deep learning classifier
QUICK_RUN=false # do a quick run that takes minimal compute time

if [ "$QUICK_RUN" = true ]
then 
    NSIMS=100 # Number of training simulations of each class
    NEPOCHS=2 # Number of epochs used to train the deep learning classifiers
    MODEL_SIMS=50 # Number of test model simulations
    INC=50 # Time increment between DL predictions in chick heart data
else
    NSIMS=10000
    NEPOCHS=200
    MODEL_SIMS=2500
    INC=5
fi

echo -e "-----\n Running code repository with NSIMS=$NSIMS, NEPOCHS=$NEPOCHS, MODEL_SIMS=$MODEL_SIMS, INC=$INC  \n-----"

if [ "$GEN_TRAINING_DATA" = true ]
then
    echo -e "-----\n Generate training data \n-----"
    cd training_data
    mkdir -p output
    python gen_training_data.py --nsims $NSIMS --verbose 0
    cd ../
fi

if [ "$TRAIN_CLASSIFIER" = true ]
then
    echo -e "-----\n Train classifier of type 1 \n-----"
    cd dl_train
    mkdir -p output
    python dl_train.py --model_type 1 --num_epochs $NEPOCHS --use_inter_train $GEN_TRAINING_DATA

    echo -e "-----\n Train classifier of type 2 \n-----"
    python dl_train.py --model_type 2 --num_epochs $NEPOCHS --use_inter_train $GEN_TRAINING_DATA
    cd ../
fi

echo -e "-----\n Make Sup Fig 1 - Confusion matrix of classifier on test data \n-----"
cd dl_train
mkdir -p output
python dl_test.py --use_inter_train $GEN_TRAINING_DATA --use_inter_classifier $TRAIN_CLASSIFIER

echo -e "-----\n Make Sup Fig 2 - example training simulations \n-----"
cd ../figure_s2
python make_fig.py

echo -e "-----\n Test DL classifier and EWS on Fox model \n-----"
cd ../test_fox
mkdir -p output
python test_fox.py --model_sims $MODEL_SIMS --use_inter_classifier $TRAIN_CLASSIFIER
python compute_roc.py

echo -e "-----\n Test DL classifier and EWS on Westerhoff model \n-----"
cd ../test_westerhoff
mkdir -p output
python test_westerhoff.py --model_sims $MODEL_SIMS --use_inter_classifier $TRAIN_CLASSIFIER
python compute_roc.py

echo -e "-----\n Test DL classifier and EWS on Ricker model \n-----"
cd ../test_ricker
mkdir -p output
python test_ricker.py --model_sims $MODEL_SIMS --use_inter_classifier $TRAIN_CLASSIFIER
python compute_roc.py

echo -e "-----\n Test DL classifier and EWS on Kot model \n-----"
cd ../test_kot
mkdir -p output
python test_kot.py --model_sims $MODEL_SIMS --use_inter_classifier $TRAIN_CLASSIFIER
python compute_roc.py

echo -e "-----\n Test DL classifier and EWS on Lorenz model \n-----"
cd ../test_lorenz
mkdir -p output
python test_lorenz.py --model_sims $MODEL_SIMS --use_inter_classifier $TRAIN_CLASSIFIER
python compute_roc.py

echo -e "-----\n Make Figure 2 - EWS and DL predictions for sample model simulations \n-----"
cd ../figure_2
mkdir -p output
python generate_data.py --use_inter_classifier $TRAIN_CLASSIFIER
python make_fig.py

echo -e "-----\n Make Sup Fig 3 - null simulations and EWS \n-----"
cd ../figure_s3
mkdir -p output
python generate_data.py --use_inter_classifier $TRAIN_CLASSIFIER
python make_fig.py

echo -e "-----\n Make Sup Fig 4 - example model test simulations \n-----"
cd ../figure_s4
mkdir -p output
python make_fig.py

echo -e "-----\n Make Sup Fig 5 - AUC at different rof and sigma \n-----"
cd ../figure_s5
mkdir -p output
python make_fig.py

echo -e "-----\n Make Sup Fig 6 - DL favourite bifurcation prop. correct \n-----"
cd ../figure_s6
mkdir -p output
python make_fig.py

echo -e "-----\n Make Sup Fig 7 - Performance of EWS for different parameter values in Fox model \n-----"
cd ../figure_s7
mkdir -p output
mkdir -p figures
python get_bifurcation_data.py
python compute_ews.py --use_inter_classifier $TRAIN_CLASSIFIER
python compute_roc_alpha.py
python compute_roc_scale.py
python make_fig.py

echo -e "-----\n Find transition times in chick heart data \n-----"
cd ../test_chick_heart
mkdir -p output
python find_transition_times.py

echo -e "-----\n Compute EWS in chick heart data \n-----"
python compute_ews.py --inc $INC --use_inter_classifier $TRAIN_CLASSIFIER

echo -e "-----\n Test EWS and DL in chick heart data \n-----"
python test_chick_heart.py --use_inter_classifier $TRAIN_CLASSIFIER
python compute_roc.py

echo -e "-----\n Make Fig S8 S9 - EWS in chick heart period-doubling \n-----"
cd ../figure_s8_s9
mkdir -p output
python make_fig.py

echo -e "-----\n Make Fig S10 S11 - EWS in chick heart null \n-----"
cd ../figure_s10_s11
mkdir -p output
python make_fig.py

echo -e "-----\n Make Figure 3 - ROC curves \n-----"
cd ../figure_3
mkdir -p output
python make_fig.py

echo -e "-----\n Make figure 4 - sample EWS and DL preds in chick heart data \n-----"
cd ../figure_4
mkdir -p output
python make_fig.py

echo -e "-----\n Make Fig S12 - demonstrate Lowess vs Gaussian detrending of chick heart data \n-----"
cd ../figure_s12
mkdir -p output
python make_fig.py

echo -e "-----\n Make Fig S13 - Performance of EWS using different detrending \n-----"
cd ../figure_s13
mkdir -p output
mkdir -p figures
python test_bandwidth.py --use_inter_classifier $TRAIN_CLASSIFIER
python test_span.py --use_inter_classifier $TRAIN_CLASSIFIER
python compute_roc_bw.py
python compute_roc_span.py
python make_fig.py

echo -e "-----\n Make Fig S14 - performance of lag-1 AC and variance for different rolling windows \n-----"
cd ../figure_s14
mkdir -p output
python test_rw.py
python compute_roc_rw.py
python make_fig.py

echo -e "-----\n Make Fig S15 - ROC curve for EWS on perturbed chick heart data \n-----"
cd ../figure_s15
mkdir -p output
mkdir -p figures
python test_sample_error.py --use_inter_classifier $TRAIN_CLASSIFIER
python compute_roc_pert.py
python make_fig.py

echo -e "-----\n Make Fig S16 - DL predictions on null simulations with different lambda \n-----"
cd ../figure_s16
mkdir -p output
python generate_data.py --use_inter_classifier $TRAIN_CLASSIFIER
python make_fig.py
