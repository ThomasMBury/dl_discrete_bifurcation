#!/bin/bash

# Shell script to execute all code required to reproduce results.

# Number of trainng simulations of each class
NSIMS = 10

# Number of epochs with which to train DL classifiers
NEPOCHS = 5

# Number of test model simulations 
MODEL_SIMS = 25

# Generate training data
cd training_data
echo Generate training data
python gen_training_data.py --nsims $NSIMS --verbose 0


# Train DL classifier of type 1
cd ../dl_train
echo Train classifier of type 1
python dl_train.py --model_type 1 --seed 0 --nsims $NSIMS --num_epochs $NEPOCHS

# Train DL classifier of type 2
echo Train classifier of type 1
python dl_train.py --model_type 2 --seed 0 --nsims $NSIMS --num_epochs $NEPOCHS

# Get F1 scores and confusion matrices - make Sup Fig 1
echo Get F1 scores on test data
python dl_test.py

# Make Sup Fig 2 - example training trajectories
cd ../figure_s2
echo Make Sup Fig 2
python make_fig.py

# Make Sup Fig 3 - example model test trajectories
cd ../figure_s3
echo make Sup Fig 3
python make_fig.py

# Fox model - test DL classifier and EWS
cd ../test_fox
echo Test DL classifier on Fox model
python test_fox.py --model_sims $MODEL_SIMS
python compute_roc.py --model_sims $MODEL_SIMS

# Westerhoff model - test DL classifier and EWS
cd ../test_westerhoff
echo Test DL classifier on Westerhoff model
python test_westerhoff.py --model_sims $MODEL_SIMS
python compute_roc.py --model_sims $MODEL_SIMS

# Ricker model - test DL classifier and EWS
cd ../test_ricker
echo Test DL classifier on Ricker model
python test_ricker.py --model_sims $MODEL_SIMS
python compute_roc.py --model_sims $MODEL_SIMS

# Kot model - test DL classifier and EWS
cd ../test_kot
echo Test DL classifier on Kot model
python test_kot.py --model_sims $MODEL_SIMS
python compute_roc.py --model_sims $MODEL_SIMS

# Lorenz model - test DL classifier and EWS
cd ../test_lorenz
echo Test DL classifier on Lorenz model
python test_lorenz.py --model_sims $MODEL_SIMS
python compute_roc.py --model_sims $MODEL_SIMS

# Make Figure 2 - sample EWS and DL predictions in model
cd ../figure_2
echo Make Figure 2
python generate_data.py
python make_fig.py

# Make Sup Fig 4 - AUC scores across rof and sigma
cd ../figure_s4
echo Make Figure S4
python make_fig.py

# Make Sup Fig 5 - DL favourite bifurcation prop. correct
cd ../figure_s5
echo Make Figure S5
python make_fig.py

# Chick heart data - get transitions and compute rolling EWS
cd ../test_chick_heart
echo Find transition times in chick heart data
python find_transition_times.py
echo Compute EWS in chick heart data
python compute_ews.py

# Test DL and EWS with chick heart data
echo Tet EWS and DL in chick heart data
python test_chick_heart.py
python compute_roc.py

# Make figure 3 - sample EWS and DL preds in chick heart data
cd ../figure_3
echo Make Figure 3
python make_fig.py

# Make Sup Fig 6, 7 - EWS in chick heart forced sims
cd ../figure_s6_s7
echo Make Figure S6 and S7
python make_fig.py

# Make Sup Fig 8, 9 - EWS in chick heart null sims
cd ../figure_s8_s9
echo Make Figure S8 and S9
python make_fig.py

# Make Figure 4 - ROC curves
cd ../figure_4
echo Make Figure 4
python make_fig.py



