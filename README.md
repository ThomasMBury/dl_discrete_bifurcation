# Predicting discrete-time bifurcations with deep learning

This is the code repository to accompany the article "Predicting discrete-time bifurcations with deep learning".

The code also appears as a [code capsule](https://codeocean.com/capsule/3359094/tree) on Code Ocean, which provides the software environment and compute resources to do a reproducible run of the results reported in the paper. Alternatively, you may do a reproducible run on your local computer following the instructions below.

## Instructions to reproduce results

- Clone the repository
```
git clone git@github.com:ThomasMBury/dl_discrete_bifurcation.git
```

- Navigate to the repository. Create a virtual environment
```
python -m venv venv
source venv/bin/activate
```

- Install package dependencies
```
pip install --upgrade pip
pip install -r requirements.txt
```

- Remove all files in `/output` and `/results` directories
```
cd code
./remove_output.sh
```

- Decide upon a reproducible run from scratch (4.45 hours on a NVIDIA P100 Pascal GPU) or using the pretrained classifier (XX hours). The parameters to set are in the shell script ```run.sh```. To check that the code is working in your environment, you can do a quick run by setting ```QUICK_RUN=true``` (10 minutes). 

To run using the pretrained classifier, set 
```
GEN_TRAINING_DATA=false
TRAIN_CLASSIFIER=false
```
To run from scratch, set
```
GEN_TRAINING_DATA=true
TRAIN_CLASSIFIER=true
```

- The results are sent to the ```/results``` directory.


