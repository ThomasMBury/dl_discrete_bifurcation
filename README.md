# Predicting discrete-time bifurcations with deep learning

<!-- ![alt text](/code/figure_1/figure_1.png) -->
<img src="/code/figure_1/figure_1.png"  width="500">

This repository accompanies the article:

***Predicting discrete-time bifurcations with deep learning***. *Thomas M. Bury, Daniel Dylewsky, Chris Bauch, Madhur Anand, Leon Glass, Alvin Shrier, Gil Bub.*

The [code capsule](https://codeocean.com/capsule/3359094/tree) on Code Ocean provides a software environment and compute resources to do a reproducible run of the results reported in the paper. Alternatively, you may do a reproducible run on your local computer following the instructions below.

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

- Decide upon a reproducible run from scratch (4 h 46 mins on a NVIDIA P100 Pascal GPU) or using the pretrained classifier (2 h 34 mins). The parameters to set are in the shell script ```run.sh```. To check that the code is working in your environment, you can do a quick run by setting ```QUICK_RUN=true``` (11 mins). 

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

- Results are saved in the ```/results``` directory.


