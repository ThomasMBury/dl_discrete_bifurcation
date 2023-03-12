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


## Instructions to apply the deep learning classifier to your own data

- Clone the repository and install package dependencies as above.
- In a Python script, import the classifiers

  ```python
  from tensorflow.keras.models import load_model
  m1 = load_model('data/classifier_1.pkl')
  m1 = load_model('data/classifier_2.pkl')
  ```
- Create an ```ewstools.TimeSeries``` object using your data

  ```python
  import ewstools
  ts = ewstools.TimeSeries(data)
  ```
  
- Detrend if necessary, e.g.
  ```python
  ts.detrend(method='Lowess', span=span)
  ```
  
- Get predictions from the classifiers at successive points in the time series
  ```python
  ts.apply_classifier_inc(m1, inc=inc, name='m1', verbose=0)
  ts.apply_classifier_inc(m2, inc=inc, name='m2', verbose=0)
  ```
  
 - Plot results
  ```python
  ts.make_plotly()
  ```
  Figure
  For further details, see the [ewstools](https://github.com/ThomasMBury/ewstools) repository.





