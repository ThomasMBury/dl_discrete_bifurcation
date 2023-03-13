# Predicting discrete-time bifurcations with deep learning

<!-- ![alt text](/code/figure_1/figure_1.png) -->
<img src="/code/figure_1/figure_1.png"  width="500">

The above shows the spontaneous beating of an aggregate of chick heart cells. The time between beats (inter-beat interval, IBI) can be regular one minute (blue) and alternating the next (green). This sudden change in dynamics is due to a period-doubling bifurcation. Can deep learning help us to predict these types of bifurcation, which are pervasive in fields ranging from physiology to economics? We address this question in the article

***Predicting discrete-time bifurcations with deep learning***. *Thomas M. Bury, Daniel Dylewsky, Chris Bauch, Madhur Anand, Leon Glass, Alvin Shrier, Gil Bub.*

This is the accompanying code repository. The [code capsule](https://codeocean.com/capsule/3359094/tree) on Code Ocean provides a software environment and compute resources to do a reproducible run of the results reported in the paper. Alternatively, you may do a reproducible run on your local computer following the instructions below.

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

A simple example of applying the deep learning classifier to data is given in ```code/test_chick_heart/example.py```. In brief:

- Clone the repository and install package dependencies as above.
- In a Python script, import the individual classifiers

  ```python
  from tensorflow.keras.models import load_model
  m1 = load_model('data/classifier_1.pkl')
  m2 = load_model('data/classifier_2.pkl')
  ```
- Create an ```ewstools.TimeSeries``` object using your data and an estimated transition time

  ```python
  import ewstools
  ts = ewstools.TimeSeries(data=data, transition=300)
  ```
  
- Detrend if necessary, e.g.
  ```python
  ts.detrend(method='Guassian', bandwidth=20)
  ```
  
- Get predictions from the classifiers at successive points in the data
  ```python
  ts.apply_classifier_inc(m1, inc=10, name='m1', verbose=0)
  ts.apply_classifier_inc(m2, inc=10, name='m2', verbose=0)
  ```
  
 - Plot results using the ensemble average of the classifiers
   ```python
   ts.make_plotly(ens_avg=True)
   ```
  <img src="/code/test_chick_heart/output/example.png"  width="500">
  
The different classes of the classifier predictions correspond to
  - 0 : null trajectory (no approaching bifurcation)
  - 1 : period-doubling bifurcation
  - 2 : Neimark-Sacker bifurcation
  - 3 : fold bifurcation
  - 4 : transcritical bifurcation
  - 5 : pitchfork bifurcation
  
In the above example, the classifier correctly predicts a period-doubling bifurcation.
  
For more details and tutorials on comuting these early warning signals in Python, check out the [ewstools](https://github.com/ThomasMBury/ewstools) repository.


