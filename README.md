# Predicting discrete-time bifurcations with deep learning

This is the code repository to accompnay the article:

***Predicting discrete-time bifurcations with deep learning***. *Thomas M. Bury, Daniel Dylewsky, Chris Bauch, Madhur Anand, Leon Glass, Alvin Shrier, Gil Bub.* 
https://doi.org/10.48550/arXiv.2303.09669

- [Overview](#overview)
- [Code capsule](#code-capsule)
- [System Requirements](#system-requirements)
- [Instructions to reproduce results](#instructions-to-reproduce-results)
- [Instructions to apply to your data](#instructions-to-apply-the-deep-learning-classifier-to-your-own-data)
- [License](#license)
- [Issues](https://github.com/thomasmbury/dl_discrete_bifurcation/issues)

## Overview

<!-- ![alt text](/code/figure_1/figure_1.png) -->
<img src="/code/figure_1/figure_1.png"  width="500">

The above figure shows the spontaneous beating of an aggregate of chick heart cells. The time between beats (inter-beat interval, IBI) can be regular one minute (blue) and alternating the next (green). This sudden change in dynamics is due to a type of discrete-time bifurcation known as a period-doubling bifurcation. Can deep learning help us to predict these types of bifurcation, which are present in fields ranging from physiology to economics? This is the question we address in the article.


## Code capsule

The article is accompanied by a [code capsule](https://codeocean.com/capsule/3359094/tree) on Code Ocean, which provides a software environment and compute resources to do a reproducible run of the results reported in the paper. This circumnavigates the need to install the software environment yourslef to reproduce the results. Alternatively, you can do a reproducible run on your local machine following the instructions below.


## System Requirements
### Hardware requirements
Training of the deep learning classifier requires access to a GPU. All other operations in the code can be run on a standard machine with enough RAM to support in-memory operations. The code has been tested on both
- MacBook Air (M1, 2020) with  8-core GPU, 8GB memory, 512GB storage
- Intel E5-2650 v4 Broadwell @ 2.2GHz with NVIDIA P100 Pascal GPU

### Software requirements
#### OS Requirements
The code is supported for both *macOS* and *Linux*. It has been tested on the following systems:
- macOS: Monterey (12.2.1)
- Linux: CentOS 7


#### Python Dependencies
The code has been tested with Python 3.10 and depends primarily on the following Python scientific stack.

```
numpy
pandas
scikit-learn
tensorflow
ewstools
pyarrow
plotly
```


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

- Decide between doing a reproducible run from scratch (4 h 46 mins on a NVIDIA P100 Pascal GPU) or using the pretrained classifier (2 h 34 mins). To run from scratch, open the shell script ```code/run.sh``` and set the parameters
  ```
  GEN_TRAINING_DATA=true
  TRAIN_CLASSIFIER=true
  QUICK_RUN=false
  ```
  and run it using the command
  ```
  ./run.sh
  ```

- To use the pretrained classifier, set 
  ```
  GEN_TRAINING_DATA=false
  TRAIN_CLASSIFIER=false
  QUICK_RUN=false
  ```
  and run it using the command
  ```
  ./run.sh
  ```
  
- Note that in either case, you can set ```QUICK_RUN=true```, which performs a quick run of the code (11 mins) using parameters that minimise computation. This is useful to check the code is working in your environment, before doing a full reproducible run.

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
  
- Get incremental predictions from the classifiers
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




## License

This project is covered under the **MIT License**.

