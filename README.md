# CS598FP
## CS 598 Final Project repository

##### 1) Author:
- Sean Kim (kwkim4@illinois.edu)
- Xiang Liu (xiang14@illinois.edu)
- Matt Iammarino (mi11@illinois.edu)
- Virender Dhiman (vdhiman2@illinois.edu)


##### 2) Directory structure 

   > - Project root: /
   > - Data: /model_data & /data
   > - Profiler and Statistics: /nnperf
   > - Trained models: /pt_files
   > - Captured KPI data: /csv

   - pyunzip.py
   - C19_model.py
   - train.py
   - main.py
   - showstats.py
   >
   - run.sh
   - run_showstats.sh

##### 3) Download and prepare the dataset
   - Download the data file from COVID-19 Radiography Database link https://www.kaggle.com/tawsifurrahman/covid19-radiography-database
   - Using pyunzip.py or other methods to unzip the file
   - Create a folder named "data" under the root directory of the project folder
   - Move the unzipped folder "COVID-19_Radiography_Dataset" into the "data" folder.
   - You are ready to go.

##### 4) Run the train.py to train the models listed in C19_model.py.
   - train.py first creates a folder "model_data" and allocates the data in "COVID-19_Radiography_Dataset" to train/val/test folder respectively based on 8:1:1 ratio.
   - After the training:
     - Trained models are saved to "pt_files" folder
     - Validation performance data are saved to "csv" folder

##### 5) Run run.sh to evaluate models and capture KPI data
   - run.sh will call main.py with a GPU flag and then call main.py without a GPU flag
   - main.py will run through the evaluation framework and output resource usage statistics and accuracy scores for every model given
   - KPI data are saved to "csv" folder


##### 6) Citations
   - Our profiler we created was an extension from the source code (https://github.com/awwong1/torchprof/blob/master/torchprof/profile.py)
   - Our custom dataset references code from https://medium.com/analytics-vidhya/detecting-covid-19-using-chest-x-ray-images-a6fc822b73cc##
