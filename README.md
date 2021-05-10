# CS598FP
CS 598 Final Project repository

Author:
Sean Kim (kwkim4@illinois.edu)
Xiang Liu (xiang14@illinois.edu)
Matt Iammarino (mi11@illinois.edu)
Virender Dhiman (vdhiman2@illinois.edu)


1) Directory structure 

Project root: /
Data: /model_data & /data
Profiler and Statistics: /nnperf
Trained models: /pt_files
Captured KPI data: /csv

pyunzip.py
C19_model.py
train.py
main.py
showstats.py

run.sh
run_showstats.sh

2) Download and prepare the dataset
  2.1 Download the data file from COVID-19 Radiography Database link https://www.kaggle.com/tawsifurrahman/covid19-radiography-database
  2.2 Using pyunzip.py or other methods to unzip the file
  2.3 Create a folder named "data" under the root directory of the project folder
  2.4 Move the unzipped folder "COVID-19_Radiography_Dataset" into the "data" folder.
  2.5 You are ready to go.

3) Run the train.py to train the models listed in C19_model.py.
  3.1 train.py will first create a folder "model_data" and allocate the data in "COVID-19_Radiography_Dataset" to train/val/test folder respectively based on 8:1:1 ratio.
  3.2 After the training:
      3.2.1 Trained models will be saved to "pt_files" folder
      3.2.2 Validation performance data will be saved to "csv" folder

4) Run run.sh to evaluate models and capture KPI data
  4.1 KPI data will be saved to "csv" folder
