#!/bin/bash

python="/usr/local/anaconda3/bin/python3"

$python ./showstats.py csv/ResNet_CPU_KPI.csv
$python ./showstats.py csv/mobilenet_CPU_KPI.csv
$python ./showstats.py csv/resnet18_GPU_KPI.csv
$python ./showstats.py csv/vgg19_CPU_KPI.csv
$python ./showstats.py csv/vgg19_GPU_KPI.csv
$python ./showstats.py csv/alexnet_GPU_KPI.csv
$python ./showstats.py csv/alexnet_CPU_KPI.csv
$python ./showstats.py csv/googlenet_GPU_KPI.csv
$python ./showstats.py csv/googlenet_CPU_KPI.csv
$python ./showstats.py csv/mobilenet_GPU_KPI.csv
$python ./showstats.py csv/resnet50_CPU_KPI.csv
$python ./showstats.py csv/squeezenet_GPU_KPI.csv
$python ./showstats.py csv/squeezenet_CPU_KPI.csv
$python ./showstats.py csv/densenet_CPU_KPI.csv
$python ./showstats.py csv/densenet_GPU_KPI.csv
$python ./showstats.py csv/resnet18_CPU_KPI.csv
$python ./showstats.py csv/resnet50_GPU_KPI.csv
