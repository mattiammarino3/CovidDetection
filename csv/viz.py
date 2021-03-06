#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 26 18:40:48 2021

@author: mattiammarino
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

dataCPU = pd.read_csv('MemoryUsage_CPU.csv')
dataCPU['Architecture'] = 'CPU'
dataCPU['CPU TOTAL TIME'] = dataCPU['CPU TOTAL TIME']/1000
dataCPU['GPU TOTAL TIME'] = dataCPU['GPU TOTAL TIME']/1000
dataCPU['CPU MEM MAX'] = dataCPU['CPU MEM MAX']/1000
dataCPU['GPU MEM MAX'] = dataCPU['GPU MEM MAX']/1000
dataCPU = pd.melt(dataCPU, id_vars=['Model', 'Architecture'], var_name='Metric', value_name='value')

dataGPU = pd.read_csv('MemoryUsage_GPU.csv')
dataGPU['Architecture'] = 'GPU'
dataGPU['CPU TOTAL TIME'] = dataGPU['CPU TOTAL TIME']/1000
dataGPU['GPU TOTAL TIME'] = dataGPU['GPU TOTAL TIME']/1000
dataGPU['CPU MEM MAX'] = dataGPU['CPU MEM MAX']/1000
dataGPU['GPU MEM MAX'] = dataGPU['GPU MEM MAX']/1000
dataGPU = pd.melt(dataGPU, id_vars=['Model', 'Architecture'], var_name='Metric', value_name='value')


data = pd.concat([dataCPU,dataGPU])


g = sns.FacetGrid(data,
            col='Architecture',
            sharex=False,
            sharey=False,
            height=5)
g = g.map(sns.barplot, 'Metric', 'value', 'Model', 
          hue_order=np.unique(data['Model']), 
          order=['CPU TOTAL TIME', 'GPU TOTAL TIME'], 
          palette=sns.color_palette(['blue','green', 'orange', 'red', 'black', 'purple', 'yellow', 'gray']))
g.add_legend()

g.axes[0,0].set_xlabel('Computation')
g.axes[0,1].set_xlabel('Computation')
g.axes[0,0].set_ylabel('Time (s)')
g.axes[0,1].set_ylabel('Time (s)')
g.fig.suptitle('Computation Time By Architecture', size=16)
g.fig.subplots_adjust(top=0.8)
plt.savefig('TimeStats.png', dpi=100)





g = sns.FacetGrid(dataCPU,
            col='Architecture',
            sharex=False,
            sharey=False,
            height=5)
g = g.map(sns.barplot, 'Metric', 'value', 'Model', 
          hue_order=np.unique(dataCPU['Model']), 
          order=['CPU MEM MAX'], 
          palette=sns.color_palette(['blue','green', 'orange', 'red', 'black', 'purple', 'yellow', 'gray']))
g.add_legend()

g.axes[0,0].set_xlabel('')
g.axes[0,0].set_ylabel('Memory (mb)')

g.fig.suptitle('Maxmimum Memory Per Model For CPU Architecture', size=12)
g.fig.subplots_adjust(top=0.8)

plt.savefig('CPUMemStats.png', dpi=100)

g = sns.FacetGrid(dataGPU,
            col='Architecture',
            sharex=False,
            sharey=False,
            height=4)
g = g.map(sns.barplot, 'Metric', 'value', 'Model', 
          hue_order=np.unique(dataGPU['Model']), 
          order=['GPU MEM MAX'], 
          palette=sns.color_palette(['blue','green', 'orange', 'red', 'black', 'purple', 'yellow', 'gray']))
g.add_legend()

g.axes[0,0].set_xlabel('')
g.axes[0,0].set_ylabel('Memory (mb)')

g.fig.suptitle('Maxmimum Memory Per Model For GPU Architecture', size=12)
g.fig.subplots_adjust(top=0.8)

plt.savefig('GPUMemStats.png', dpi=100)

g = sns.FacetGrid(dataCPU,
            col='Architecture',
            sharex=False,
            sharey=False,
            height=5)
g = g.map(sns.barplot, 'Metric', 'value', 'Model', 
          hue_order=np.unique(dataGPU['Model']), 
          order=['PARAMS'], 
          palette=sns.color_palette(['blue','green', 'orange', 'red', 'black', 'purple', 'yellow', 'gray']))
g.add_legend()

g.axes[0,0].set_xlabel('')
g.axes[0,0].set_ylabel('Total Number')

g.fig.suptitle('Number of Parameters Per Model', size=16)
g.fig.subplots_adjust(top=0.8)

plt.savefig('Params.png', dpi=100)





