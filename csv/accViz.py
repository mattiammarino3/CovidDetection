#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 30 15:04:58 2021

@author: mattiammarino
"""
import matplotlib.pyplot as plt

import pandas as pd
import seaborn as sns
import numpy as np

dataAcc = pd.read_csv('TestScores_CPU.csv')
dataAcc.columns = ['Model', 'Accuracy', 'F1', 'Calls']
dataAcc = dataAcc.drop(['Calls'], axis=1)
dataAcc = pd.melt(dataAcc, id_vars="Model", var_name="Metric", value_name="Score")

#ax = sns.barplot(x = "Model" , y = ["Accuracy", "F1"], palette = 'bright', data = dataAcc)
sns.catplot(x='Metric', y='Score', hue='Model', palette=sns.color_palette(['blue','green', 'orange', 'red', 'black', 'purple', 'yellow', 'gray']), data=dataAcc, kind='bar')

# displaying the title
plt.title(label="Accuracy/F1 Score Per Model",
          fontsize=16,
          color="black")

plt.subplots_adjust(top=.9)
plt.savefig('Accuracy.png', dpi=100)






classAlex = pd.read_csv('Classification_alexnet_GPU.csv')
classAlex['Model'] = 'AlexNet'
classDense = pd.read_csv('Classification_densenet_GPU.csv')
classDense['Model'] = 'DenseNet'
classResnet18 = pd.read_csv('Classification_resnet18_GPU.csv')
classResnet18['Model'] = 'Resnet18'
classResnet50 = pd.read_csv('Classification_resnet50_GPU.csv')
classResnet50['Model'] = 'Resnet50'
classMobile = pd.read_csv('Classification_mobilenet_GPU.csv')
classMobile['Model'] = 'MobileNet'
classGoogle = pd.read_csv('Classification_googlenet_GPU.csv')
classGoogle['Model'] = 'GoogleNet'
classVGG = pd.read_csv('Classification_vgg19_GPU.csv')
classVGG['Model'] = 'VGG'
classSqueeze = pd.read_csv('Classification_squeezenet_GPU.csv')
classSqueeze['Model'] = 'Squeezenet'

classData = pd.concat([classAlex, classDense, classResnet18, classResnet50, classMobile, classGoogle, classVGG, classSqueeze])
classData.columns = ['Class', 'Precision','Recall' ,'F1', 'Support', 'Model']
classData = pd.melt(classData, id_vars=['Model', 'Class'], var_name='Metric', value_name='Score')

mask = classData['Class'].isin(['Normal', 'COVID-19', 'Lung_Opacity', 'Viral'])
classData = classData[mask]


g = sns.FacetGrid(classData,
            col='Class',
            sharex=False,
            sharey=False,
            height=3.5, col_wrap = 2)
g = g.map(sns.barplot, 'Metric', 'Score', 'Model', 
          hue_order=np.unique(classData['Model']), 
          order=['Precision','Recall' ,'F1'], 
          palette=sns.color_palette(['blue','green', 'orange', 'red', 'black', 'purple', 'yellow', 'gray']))
g.add_legend()



g.axes[2].set_xlabel('')
g.axes[3].set_xlabel('')
g.fig.suptitle("Classification Report By Model and Class", size=16)
g.fig.subplots_adjust(top=.9)

#plt.tight_layout()
plt.savefig("Classifications.png", dpi=100)