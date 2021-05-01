#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 30 15:04:58 2021

@author: mattiammarino
"""
import matplotlib.pyplot as plt

import pandas as pd
import seaborn as sns

dataAcc = pd.read_csv('TestScores_CPU.csv')
dataAcc.columns = ['Model', 'Accuracy', 'F1', 'Calls']

ax = sns.barplot(x = "Accuracy", y = "Model", palette = 'bright', data = dataAcc)

# displaying the title
plt.title(label="Accuracy Per Model",
          fontsize=20,
          color="black")

for i, p in enumerate(ax.patches):
    ax.annotate("%.2f" % (p.get_width()),
                (p.get_x() + p.get_width() - 0.1, p.get_y() +0.75),
                xytext=(5, 10), textcoords='offset points', color='white')

plt.tight_layout()
plt.savefig('Accuracy.png', dpi=100)