import os
import shutil
import random
import torch
import torchvision
import numpy as np
import torch.nn as nn

from PIL import Image
from matplotlib import pyplot as plt

class DLH_model(object):

   def __init__(self):
      self.model = null
      self.loss = null
      self.optimizer = null

   def get_model(self):
      return self.model

   def get_loss(self):
      return self.loss

   def get_optimizer(self):
      return self.optimizer


class resnet18(DLH_model):
   def __init__(self):
      self.model = torchvision.models.resnet18(pretrained=True)
      #Changing the last fc to 4 output features
      self.model.fc = torch.nn.Linear(in_features=512, out_features=4)
      
      self.loss = torch.nn.CrossEntropyLoss()

      self.optimizer = torch.optim.Adam(self.model.parameters(), lr=3e-5)

class densenet(DLH_model):
   def __init__(self):
      self.model = torchvision.models.densenet121(pretrained=True)
      #Changing the last fc to 4 output features
      self.model.classifier = nn.Linear(in_features=1024, out_features=4)
      
      self.loss = torch.nn.CrossEntropyLoss()

      self.optimizer = torch.optim.Adam(self.model.parameters(), lr=3e-5)
