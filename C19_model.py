import os
import shutil
import random
import torch
import torchvision
import numpy as np
import torch.nn as nn

from PIL import Image
from matplotlib import pyplot as plt

# abstract model with methods to access model, loss function, and optimizer defined in _init_
class DLH_model(object):
   def __init__(self):
      self.model = null
      self.loss = null
      self.optimizer = null
   # Access model
   def get_model(self):
      return self.model
   # Access loss function
   def get_loss(self):
      return self.loss
   # Access optimizer
   def get_optimizer(self):
      return self.optimizer

#individual class for each model to define pre-trained model, parameters, loss function, and optimizer in _init_
class resnet18(DLH_model):
   def __init__(self):
      self.model = torchvision.models.resnet18(pretrained=True)
      #Changing the last layer to 4 output features
      self.model.fc = torch.nn.Linear(in_features=512, out_features=4)
      
      self.loss = torch.nn.CrossEntropyLoss()
      self.optimizer = torch.optim.Adam(self.model.parameters(), lr=3e-5)

#individual class for each model to define pre-trained model, parameters, loss function, and optimizer in _init_
class densenet(DLH_model):
   def __init__(self):
      self.model = torchvision.models.densenet121(pretrained=True)
      #Changing the last layer to 4 output features
      self.model.classifier = nn.Linear(in_features=1024, out_features=4)
      
      self.loss = torch.nn.CrossEntropyLoss()
      self.optimizer = torch.optim.Adam(self.model.parameters(), lr=3e-5)

#individual class for each model to define pre-trained model, parameters, loss function, and optimizer in _init_
class resnet50(DLH_model):
   def __init__(self):
      self.model = torchvision.models.resnet50(pretrained=True)
      #Changing the last layer to 4 output features
      self.model.fc = nn.Sequential(
         nn.Linear(2048, 64),
         nn.ReLU(inplace=True),
         nn.Linear(64, 4)
      )
      
      self.loss = torch.nn.CrossEntropyLoss()
      self.optimizer = torch.optim.Adam(self.model.parameters(), lr=3e-5)

#individual class for each model to define pre-trained model, parameters, loss function, and optimizer in _init_
class alexnet(DLH_model):
   def __init__(self):
      self.model = torchvision.models.alexnet(pretrained=True)
      #Changing the last layer to 4 output features
      self.model.classifier = nn.Sequential(
             nn.Dropout(p=0.5, inplace=False),
             nn.Linear(in_features=9216, out_features=4096, bias=True),
             nn.ReLU(inplace=True),
             nn.Dropout(p=0.5, inplace=False),
             nn.Linear(in_features=4096, out_features=1024, bias=True),
             nn.ReLU(inplace=True),
             nn.Linear(in_features=1024, out_features=4, bias=True)
        )
      
      self.loss = torch.nn.CrossEntropyLoss()
      self.optimizer = torch.optim.Adam(self.model.parameters(), lr=3e-5)
	 
#individual class for each model to define pre-trained model, parameters, loss function, and optimizer in _init_
class squeezenet(DLH_model):
   def __init__(self):
      self.model = torchvision.models.squeezenet1_0(pretrained=True)
      #Changing the last layer to 4 output features
      self.model.classifier = nn.Sequential(
		nn.Dropout(p=0.5, inplace=False),
		nn.Conv2d(512, 4, kernel_size=(1, 1), stride=(1, 1)),
		nn.ReLU(inplace=True),
		nn.AdaptiveAvgPool2d(output_size=(1, 1))
		)

      self.loss = torch.nn.CrossEntropyLoss()
      self.optimizer = torch.optim.Adam(self.model.parameters(), lr=3e-5)

#individual class for each model to define pre-trained model, parameters, loss function, and optimizer in _init_
class mobilenet(DLH_model):
   def __init__(self):
      self.model = torchvision.models.mobilenet_v2(pretrained=True)
      #Changing the last layer to 4 output features
      self.model.classifier = nn.Sequential(
          nn.Dropout(p=0.2, inplace=False),
          nn.Linear(in_features=1280, out_features=4, bias=True)
        )

      self.loss = torch.nn.CrossEntropyLoss()
      self.optimizer = torch.optim.Adam(self.model.parameters(), lr=3e-5)

#individual class for each model to define pre-trained model, parameters, loss function, and optimizer in _init_
class googlenet(DLH_model):
   def __init__(self):
      self.model = torchvision.models.googlenet(pretrained=True)
      #Changing the last layer to 4 output features
      self.model.fc = nn.Linear(in_features=1024, out_features=4)

      self.loss = torch.nn.CrossEntropyLoss()
      self.optimizer = torch.optim.Adam(self.model.parameters(), lr=3e-5)

#individual class for each model to define pre-trained model, parameters, loss function, and optimizer in _init_
class vgg19(DLH_model):
   def __init__(self):
      self.model = torchvision.models.vgg19(pretrained=True)
      #Changing the last layer to 4 output features
      self.model.fc = torch.nn.Linear(in_features=4096, out_features=4)
      
      self.loss = torch.nn.CrossEntropyLoss()
      self.optimizer = torch.optim.Adam(self.model.parameters(), lr=3e-5)
