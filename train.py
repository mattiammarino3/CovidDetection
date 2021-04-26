#python3.8 train.py

### Importing Libraries ###
import C19_model
from nnperf.kpiprofile import Profile
from nnperf.stats import nnPerf

import os
import shutil
import random
import torch
import torchvision
import numpy as np
import torch.nn as nn
import splitfolders
from sklearn.metrics import *

from PIL import Image
from matplotlib import pyplot as plt

torch.manual_seed(0)

print('Using PyTorch version', torch.__version__)

### Preparing Training and Test Sets ###
directory = './data/COVID-19_Radiography_Dataset'

source_dirs = ['Normal', 'Viral Pneumonia', 'COVID','Lung_Opacity']
train_dir = './model_data/train'
val_dir = './model_data/val'
test_dir = './model_data/test'

if not os.path.isdir("model_data"):
    splitfolders.ratio('./data/COVID-19_Radiography_Dataset', output="model_data", seed=1337, ratio=(.8, 0.1,0.1)) 



### Creating Custom Dataset ###
class ChestXRayDataset(torch.utils.data.Dataset):
    def __init__(self, image_dirs, transform):
        def get_images(class_name):
            images = [x for x in os.listdir(image_dirs[class_name]) if x.lower().endswith('png')]
            print(f'Found {len(images)} {class_name} examples')
            return images
        
        self.images = {}
        self.class_names = ['Normal', 'Viral', 'COVID-19','Lung_Opacity']
        
        for class_name in self.class_names:
            self.images[class_name] = get_images(class_name)
            
        self.image_dirs = image_dirs
        self.transform = transform
        
    
    def __len__(self):
        return sum([len(self.images[class_name]) for class_name in self.class_names])
    
    
    def __getitem__(self, index):
        class_name = random.choice(self.class_names)
        index = index % len(self.images[class_name])
        image_name = self.images[class_name][index]
        image_path = os.path.join(self.image_dirs[class_name], image_name)
        image = Image.open(image_path).convert('RGB')
        return self.transform(image), self.class_names.index(class_name)



### Prepare DataLoader ###
def getDataSet(dataset_dir):
    dataset_dirs = {
        'Normal': dataset_dir + '/Normal',
        'Viral': dataset_dir + '/Viral Pneumonia',
        'COVID-19': dataset_dir + '/COVID',
        'Lung_Opacity': dataset_dir + '/Lung_Opacity'
    }
    data_transform = torchvision.transforms.Compose([
    #Converting images to the size that the model expects
    torchvision.transforms.Resize(size=(224,224)),
    torchvision.transforms.RandomHorizontalFlip(), #A RandomHorizontalFlip to augment our data
    torchvision.transforms.ToTensor(), #Converting to tensor    
    torchvision.transforms.Normalize(mean = [0.485, 0.456, 0.406],
                                    std = [0.229, 0.224, 0.225]) #Normalizing the data using the Imagenet pretrianed model
    
    ])
    return ChestXRayDataset(dataset_dirs, data_transform)


train_dataset = getDataSet(train_dir)
val_dataset = getDataSet(val_dir)
test_dataset = getDataSet(test_dir)

batch_size = 6

dl_train = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
dl_val = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
dl_test = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

print('Num of training batches', len(dl_train))
print('Num of validation batches', len(dl_val))
print('Num of test batches', len(dl_test))

### Data Visualization ###
class_names = train_dataset.class_names

#input: Y_pred,Y_true
#output: accuracy, auc, precision, recall, f1-score
def classification_metrics(Y_pred, Y_true):
    return [accuracy_score(Y_true, Y_pred),
            roc_auc_score(Y_true, Y_pred),
            precision_score(Y_true, Y_pred),
            recall_score(Y_true, Y_pred),
            f1_score(Y_true, Y_pred)]

def show_images(images,labels, preds):
    plt.figure(figsize=(8,4))
    
    for i, image in enumerate(images):
        plt.subplot(1, 6, i + 1, xticks = [], yticks =[]) # x & y ticks are set to blank
        image = image.numpy().transpose((1, 2, 0)) # Channel first then height and width
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = image * std + mean
        image = np.clip(image, 0., 1.)
        plt.imshow(image)
        
        col = 'green' if preds[i] == labels[i] else 'red'
        
        plt.xlabel(f'{class_names[int(labels[i].numpy())]}')
        plt.ylabel(f'{class_names[int(preds[i].numpy())]}', color=col)
    plt.tight_layout()
    plt.show()

# images, labels = next(iter(dl_train)) #Fetch the next batch of images
# show_images(images, labels, labels)

# images, labels = next(iter(dl_val))
# show_images(images, labels, labels)

# images, labels = next(iter(dl_test))
# show_images(images, labels, labels)


### Creating the Model ###
C_models = [
          #('resnet18', C19_model.resnet18()),
          #('densenet', C19_model.densenet()),
          ('resnet50', C19_model.resnet50()),
          ('alexnet', C19_model.alexnet()),
          ('mobilenet', C19_model.mobilenet()),
          ('googlenet', C19_model.googlenet()),
          ('vgg19', C19_model.vgg19())
        ]

def show_preds():
    model.eval()  #Setting the model to evaluation mode
    images, labels = next(iter(dl_test))

    with Profile(model, use_cuda=False, profile_memory=True) as prof:
        outputs = model(images)
        
    if torch.cuda.device_count() > 1:
        outputs = outputs.cpu()

    _, preds=torch.max(outputs, 1)
    #print(prof.display(show_events=False))
    prof.getKPIData()

    show_images(images, labels, preds)
    acc, auc_, precision, recall, f1score = classification_metrics(preds,labels)


### Training the Model ###
def train(epochs, name):
    print('*'*40)
    print('Starting training model: ' + name + ' ...')
    print('*'*40)
    global model
    n_epochs_stop = 3
    min_val_loss = np.Inf
    epochs_no_improve = 0

    for e in range(0, epochs):
        print('='*20)
        print(f'Starting epoch {e + 1}/{epochs}')
        print('='*20)

        train_loss = 0.
        val_loss = 0.  #Not computing val_loss since we'll be evaluating the model multiple times within one epoch
        
        model.train() # set model to training phase
        
        for train_step, (images, labels) in enumerate(dl_train):
            optimizer.zero_grad()
            if torch.cuda.device_count() > 0:
                outputs = model(images.to(device))
                labels = labels.to(device)
            else:                
                outputs = model(images)
            loss = loss_fn(outputs, labels)
            #Once we get the loss we need to take a gradient step
            loss.backward() #Back propogation
            optimizer.step() #Completes the gradient step by updating all the parameter values(We are using all parameters)
            train_loss += loss.item() #Loss is a tensor which can't be added to train_loss so .item() converts it to float
            
        train_loss /= (train_step + 1)
        print(f'Training Loss: {train_loss:.4f}')

        #Evaluating the model at the end of epoch
        print(f'Evaluating at epoch {e + 1}/{epochs}')
        #accuracy = 0
        y_true = []
        y_pred = []
        model.eval() # set model to eval phase
        for val_step, (images, labels) in enumerate(dl_val):
            if torch.cuda.device_count() > 0:
                outputs = model(images.to(device))
                labels = labels.to(device)
            else:
                outputs = model(images)

            loss = loss_fn(outputs, labels)
            val_loss += loss.item()
            if torch.cuda.device_count() > 0:
                outputs = outputs.cpu()
                labels = labels.cpu()

            _, preds = torch.max(outputs, 1) # 1 corresponds to the values and ) corresponds to the no of examples
            #accuracy += sum((preds == labels).numpy()) #adding correct preds to acc
            y_true = np.append(y_true, labels.numpy())
            y_pred = np.append(y_pred, preds.numpy())

        val_loss /= (val_step + 1) # 15 test batches so this logic gives the value for each step
        #accuracy = accuracy/len(test_dataset)
        print(f'Validation Loss: {val_loss:.4f}')
        f1_val = f1_score(y_true, y_pred, average='weighted')
        accuracy_val = accuracy_score(y_true, y_pred)
        print(f"F1 Score: {f1_val:.4f}, Accuracy: {accuracy_val :.4f}")
        C_nnPerf = nnPerf()
        C_nnPerf.saveAccToCSV(csv_filepath=val_file, epoch=e, accuracy=accuracy_val, f1score=f1_val)

        # If the validation loss is at a minimum
        if val_loss < min_val_loss:
            # Save the model
            torch.save(model, "pt_files/"+name+".pt")
            epochs_no_improve = 0
            min_val_loss = val_loss
        else:
            epochs_no_improve += 1

            # Check early stopping condition
            if epochs_no_improve == n_epochs_stop:
                print('Training stopped early ...')
                return

        model.train()    
            
    print('Training completed ...')

for name, c_model in C_models:
    model = c_model.get_model()
    loss_fn = c_model.get_loss()
    optimizer = c_model.get_optimizer()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        model = nn.DataParallel(model)

    model.to(device)

    val_file = 'csv/Validation_' + name + '.csv'
    
    open(val_file, "w")

    train(epochs=10, name=name)

    # torch.save(model, name + '.pt')
    print('='*20)
    print("Model: ", name , " was saved.")
    print('='*20)