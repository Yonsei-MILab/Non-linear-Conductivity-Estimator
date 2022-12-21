import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader
from scipy import io
import math
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.optim as optim
import os
import pdb

# specify which GPU(s) to be used
device = torch.device('cuda:0')

# Load training dataset
#dataset = io.loadmat('Data/training_dataset.mat')
training_syn_dataset = torch.tensor(dataset['dataset_training'])
training_syn_dataset = np.expand_dims(training_syn_dataset,1)
training_syn_dataset = torch.tensor(training_syn_dataset)

training_mask = torch.tensor(dataset['dataset_mask'])
training_mask = np.expand_dims(training_mask,1)
training_mask = torch.tensor(training_mask)


training_label = torch.tensor(dataset['dataset_label'])
training_label = np.expand_dims(training_label,1)
training_label = torch.tensor(training_label)


kernel_h, kernel_w = 11, 11
step, n_channels = 1, 1
training_syn_dataset_patch = np.array(training_syn_dataset.unfold(2, kernel_h, step).unfold(3, kernel_w, step).permute(2, 3, 0, 1, 4, 5).reshape(-1, n_channels, kernel_h, kernel_w))

training_mask_patch = np.array(training_mask.unfold(2, kernel_h, step).unfold(3, kernel_w, step).permute(2, 3, 0, 1, 4, 5).reshape(-1, n_channels, kernel_h, kernel_w))

training_label_patch = training_label.unfold(2, kernel_h, step).unfold(3, kernel_w, step).permute(2, 3, 0, 1, 4, 5).reshape(-1, n_channels, kernel_h, kernel_w)
training_label_patch = np.array(training_label_patch[:,:,5,5])


from numba import jit

@jit(nopython=True)        
def add_noise(x):
    noise = np.random.normal(0,0.013,x.shape)
    return x+noise

@jit(nopython=True)        
def add_bias(x, nor_factor):
    bias = nor_factor*np.abs(x+(0.1*np.random.rand(1)))*np.exp(1j*np.angle(x))
    return bias

@jit(nopython=True)        
def add_noise_to_complex(x,y):
    z = (add_noise(np.real(x))+1j*add_noise(np.imag(x)))*y
    return z    

@jit(nopython=True)        
def combine_final(x,y):
    nor_factor = (1.3-0.7)*np.random.rand(1)+ 0.7
    xx = add_noise_to_complex(add_bias(x,nor_factor), y)
    #xx = add_noise_to_complex(x, y)
    z = np.concatenate(((np.abs(xx)/nor_factor)*y, ((np.angle(xx)) - np.min((np.angle(xx))))*y),0)
    return z    


class Custom_dataset(Dataset):
    #Dataset = int(str(Dataset), 16)
    #@jit(nopython=True)         
    def __init__(self,training_syn_dataset_patch):
        self.training_syn_dataset_patch = (training_syn_dataset_patch)
        self.training_mask_patch = (training_mask_patch)

    
      
#        return output   # => [3416448, 1, 11, 11]
    def __len__(self):
        
   
        return len(self.training_syn_dataset_patch)
    
               
    def __getitem__(self,idx):
        x = combine_final(self.training_syn_dataset_patch[idx], training_mask_patch[idx])
        y = training_label_patch[idx] 
        return x,y


dataset = Custom_dataset(training_syn_dataset_patch)
dataloader = DataLoader(dataset,batch_size= 100000,shuffle=True)

# Network
class EPT_Model(nn.Module):
    def __init__(self):
        super(EPT_Model, self).__init__()     
        self.fc = nn.Sequential(
            nn.Linear(242, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.2)
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.2)
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.2)
            nn.Linear(512, 1),
            nn.Sigmoid()
        )
            
    def forward(self,x):
        return self.fc(x.view(-1,242))
    
model = EPT_Model()
print(model)

# loss function
criterion = nn.MSELoss()

# optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

# Network Training

loss_function = []
dir_checkpoint = 'Training_loss/'

for epoch in tqdm(range(max_epochs)):
    total_loss = 0
    model.train()
    
    for x,y in dataloader:
        optimizer.zero_grad()
        output = model(x.to(device).float())
        loss = criterion(output,y.to(device).float())
        loss.backward()
        optimizer.step()
        total_loss = total_loss + loss.item()*x.size(0)/len(dataloader.dataset)
    loss_function.append(total_loss)
    
    if (epoch+1) % 10 == 0:
        try:
            os.mkdir(dir_checkpoint)
        except OSError:
            pass
        torch.save(model.state_dict(), dir_checkpoint + f'CP_ANN_epoch{epoch + 1}_MSE.pth')
