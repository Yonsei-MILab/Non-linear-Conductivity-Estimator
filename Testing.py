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

# Load in-vivo test data
#dataset = io.loadmat('Data/training_dataset.mat')
feasibility = torch.tensor(dataset['dataset_training'])
feasibility_1 = torch.tensor(feasibility.clone())
feasibility_extend = np.expand_dims(feasibility_1,1)
feasibility_extend_1 = torch.tensor(feasibility_extend)

training_mask = torch.tensor(dataset['dataset_mask'])
training_mask_1 = torch.tensor(training_mask.clone())
training_mask_extend = np.expand_dims(training_mask_1,1)
training_mask_extend_1 = torch.tensor(training_mask_extend)


training_label = torch.tensor(dataset['dataset_label'])
training_label_1 = torch.tensor(training_label.clone())
training_label_extend = np.expand_dims(training_label_1,1)
training_label_extend_1 = torch.tensor(training_label_extend)


kernel_h, kernel_w = 11, 11
step, n_channels = 1, 1
feasibility_patch_1 = feasibility_extend_1.unfold(2, kernel_h, step).unfold(3, kernel_w, step).permute(2, 3, 0, 1, 4, 5).reshape(-1, n_channels, kernel_h, kernel_w)

mask_patch_1 = training_mask_extend_1.unfold(2, kernel_h, step).unfold(3, kernel_w, step).permute(2, 3, 0, 1, 4, 5).reshape(-1, n_channels, kernel_h, kernel_w)

label_patch_1 = training_label_extend_1.unfold(2, kernel_h, step).unfold(3, kernel_w, step).permute(2, 3, 0, 1, 4, 5).reshape(-1, n_channels, kernel_h, kernel_w)
label_patch_1 = label_patch_1[:,:,5,5]

label_patch = np.array(label_patch_1)
mask_patch = np.array(mask_patch_1)
feasibility_patch = np.array(feasibility_patch_1)


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
    def __init__(self,feasibility_patch):
        self.feasibility_patch = (feasibility_patch)
        self.mask_patch = (mask_patch)

    
      
#        return output   # => [3416448, 1, 11, 11]
    def __len__(self):
        
   
        return len(self.feasibility_patch)
    
               
    def __getitem__(self,idx):
        x = combine_final(self.feasibility_patch[idx], mask_patch[idx])
        y = label_patch[idx] 
        return x,y


dataset = Custom_dataset(feasibility_patch)
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
