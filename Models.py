# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 14:01:42 2024

@author: atifr
"""
import torch
from torch import nn
import torch.nn.functional as F

import torchvision.models as models
def model_size_in_bits(model: torch.nn.Module) -> int:
    total_bits = 0
    
    for param in model.parameters():
        total_bits += param.numel() * param.element_size() * 8  # numel gives the number of elements, element_size gives the size of each element in bytes        
    return total_bits,


    

class AlexNetCMnist(nn.Module):
    def __init__(self, width_mult=1):
        super(AlexNetCMnist, self).__init__()
        self.layer1 = nn.Sequential( # Input 1*28*28
            nn.Conv2d(1, 32, kernel_size=3, padding=1), # 32*28*28
            nn.MaxPool2d(kernel_size=2, stride=2), # 32*14*14
            nn.ReLU(inplace=True),
            )
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1), # 64*14*14
            nn.MaxPool2d(kernel_size=2, stride=2), # 64*7*7
            nn.ReLU(inplace=True),
            )
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1), # 128*7*7
            )
        
        self.layer4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1), # 256*7*7
            )

        #AUX 
        self.layer5 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=2), # 256*3*3
            nn.ReLU(inplace=True)
            )  
        self.fc1 = nn.Linear(256*3*3, 10)
 
    def forward(self, x):
        reps = self.layer1(x)
        reps = self.layer2(reps)
        reps = self.layer3(reps)
        reps = self.layer4(reps)
        out = self.layer5(reps)
        out = out.view(-1, 256*3*3)
        out = self.fc1(out)
        return out, reps

class AlexNetSMnist(nn.Module):
    def __init__(self, width_mult=1):
        super(AlexNetSMnist, self).__init__()
        self.layer5 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1), # 256*7*7
            nn.MaxPool2d(kernel_size=3, stride=2), # 256*3*3
            nn.ReLU(inplace=True),
            )
        self.fc1 = nn.Linear(256*3*3, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 10)
 
    def forward(self, x):
        reps = self.layer5(x)
        out = reps.view(-1, 256*3*3)
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.fc3(out)
        return out, reps



    
class AlexNetCifarC(nn.Module):
    def __init__(self, c=10, width_mult=1):
        super(AlexNetCifarC, self).__init__()
        self.layer1 = nn.Sequential(  # Input: 3*32*32
            nn.Conv2d(3, 32, kernel_size=3, padding=1),  # 32*32*32
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),       # 32*16*16
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1), # 64*16*16
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),       # 64*8*8
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1), # 128*8*8
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1), # 256*8*8
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        
        self.exit_fc = nn.Linear(256*8*8, c)  # Early exit at 256*8*8

    def forward(self, x):
        reps = self.layer1(x)
        reps = self.layer2(reps)
        reps = self.layer3(reps)
        reps = self.layer4(reps)
        exit_out = reps.view(-1, 256*8*8)
        exit_out = self.exit_fc(exit_out)
        return exit_out, reps
    
class AlexNetCifarS(nn.Module):
    def __init__(self, c=1, width_mult=1):
        super(AlexNetCifarS, self).__init__()
        self.layer5 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1), # 256*7*7
            nn.MaxPool2d(kernel_size=3, stride=2), # 256*3*3
            nn.ReLU(inplace=True),
            )
        self.fc1 = nn.Linear(256*3*3, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, c)
 
    def forward(self, x):
        reps = self.layer5(x)
        out = reps.view(-1, 256*3*3)
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.fc3(out)
        return out, reps

class ClientModel_cifar100(nn.Module):
    def __init__(self):
        super(ClientModel_cifar100, self).__init__()
        base = models.resnet50(pretrained=False)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = base.bn1
        self.relu = base.relu
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.early_exit = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(1024, 100)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        early_logits = self.early_exit(x)
        return early_logits, x

# Server side model (layer4 + final classifier)
class ServerModel_cifar100(nn.Module):
    def __init__(self):
        super(ServerModel_cifar100, self).__init__()
        base = models.resnet50(pretrained=False)
        self.layer4 = base.layer4
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(2048, 100)

    def forward(self, x):
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        final_logits = self.fc(x)
        return final_logits, ''


def select_model(model_name):
    if model_name == 'cifar' :
        client_model = AlexNetCifarC(10)
        server_model = AlexNetCifarS(10)
    
    if model_name == 'cifar100':
        client_model = ClientModel_cifar100()
        server_model = ServerModel_cifar100()
        
    if model_name == 'mnist':
        client_model = AlexNetCMnist()
        server_model = AlexNetSMnist()
        
    return client_model, server_model

if __name__ == '__main__':
    print(model_size_in_bits(AlexNetCifarS()))