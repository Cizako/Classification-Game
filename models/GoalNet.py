import os
import torch
import torch.nn as nn
import torch.nn.functional as F

class GoalNet(nn.Module):
    def __init__(self, bn):
        
        super(GoalNet, self).__init__()
       
        self.conv1 = nn.Conv2d(3, 32, 5, padding=2)
        
        self.conv2 = nn.Conv2d(32,32,5,padding=2)
        self.conv3 = nn.Conv2d(32,64,5,padding=2)
        
        self.conv4 = nn.Conv2d(64,64,4)
        if bn:
            self.conv3_bn = nn.BatchNorm2d(64)
            self.conv1_bn = nn.BatchNorm2d(32)
        
        self.use_batchnorm = bn
        self.prediction = nn.Linear(64, 10)
        self.loss = nn.LogSoftmax(1)
               
    def forward(self, x, soft_on = True):
        out = self.conv1(x)
        out = F.max_pool2d(out, kernel_size=(2,2), stride=2) # samma stride som kernel size?
        out = F.relu(out) # relu1
        out = self.conv2(out) # vad ska paddingen vara?
        out = F.relu(out)
        out = F.max_pool2d(out, kernel_size=(2,2), stride=2) # samma stride som kernel size?

        out = self.conv3(out) # vad ska paddingen vara?
        out = F.relu(out)
        out = F.max_pool2d(out, kernel_size=(2,2), stride=2) # samma stride som kernel size?

        out = self.conv4(out) # vad ska paddingen vara?
        if self.use_batchnorm:
            out = self.conv3_bn(out)
        out = F.relu(out) # relu2
        
        #out = out.view(out.size(0), -1)
        out = out.view(-1, 64)  # Flatten the tensor
        out = self.prediction(out)
        
        if soft_on == True:
            out = self.loss(out)

        return out

    


