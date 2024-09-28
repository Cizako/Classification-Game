import torch
import torch.nn as nn
import torch.optim as optim
import os
import json
from models.GoalNet import GoalNet

class Ensemble(nn.Module):
    def __init__(self, PATHS, soft_on = True): 
        super(Ensemble, self).__init__()
        self.soft_on = soft_on
        #self.model1 = GoalNet()
        #self.model2 = GoalNet()
        #self.model3 = GoalNet()
        #self.model1.load_state_dict(torch.load(PATHS[0], weights_only = True))
        #self.model2.load_state_dict(torch.load(PATHS[1], weights_only = True))
        #self.model3.load_state_dict(torch.load(PATHS[2], weights_only = True))
        #self.model = GoalNet()

        self.models=[]
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
        for idx,PATH in enumerate(PATHS):
            with open(PATH, 'r') as f:
                config = json.load(f)
            

            saved_weights = f'saved_models/{config["model_info"]["model_name"]}_{config["model_info"]["unique_id"]}_final'
            use_batchnorm = config["tp"].get("Batch_norm")
            
            if use_batchnorm is None:
                use_batchnorm = False
            print(use_batchnorm)
            model = GoalNet(bn=use_batchnorm).to(device)
            model.load_state_dict(torch.load(saved_weights, weights_only=True, map_location=device)) 



            self.models.append(model)  
    def forward(self, x):

        #pred1 = self.models[0](x, self.soft_on)
        #pred2 = self.models[1](x, self.soft_on)
        #pred3 = self.models[2](x, self.soft_on)
        # pred = (pred1+pred2+pred3) / 3

        device = next(self.models[0].parameters()).device  # Hämta enhet från första modellens parametrar
        x = x.to(device)

        tot_predict = torch.zeros_like(self.models[0](x,self.soft_on))
        for pred in self.models:
            pred = pred(x,self.soft_on)
            tot_predict += pred
        if self.soft_on:
            return tot_predict / len(self.models)
        else:
            softmax = nn.LogSoftmax(1)
            return softmax(tot_predict / len(self.models))

