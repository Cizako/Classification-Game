import torch
import torch.nn as nn
import os
import csv
from models.GradCam import GradCAM, save_grad_cam_heatmap
from torch.optim.lr_scheduler import StepLR
import numpy as np
import matplotlib.pyplot as plt
from livelossplot import PlotLosses
"""from live_plot import LivePlot

# Initialize the LivePlot class
live_plotter = LivePlot()"""



def train(model, trainloader, valloader, optimizer, objective, device, start_epoch, num_epochs, model_name, unique_id, scheduler=None):
    print("Starting training on device:", device)

    train_loss_log = []
    train_acc_log = [] 
    val_loss_log = []
    val_acc_log = []

    liveloss = PlotLosses()

    for epoch in range(start_epoch, num_epochs + 1):
        print(f'\nEpoch: {epoch}')

        model.train()
        running_loss = 0
        correct = 0
        total = 0

        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = objective(outputs, targets)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
    
            _, predicted_indices = torch.max(outputs.data, 1)
            
            targets = targets.argmax(dim=1)
            total += targets.size(0)
            correct += (predicted_indices == targets).sum().item()


        # Training accuracy and loss
        train_loss = running_loss / len(trainloader)
        train_acc = 100. * (correct / total)

        train_loss_log.append(train_loss)
        train_acc_log.append(train_acc)

        # Validation phase
        model.eval()
        val_correct = 0
        val_loss = 0
        val_total = 0

        with torch.no_grad():
            for val_inputs, val_targets in valloader:
                val_inputs, val_targets = val_inputs.to(device), val_targets.to(device)

                val_outputs = model(val_inputs)
                val_loss += objective(val_outputs, val_targets).item()

                _, val_predicted = torch.max(val_outputs, 1)
                val_targets = val_targets.argmax(dim=1)
                val_total += val_targets.size(0)
                val_correct += (val_predicted == val_targets).sum().item()

        val_loss = val_loss / len(valloader)
        val_acc = 100. * (val_correct / val_total)

        val_loss_log.append(val_loss)
        val_acc_log.append(val_acc)

        # Live loss plotting
        logs = {
            'log loss': train_loss,
            'accuracy': train_acc,
            'val_log loss': val_loss,
            'val_accuracy': val_acc
        }
        liveloss.update(logs)
        liveloss.send()


    # Save final model
    
    print('Training Finished!')
    return model, train_loss_log, train_acc_log
