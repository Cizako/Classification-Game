import torch
import torch.nn as nn
import os
import csv
from models.GradCam import GradCAM, save_grad_cam_heatmap
from torch.optim.lr_scheduler import StepLR
import numpy as np
import matplotlib.pyplot as plt



### Train the network 
# Start training the network for a number of epochs (here we set it to 50). In each epoch: 
# - We loop over all images in the training set
# - Feed them forward through the network
# - Calculate the loss after the last layer.
# - Backpropagate the loss until the first layer.
# - Update the weights for different convolution layers based on the gradients.

def train(model, trainloader,
          valloader,
          optimizer,
          objective,
          device,
          start_epoch,
          num_epochs,
          model_name,
          unique_id,
          scheduler=None):
    

    print("Starting training on device: ",device)

    train_loss_log = []
    train_acc_log = [] 
    val_loss_log = []
    val_acc_log = []



    for epoch in range(start_epoch, num_epochs+1):

        print('\nEpoch: %d' % epoch)

        model.train()

        # Variables for training status
        train_loss = 0
        correct = 0
        total = 0

        # Loop on all images in the dataset
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            
            # Move data to GPU if CUDA is available
            
            inputs, targets = inputs.to(device), targets.to(device)        
           

            # Clear the gradients of all variables 
            optimizer.zero_grad()

            # Feed-forward the network
            outputs = model(inputs)

            #print(targets.size())
            #print("GT: ", targets)
            #print("out: ", outputs)

            #print(outputs.size())
            # Calculate the loss
            loss = objective(outputs, targets)

            # Feed Backward
            loss.backward()
            
            # Update the weights
            optimizer.step()

            # Update training status
            train_loss += loss.item()

            # Find the class with the highest output
            _, predicted_indices = torch.max(outputs.data, 1)  # Shape: (batch_size,)

            # Convert the predicted indices to one-hot encoding
            
            #print("pred: ", predicted_indices)

            gt = torch.argmax(targets)


            # Count number total number of images trained so far and the correctly classified ones

           
            total += targets.size(0)

            
            correct += (predicted_indices == gt).sum().item()

            print('Loss: {:.8f} | Acc: {:.2f}% ({}/{})'.format((train_loss/(batch_idx+1)), 100.*correct/total, correct, total))

        
                

        train_loss_log.append((train_loss/(batch_idx+1)))
        train_acc_log.append(100.*correct/total)

        model.eval()

        val_correct = 0
        val_loss = 0
        val_total = 0

        for val_batch_idx, (val_inputs, val_targets) in enumerate(valloader):

             
            val_inputs, val_targets = val_inputs.to(device), val_targets.to(device)

            val_inputs, val_targets = torch.tensor(val_inputs, requires_grad=True), torch.tensor(val_targets).long()

            val_outputs = model(val_inputs)

            vloss = objective(val_outputs, val_targets)
            val_loss += vloss.item()

            _, val_predicted = torch.max(val_outputs, 1)

            val_total += val_targets.size(0)
            val_correct += (val_predicted == val_targets).sum().item()

            print('Val_Loss: {:.8f} | Val_Acc: {:.2f}% ({}/{})'.format((val_loss/(val_batch_idx+1)), 100.*val_correct/val_total, val_correct, val_total))


        val_loss_log.append(val_loss/(val_batch_idx+1))
        val_acc_log.append(100.*val_correct/val_total)

        if epoch % 5 == 0:
            # Save a checkpoint when the epoch finishes
            state = {
                'epoch': epoch,
                'net': model.state_dict(),
                'optimizer' : optimizer.state_dict(),
            }
            save_dir = f'./checkpoints/{model_name}_{unique_id}'
            if not os.path.isdir(save_dir):
                os.mkdir(save_dir)
            file_path = f'{save_dir}/checkpoint_{epoch}.ckpt'
            torch.save(state, file_path)

        # Step the scheduler
        if scheduler is not None:
            scheduler.step()
            print(optimizer.defaults["lr"])
    # Save the final model    
    torch.save(model.state_dict(), f'./saved_models/{model_name}_{unique_id}_final')
    print('Training Finished!')


    #save loss to csv
    csv_file = f"{save_dir}/training_metrics.csv"
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        
        # Write header
        writer.writerow(["Train_Loss", "Train_Accuracy", "Val_Loss", "Val_Accuracy"])
        
        # Write the loss and accuracy values
        for t_loss, t_acc, v_loss, v_acc in zip(train_loss_log, train_acc_log, val_loss_log, val_acc_log):
            writer.writerow([t_loss, t_acc, v_loss, v_acc])



    return model, train_loss_log, train_acc_log
