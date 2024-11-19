#!/usr/bin/env python3
############################################################
##    File name: main.py
##    Author: Abdelrahman Eldesokey
##    Email: abdelrahman.eldesokey@liu.se
##    Date created: 2018-08-28
##    Date last modified: 2021-09-02
##    Python Version: 3.6
##    Description: TSBB19 course project (1) starter code.
############################################################
from models.initialization import initialize_weights_xavier, initialize_weights_zeros, initialize_weights_kaiming
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

if __name__ == "__main__":
    import os
    import glob
    import argparse
    import json

    import torch
    import torch.nn as nn
    import torch.backends.cudnn as cudnn

    import torchvision
    import torchvision.transforms as transforms

    from models.cvlNet import cvlNet
    from models.GoalNet import GoalNet
    from models.Ensemble import Ensemble
    from torch.optim.lr_scheduler import StepLR


    from train import train
    #from imagenet.imagenet.imagenet_v2 import ImageNet64LikeCifar10

    parser = argparse.ArgumentParser()

    parser.add_argument("config")

    args = parser.parse_args()

    config_path = args.config

    with open(config_path, 'r') as f:
        config = json.load(f)

    print(print(config))


    # Check if CUDA support is available (GPU)
    if torch.cuda.is_available():
        device = torch.device("cuda")  # Use CUDA
        use_cuda = True
    elif torch.backends.mps.is_available():
        device = torch.device("mps")   # Use MPS on Mac with Apple Silicon
        use_cuda = False  # Since MPS is not CUDA, keep this as False
    else:
        device = torch.device("cpu")   # Default to CPU if no GPU or MPS support
        use_cuda = False




    ##################################
    ## Download the CIFAR10 dataset ##
    ##################################


    # Image transformations to apply to all images in the dataset (Data Augmentation)
    transform_train = transforms.Compose([
        transforms.ToTensor(),                # Convert images to Tensors (The data structure that is used by Pytorch)
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), # Normalize the images to zero mean and unit std
    ])

    # Image transformations for the test set.
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # imagenet dubbla sista pooling
    # imagenet num_workers = 0

    batch_size = (config["tp"].get("batchsize"))

    if batch_size is None:
        batch_size = 256
    

    # Specify the path to the CIFAR-10 dataset and create a dataloader where you specify the "batch_size"
    trainset = torchvision.datasets.CIFAR10(root='visual-object-recognition/data/cifar-10-batches-py', train=True, download=True, transform=transform_train)
    #trainset = ImageNet64LikeCifar10(transform=transform_train)
    generator1 = torch.Generator().manual_seed(100) #Creates a reproducible 
    trainset, validationset = torch.utils.data.random_split(trainset, [0.8, 0.2], generator = generator1) # divide dataset into training and validation data
    
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=int(batch_size), shuffle=False, num_workers=0)
    valloader = torch.utils.data.DataLoader(validationset, batch_size=128, shuffle=True, num_workers=0)

    testset = torchvision.datasets.CIFAR10(root='visual-object-recognition/data/cifar-10-batches-py', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=True, num_workers=2)

    # Specify classes labels
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


    ####################################
    ## Init the network and optimizer ##
    ####################################


    # Load and initialize the network architecture

    use_batchnorm = config["tp"].get("Batch_norm")

    if config["model_info"].get("model") == "cvlNET":
        model = cvlNet()
    elif config["model_info"].get("model") == "GoalNet":
        model = GoalNet(bn=use_batchnorm)
    else:
        exit(0)


    model.to(device)



    if use_cuda:
        cudnn.benchmark = True

    #extract learning rate from config
    lr = config["tp"].get("lr")
    # The objective (loss) function
    objective = nn.NLLLoss()

    weight_decay = config["tp"].get("weight_decay")
    lr_scheduler = config["tp"].get("lr_scheduler")
    



    # Initialize weights based on the specified method
    init_method = config["tp"].get("init_method")

    if init_method == "1":  # Xavier initialization
        model.apply(initialize_weights_xavier)
        print("Xavier initialization applied.")
    elif init_method == "2":  # Zero initialization
        model.apply(initialize_weights_zeros)
        print("Zero initialization applied.")
    elif init_method == "3":  # Kaiming initialization
        model.apply(initialize_weights_kaiming)
        print("Kaiming initialization applied.")
    else:
        print("No specific initialization method selected, using default initialization that is random initialization.")



    # The optimizer used for training the model
    if weight_decay:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.01) 
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr)
        
    if lr_scheduler:
        scheduler = StepLR(optimizer, step_size=config["tp"]["step_size"], gamma=config["tp"]["gamma"])
    else:
        scheduler = None

    #######################
    ## Train the network ##
    #######################

    start_epoch = int(config["model_info"].get("epochs_trained"))
    num_epochs = int(config["tp"].get("epochs"))
    model_name = config["model_info"].get("model_name")
    unique_id = config["model_info"].get("unique_id")

    print("Doooing Warm up")
    for _ in range(1000):
        torch.matmul(torch.rand(500,500).to(device), torch.rand(500,500).to(device))

    model, loss_log, acc_log = train(model,
                                     trainloader,
                                     valloader,
                                     optimizer,
                                     objective,
                                     device,
                                     start_epoch,
                                     num_epochs=num_epochs,
                                     model_name=model_name,
                                     unique_id=unique_id,
                                     scheduler =scheduler )

    



    ##########################
    ## Evaluate the network ##
    ##########################
    test_acc = test(model,
                    testloader,
                    device,
                    model_name=model_name,
                    unique_id=unique_id)

