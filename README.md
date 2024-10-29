# Classification-Game


# BUDGET: 100 Billion Nerual nuggets

# Shop
- more data - Data Potion/ Info infusion : 20 B NN
- learning rate decay : 5 B NN
- augmentation differnt kinds
  - rotate - Whirl and swirl: 10 B NN
  - blur - ?? : 5 B NN 
  - flip - flip trick   : 10 B NN
  - upsample/downsample - Resolution rocket : 5 B NN
  - shift - Need for shift: Tokyo data drift : 5 B NN
  - Cutout - Cutout mask : 10 B NN
- ensemble learning - Ensemble enchanter 60 B NN
- one more convulutional layers - Neural turbo boost : 20 B NN 
- Learning rate scheduler : Speed dial : 10 B NN
- Dropout - Drop shield : 10 B NN
- Weight Decay : 10 : 20 B NN
- Mixup - Mixup mixer : 20 B NN
- Focus on one class - Herr Nilsson's friend


# Things to try out for yourself
- Different learning rates
- Batch sizes


# Instructions on how to implement things from the shop

</details>

<details>
<summary><strong> Implement Augmentations </strong> </summary>


  In order to implement augmentations for the data one needs to change the cell called "DATASET"  
  
  ```python
transform = transforms.Compose([ 
      #Put augmentations here preferably
      transforms.ToTensor(), 
      transforms.Resize((IMAGE_SIZE[0], IMAGE_SIZE[1]))
      ])
  ```

### Whirl and Swirl

```python
transforms.RandomRotation(degrees=15),  # Rotate the image randomly within a 15-degree range
```

### Flip trick

```python
transforms.RandomHorizontalFlip(p=0.2),  # Randomly flip the image horizontally with 20% probability
transforms.RandomVerticalFlip(p=0.2),  # Randomly flip the image vertically with 20% probability
```

### Need for shift: Tokyo data drift

```python
transforms.RandomHorizontalFlip(p=0.2),  # Randomly flip the image horizontally with 20% probability
transforms.RandomVerticalFlip(p=0.2),  # Randomly flip the image vertically with 20% probability
```

### Foggy Lens  

```python
transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),  # Apply Gaussian blur with a random kernel size

```

### Missing Pieces (Cutout Chaos) 

```python
transforms.RandomErasing(p=0.5, scale=(0.02, 0.2), ratio=(0.3, 3.3)),  # Randomly erases parts of the image
```
  </details>     </details>


</details>




</details>




</details>

<details>
<summary><strong> Implement Info infusion </strong> </summary>

Change the variable ```DATA_PERCENTAGE``` to 1 in the dataset code block


</details>


</details>

<details>
<summary><strong> Implement Ensemble Enchanter </strong> </summary>

This one is a bit more tricky....

Ensembling combines predictions from multiple models to improve accuracy and stability. By merging outputs (through averaging, voting, or stacking), ensembles reduce individual model errors, leading to more robust and reliable predictions, especially on complex tasks.

You could implement ensembling by creating a class that takes a list of models and combines their predictions. This class could run each model independently, then combine their outputs through  majority voting.

It could look like this:

```python
class Ensemble(nn.Module):
    def __init__(self, models):
        super(Ensemble, self).__init__()
        # Store the list of models
        self.models = nn.ModuleList(models)

    def forward(self, x):
        # Get predictions from each model and store them in a list        
        outputs = [F.softmax(model(x), dim=1) for model in self.models]

        # Take the average of them
        output = torch.mean(torch.stack(outputs), dim=0)
        
        return output
```

The next step is to train a few models, preferably with different hyperparameters (to introduce some variability) and add them to a list.

For this you need to create different optimizers for each model, since these hold the model parameters in them. So if you want three models in your ensemble it could look like this:


```python

model1 = ClassificationModel(num_classes=NUM_OF_CLASSES, input_size=IMAGE_SIZE)
optimizer1 = torch.optim.SGD(model1.parameters(), lr=LR)

model2 = ClassificationModel(num_classes=NUM_OF_CLASSES, input_size=IMAGE_SIZE)
optimizer2 = torch.optim.SGD(model2.parameters(), lr=LR)


model3 = ClassificationModel(num_classes=NUM_OF_CLASSES, input_size=IMAGE_SIZE)
optimizer3 = torch.optim.SGD(model1.parameters(), lr=LR)


model1.to(device)
model1, t_loss, t_acc, v_loss, v_acc = train(model1, train_loader, val_loader, optimizer1, criterion, device, start_epoch=START_EPOCH, num_epochs=EPOCHS, model_name=MODEL_NAME, unique_id=ID)


model2.to(device)
model2, t_loss, t_acc, v_loss, v_acc = train(model2, train_loader, val_loader, optimizer2, criterion, device, start_epoch=START_EPOCH, num_epochs=EPOCHS, model_name=MODEL_NAME, unique_id=ID)


model3.to(device)
model3, t_loss, t_acc, v_loss, v_acc = train(model3, train_loader, val_loader, optimizer3, criterion, device, start_epoch=START_EPOCH, num_epochs=EPOCHS, model_name=MODEL_NAME, unique_id=ID)


models = [model1, model2, model3]
```

Then run the ensemble on the validation data!


```

ensemble = Ensemble(models)

acc = test(model=ensemble, testloader=val_loader, device=device, model_name="ensemble", unique_id=ID)


```


</details>


<details>
<summary><strong> Implement Neural turbo boost </strong> </summary>

Why add more layers? Adding a layer to a Convolutional Neural Network (CNN) increases the model’s depth, allowing it to learn more complex features from the input data. New layers, like convolutional, pooling, or fully connected layers, enhance the network's ability to capture patterns such as edges, textures, or object parts. Adding layers can improve model performance but also increases computational requirements and the risk of overfitting.

For this you should change the ClassificationModel. You need to add the convolutional layer to the constructor, the forward method and the get_fc_input_size method.

For the constructor add a Conv2d as following:

```python

class ClassificationModel(nn.Module):
    def __init__(self, num_classes=10, input_size=(500, 500)):
        super(ClassificationModel, self).__init__()
        
        # First convolutional layer: 3 input channels (RGB), 32 output channels, kernel size 5, padding 2 to preserve size
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding=2)

```

To note is that the in_channels must match with the previous layers out_channel

For the forward method:

```python

def forward(self, x):
        # First conv -> ReLU -> Max Pooling
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)

        # Second conv -> ReLu -> Max Pooling
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
```

Make sure that you add the activation (ReLu) and maxpooling. Here you can experiment with the pooling parameters if you add more layers

For the get_fc_input_size() (This method calculates how large the fully connected layer input should be):

```python

def _get_fc_input_size(self, input_size):
        x = torch.zeros(1, 3, *input_size)  # Create a dummy input tensor
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, kernel_size=2)
        return x.numel()  # Total number of elements after conv layers

```
</details>


<details>
<summary><strong> Weight decay </strong> </summary>
What is weight decay? Weight decay is a regularization technique used to prevent overfitting in machine learning models by adding a penalty to the model's loss function based on the size of its weights. It works by slightly reducing the weights during training, encouraging simpler models with smaller weights, which often generalize better to new data. This technique is especially useful in neural networks, where complex models can easily overfit to the training data.


```python
weight_decay = 1e-4  # Adjust weight decay as needed
optimizer = torch.optim.SGD(model.parameters(), lr=LR, weight_decay=weight_decay)
```

Here you can experiment with the weight decay parameter. It controls how much it should penalize large weights.

</details>



</details>

<details>
<summary><strong> Implement Rate Rollercoaster </strong> </summary>

Why have learning rate scheduler?A learning rate scheduler is used to adjust the learning rate during training to improve the performance of a machine learning model. By modifying the learning rate, the scheduler helps to balance the trade-off between convergence speed and stability. A high learning rate can lead to unstable training and overshooting the optimal solution, while a low learning rate can result in slow convergence. Learning rate schedulers can implement strategies such as gradually decreasing the learning rate over time or adjusting it based on performance metrics, allowing the model to escape local minima and achieve better overall accuracy. This dynamic approach enhances training efficiency and often leads to improved model performance.




</details>


</details>

<details>
<summary><strong> Implement Speed Boost </strong> </summary>

Why momentum? Momentum is an optimization technique that helps accelerate gradients vectors in the right directions, thus leading to faster converging. It works by adding a fraction of the previous update to the current update, which helps to smooth out the updates and reduces oscillation, especially in areas with noisy gradients. This technique mimics the physical concept of momentum, where the optimizer retains a memory of past gradients to guide its current direction.

Add momemntum to the optimizer:

```python
momentum = 0.9
optimizer1 = torch.optim.SGD(model1.parameters(), lr=LR, momentum=momentum)
```
</details>


</details>

<details>
<summary><strong> Herr Nilsson's friend</strong> </summary>
If you want to get the title of Herr Nilssons friend you might want to give an extra reward to the model when it makes corrects predictions for the squirrel monkey class. You can do this by:

```python
class_weights = torch.ones(NUM_OF_CLASSES)
class_weights[7] = 5
class_weights = class_weights.to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights)
```

Remeber that you still want high recall so you still want to predict some of the other monkeys correctly
</details>






<details>
<summary><strong> Implement Drop shield </strong> </summary>



Dropout is a regularization technique that helps prevent overfitting by randomly "dropping out" a proportion of neurons during training. These changes are supposed to be implemented in the cellblock for the model.

#### 1. Add a dropout_rate parameter to __init__ (default 0.5)
```python
def __init__(self, num_classes=10, input_size=(500, 500), dropout_rate=0.5):
```
#### 2. Add two types of dropout layers in the init method

```python
# Dropout layers
        self.dropout1 = nn.Dropout2d(p=dropout_rate)  # Spatial dropout for convolutional layers
        self.dropout2 = nn.Dropout(p=dropout_rate)    # Regular dropout for fully connected layers
```
#### 3. Apply dropout after activation functions but before pooling layers in the forwards method

One convolutional block should then look like this
```python
# First conv -> ReLU -> Dropout -> Max Pooling
        x = F.relu(self.conv1(x))
        x = self.dropout1(x)  # Apply spatial dropout
        x = F.max_pool2d(x, kernel_size=2, stride=2)
```
#### 4. Add dropout before the final prediction layer

```python
# Fully connected layer -> ReLU -> Dropout
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)  # Apply regular dropout before final layer
```

#### 5. Use dropout
To use this model, you can instantiate it with different dropout rates:
```python
# Default dropout rate (0.5)
model = ClassificationModel(num_classes=10, input_size=(500, 500))

# Custom dropout rate
model = ClassificationModel(num_classes=10, input_size=(500, 500), dropout_rate=0.3)
```

</details>


</details>
