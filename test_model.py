import torch
import torch.nn as nn
from torch.autograd import Variable
import csv
import os
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from main import classes
from models.GradCam import GradCAM, generate_and_save_grad_cams, save_grad_cam_heatmap
from translate import translate






def test(model, testloader, device, model_name, unique_id):
    model.to(device)
    model.eval()
    correct = 0
    total = 0
    all_labels = []
    all_preds = []
    use_grad_cam = False
    if use_grad_cam:
        grad_cam = GradCAM(model, model.conv3)  # Initialize GradCAM for conv3 layer

    for idx, (inputs, targets) in enumerate(testloader):

        # Move data to GPU if CUDA is available
         
        inputs, targets = inputs.to(device), targets.to(device)

        inputs = torch.tensor(inputs, requires_grad=False)

        # Feed-forward the network
        with torch.no_grad():
            outputs = model(inputs)

        _, predicted = torch.max(outputs.data, 1)
        gt = torch.argmax(targets.data, 1)

        if use_grad_cam:
                model.eval()
                target_class = targets[3].item()
                pred_label = predicted[3].item()

                str_target = classes[target_class]
                str_pred = classes[pred_label]

                batch_acc = 100. * (predicted == targets).sum().item() / targets.size(0)

                # Generate Grad-CAM
                if idx <= 10:
                    for i in range(len(classes)):
                        
                        cam = grad_cam.generate_cam(class_idx=i, input_tensor=inputs[3:4])
                        img = inputs[3]
                        if not os.path.exists("grad_cams_res"):
                            os.makedirs("grad_cams_res")

                        save_path = f"grad_cams_res/{model_name}_{unique_id}_gradCam_{idx}_{classes[i]}.png"
                        save_grad_cam_heatmap(cam, img, save_path, str_pred=str_pred, str_target=str_target, batch_accuracy=batch_acc, epoch=None)

        total += gt.size(0)
        correct += (predicted == gt).sum().item()

        all_labels.extend(gt.cpu().numpy())
        all_preds.extend(predicted.cpu().numpy())

    # Compute confusion matrix
    nr_classes = (targets.size())[1]
    classes = list(translate.values())[0:nr_classes]
    print(nr_classes)
    plot_cm(all_labels, all_preds, classes )
    # Compute precision, recall, and F1 score
    #precision = precision_score(all_labels, all_preds, average='weighted')
    #recall = recall_score(all_labels, all_preds, average='weighted')
    #f1 = f1_score(all_labels, all_preds, average='weighted')


    #print(f"Precision (weighted): {precision:.4f}")
    #print(f"Recall (weighted): {recall:.4f}")
    #print(f"F1-Score (weighted): {f1:.4f}")

   

    print('Accuracy of the network on the test images: %d %%' % (
        100 * correct / total))
    
    #save_dir = f'./checkpoints/{model_name}_{unique_id}'
    #if not os.path.exists(save_dir):
    #    os.makedirs(save_dir)

    #test_metric = f"{save_dir}/test_score.csv"
    #with open(test_metric, mode='w', newline='') as file:
    #    writer = csv.writer(file)
        
        # Write header
    #    writer.writerow(["Test acc"])
    #    writer.writerow([100 * correct / total])

    
    return 100 * correct / total



def plot_cm(all_labels, all_preds, classes):
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.show()