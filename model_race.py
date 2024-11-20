import os
import torch
import onnx
import onnxruntime as ort
from torchvision import transforms
from Dataset import CustomImageDataset, MonkeyImageDataset
from torch.utils.data import DataLoader
from test_model import test
import traceback
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from translate import translate
from livelossplot import PlotLosses
import seaborn as sns
import numpy as np

def plot_cm(all_labels, all_preds, classes):
    # Compute confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    #cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]  # Normalize by        
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues",
            xticklabels=classes, yticklabels=classes)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.show()


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


def evaluate_onnx_model(model_path, testloader, device='mps'):
    """
    Evaluate an ONNX model
    """
    # Validate ONNX model
    onnx_model = onnx.load(model_path)
    onnx.checker.check_model(onnx_model)
    
    # Create ONNX inference session
    ort_session = ort.InferenceSession(model_path)
    
    # Tracking variables
    total = 0
    correct = 0
    all_labels = []
    all_preds = []
    
    # Inference loop
    for idx, (inputs, targets) in enumerate(testloader):
        # Ensure 4D tensor shape: [batch, channels, height, width]
        if inputs.dim() == 3:
            inputs = inputs.unsqueeze(0)  # Add batch dimension if missing
            print("we got 3 dimensional input, adding one dimension in order to get the missing batch")
        
        # Convert inputs to numpy for ONNX inference
        inputs_np = inputs.numpy()
        
        # Prepare input for ONNX Runtime
        ort_session = ort.InferenceSession(
        model_path, 
        providers=["CPUExecutionProvider"])  # or ["CUDAExecutionProvider"] if using GPU

        ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(inputs)}
        outputs = ort_session.run(None, ort_inputs)[0]
        # Convert outputs to PyTorch tensors for processing
        pred = torch.tensor(outputs)
        pred = torch.softmax(pred, dim=1)
        pred = torch.argmax(pred, dim=1)
        
        # Convert targets to match original function
        gt = torch.argmax(targets, 1)
        
        # Update tracking metrics
        total += gt.size(0)
        correct += (pred == gt).sum().item()
        
        # Collect labels and predictions
        all_labels.extend(gt.numpy())
        all_preds.extend(pred.numpy())
        
    # Calculate accuracy
    accuracy = correct / total

    # Compute confusion matrix
    class_names = testloader.dataset.class_names

    # Plot confusion matrix
    plot_cm(all_labels, all_preds, class_names)

    # Calculate additional metrics
    accuracy = 100 * correct / total
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)

    # Print metrics
    print(f'Accuracy: {accuracy:.2f}%')
    print(f'Precision (weighted): {precision:.2f}')
    print(f'Recall (weighted): {recall:.2f}')
    print(f'F1-score (weighted): {f1:.2f}')
    
    return {
        'accuracy': accuracy,
        'recall': recall,
        'f1': f1,
        'precision': precision
    }

def eval_models(models_folder, dataset_path, image_size=(64, 64), device='mps', num_of_classes=10, data_percentage=1):
    """
    Evaluate PyTorch (.pt) and ONNX (.onnx) models in a given folder
    
    Args:
    - models_folder (str): Path to folder containing model files
    - dataset_path (str): Path to dataset
    - image_size (tuple): Resize dimensions for images
    - device (str): Compute device (e.g., 'mps', 'cuda', 'cpu')
    - num_of_classes (int): Number of classes in the dataset
    - data_percentage (float): Percentage of dataset to use
    
    Returns:
    - dict: Model names and their corresponding accuracies
    """
    # Create transform
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(image_size)
    ])
    
    # Load dataset
    dataset = MonkeyImageDataset(
        dataset_path, 
        transform=transform, 
        amount_of_classes=num_of_classes, 
        data_percentage=data_percentage
    )
    test_loader = DataLoader(dataset=dataset, batch_size=1, shuffle=False)
    # Dictionary to store results
    results = {}

    # Iterate through all model files in the models folder
    for model_file in os.listdir(models_folder):
        model_path = os.path.join(models_folder, model_file)
        
        try:
            # PyTorch TorchScript models (.pt)
            if model_file.endswith('.pt'):
                model = torch.jit.load(model_path)
                accuracy = test(
                    model=model, 
                    testloader=dataset, 
                    device=device, 
                    model_name=model_file, 
                    unique_id=model_file.split('.')[0]
                )
                results[model_file] = accuracy
            
            # ONNX models (.onnx)
            elif model_file.endswith('.onnx'):
                # Use new ONNX evaluation method
                model_results = evaluate_onnx_model(
                    model_path, 
                    testloader=test_loader, 
                    device=device
                )
                results[model_file] = model_results['accuracy']
        
        except Exception as e:
            error_message = traceback.format_exc() 
            print("An error occurred:", error_message)
    
    return results

# Example usage
if __name__ == "__main__":
    models_folder = 'model_folder'
    dataset_path = 'Monkey/validation/validation'
    
    # Run evaluation
    model_results = eval_models(models_folder, dataset_path)
    
    # Print results
    for model, accuracy in model_results.items():
        print(f"Model {model}: Accuracy = {accuracy}")