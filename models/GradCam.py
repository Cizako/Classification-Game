import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import cv2
from torchvision import transforms
import os


# Function to register hook and get gradients from a specific layer (conv4)
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hook()

    def hook(self):
        def forward_hook(module, input, output):
            self.activations = output

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0]

        # Register hooks
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)

    def generate_cam(self, class_idx, input_tensor):
        self.model.eval()
        
        # Ensure the input tensor requires gradients
        #input_tensor.requires_grad = True

        # Forward pass
        output = self.model(input_tensor)

        # Get the class score for the target class
        class_score = output[:, class_idx]

        # Backward pass
        self.model.zero_grad()
        class_score.backward(retain_graph=True)

        # Compute Grad-CAM
        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])
        for i in range(self.activations.size(1)):
            self.activations[:, i, :, :] *= pooled_gradients[i]

        heatmap = torch.mean(self.activations, dim=1).squeeze()
        heatmap = np.maximum(heatmap.cpu().detach().numpy(), 0)
        heatmap /= np.max(heatmap).squeeze()
        return heatmap
    

def save_grad_cam_heatmap(cam, img, file_path, str_pred=None, str_target=None, batch_accuracy=None, epoch=None):
    # Resize the heatmap to match the input image size
    heatmap = cv2.resize(cam, (img.shape[2], img.shape[1]))

    # Normalize and convert heatmap to 8-bit color image
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)  # Apply the JET color map
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)      # Convert BGR (OpenCV) to RGB (Matplotlib)

    # Process the input image
    img = img.detach().cpu().numpy().transpose(1, 2, 0)    # Detach and transpose to (H, W, C)
    img = (img - img.min()) / (img.max() - img.min())       # Normalize the image to range [0, 1]
    img = np.uint8(255 * img)                               # Convert to uint8 for overlay

    # Combine the heatmap with the image
    overlay = cv2.addWeighted(heatmap, 0.3, img, 0.7, 0)

    # Create a figure and axes for displaying the image and colorbar
    fig, (ax_img, original_im, ax_colorbar) = plt.subplots(1, 3, figsize=(12, 6), gridspec_kw={'width_ratios': [4, 4, 0.1]})

    # Display the image with heatmap overlay
    ax_img.imshow(overlay)
    ax_img.axis('off')  # Hide axis for the overlay image

    # Display the original image
    original_im.imshow(img)
    original_im.axis('off')  # Hide axis for the original image

    # Add title with prediction, ground truth, batch accuracy, and epoch information
    title = ""
    if str_pred is not None:
        title += f'Pred: {str_pred}  '
    if str_target is not None:
        title += f'GT: {str_target} '
    if batch_accuracy is not None:
        title += f'Batch accuracy: {batch_accuracy:.2f} '
    if epoch is not None:
        title += f'Epoch: {epoch}'

    # Set the title with a background for contrast
    ax_img.set_title(title, color='white', fontsize=14, weight='bold',
                     bbox=dict(facecolor='black', alpha=0.7))

    # Display the colorbar for the overlay (heatmap)
    norm = plt.Normalize(vmin=0, vmax=255)
    sm = plt.cm.ScalarMappable(cmap='jet', norm=norm)
    cbar = plt.colorbar(sm, cax=ax_colorbar)
    cbar.set_label('Heatmap Intensity', rotation=270, labelpad=15)

    # Save the figure as a PNG file
    plt.savefig(file_path, bbox_inches='tight', pad_inches=0)
    plt.close(fig)




def get_one_example_per_class(test_loader, num_classes):
    examples = {}
    for inputs, targets in test_loader:
        for i, target in enumerate(targets):
            if target.item() not in examples:
                examples[target.item()] = inputs[i]
            if len(examples) == num_classes:
                return examples
    return examples    

def generate_and_save_grad_cams(model, grad_cam, test_loader, classes, save_dir, device):
    # Ensure the model is in evaluation mode and on the correct device
    model.to(device)
    model.eval()

    # Get one example per class
    examples = get_one_example_per_class(test_loader, num_classes=len(classes))

    # Generate Grad-CAM for each example
    for class_idx, example in examples.items():
        input_tensor = example.unsqueeze(0).to(device)  # Add batch dimension and move to device
        target_class = class_idx
        pred_label = target_class  # For visualization, use the same class as prediction

        # Generate Grad-CAM
        cam = grad_cam.generate_cam(class_idx=target_class, input_tensor=input_tensor)
        img = input_tensor.squeeze().detach().cpu().numpy().transpose(1, 2, 0)  # Convert to HWC format

        str_target = classes[target_class]
        str_pred = classes[pred_label]
        batch_acc = 100.0  # No batch accuracy computation needed here, set to 100%

        # Save Grad-CAM visualization
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        save_path = f"{save_dir}/grad_cam_class_{target_class}.png"
        save_grad_cam_heatmap(cam, img, save_path, str_pred=str_pred, str_target=str_target, batch_accuracy=batch_acc)
