import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import cv2

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load pretrained model (ResNet18 for simplicity)
model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 2)  # Normal / Abnormal
model = model.to(device)
model.eval()

# Image transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Grad-CAM function
def generate_heatmap(image_path):
    img = Image.open(image_path).convert("RGB")
    input_tensor = transform(img).unsqueeze(0).to(device)

    # Hook
    gradients = []
    activations = []

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    def forward_hook(module, input, output):
        activations.append(output)

    target_layer = model.layer4[1].conv2
    target_layer.register_forward_hook(forward_hook)
    target_layer.register_backward_hook(backward_hook)

    # Forward
    output = model(input_tensor)
    class_idx = output.argmax(dim=1)
    model.zero_grad()
    output[0, class_idx].backward()

    # Grad-CAM
    grads = gradients[0].cpu().numpy()
    acts = activations[0].cpu().detach().numpy()
    weights = np.mean(grads, axis=(2, 3))[0]
    cam = np.zeros(acts.shape[2:], dtype=np.float32)

    for i, w in enumerate(weights):
        cam += w * acts[0, i]

    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, img.size)
    cam = cam / cam.max()

    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    original = np.array(img)
    overlay = cv2.addWeighted(original, 0.6, heatmap, 0.4, 0)

    return overlay
