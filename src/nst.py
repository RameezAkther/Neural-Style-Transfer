import torch
import torch.nn as nn
import torch.optim as optim
import copy
import numpy as np
import cv2
import os
import uuid
from torchvision import models, transforms

# Hyperparameters
MAX_IMAGE_SIZE = 512
OPTIMIZER = 'adam'
ADAM_LR = 10
CONTENT_WEIGHT = 5e0
STYLE_WEIGHT = 1e2
TV_WEIGHT = 1e-3
NUM_ITER = 500
SHOW_ITER = 10
PIXEL_CLIP = 'True'
PRESERVE_COLOR = 'False'

VGG19_PATH = 'models/vgg19-d01eb7cb.pth'
device = ("cuda" if torch.cuda.is_available() else "cpu")

# Load VGG19 Skeleton
vgg = models.vgg19(pretrained=False)
vgg.load_state_dict(torch.load(VGG19_PATH, weights_only=False), strict=False)
#vgg.load_state_dict(torch.load(VGG19_PATH), strict=False)
model = copy.deepcopy(vgg.features).to(device)
for param in model.parameters():
    param.requires_grad = False

mse_loss = torch.nn.MSELoss()

def initialize_global_variables(content_weight, style_weight, preserve_color, num_iter):
    global CONTENT_WEIGHT, STYLE_WEIGHT, PRESERVE_COLOR, NUM_ITER
    CONTENT_WEIGHT = float(content_weight)
    STYLE_WEIGHT = float(style_weight)
    if preserve_color == "true":
        PRESERVE_COLOR = True
    else:
        PRESERVE_COLOR = False
    NUM_ITER = int(num_iter)

# Color transfer
def transfer_color(src, dest):
    if (PIXEL_CLIP=='True'):
        src, dest = src.clip(0,255), dest.clip(0,255)
        
    # Resize src to dest's size
    H,W,_ = src.shape 
    dest = cv2.resize(dest, dsize=(W, H), interpolation=cv2.INTER_CUBIC)
    
    dest_gray = cv2.cvtColor(dest, cv2.COLOR_BGR2GRAY) #1 Extract the Destination's luminance
    src_yiq = cv2.cvtColor(src, cv2.COLOR_BGR2YCrCb)   #2 Convert the Source from BGR to YIQ/YCbCr
    src_yiq[...,0] = dest_gray                         #3 Combine Destination's luminance and Source's IQ/CbCr
    
    return cv2.cvtColor(src_yiq, cv2.COLOR_YCrCb2BGR)  #4 Convert new image from YIQ back to BGR

def itot(img):
    H, W, C = img.shape
    image_size = tuple([int((float(MAX_IMAGE_SIZE) / max([H, W])) * x) for x in [H, W]])
    itot_t = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(image_size),
        transforms.ToTensor()
    ])
    normalize_t = transforms.Normalize([103.939, 116.779, 123.68], [1, 1, 1])
    tensor = normalize_t(itot_t(img) * 255).unsqueeze(dim=0)
    return tensor

def ttoi(tensor):
    ttoi_t = transforms.Compose([
        transforms.Normalize([-103.939, -116.779, -123.68], [1, 1, 1])])
    tensor = tensor.squeeze()
    img = ttoi_t(tensor).cpu().numpy()
    img = img.transpose(1, 2, 0)
    return img

def gram(tensor):
    B, C, H, W = tensor.shape
    x = tensor.view(C, H * W)
    return torch.mm(x, x.t())

def content_loss(g, c):
    return mse_loss(g, c)

def style_loss(g, s):
    return mse_loss(g, s) / (g.shape[0] ** 2)

def tv_loss(c):
    x = c[:, :, 1:, :] - c[:, :, :-1, :]
    y = c[:, :, :, 1:] - c[:, :, :, :-1]
    return torch.sum(torch.abs(x)) + torch.sum(torch.abs(y))

def get_features(model, tensor):
    layers = {
        '3': 'relu1_2',
        '8': 'relu2_2',
        '17': 'relu3_3',
        '26': 'relu4_3',
        '35': 'relu5_3',
        '22': 'relu4_2',
    }
    features = {}
    x = tensor
    for name, layer in model._modules.items():
        x = layer(x)
        if name in layers:
            if name == '22':
                features[layers[name]] = x
            else:
                features[layers[name]] = gram(x) / (x.shape[2] * x.shape[3])
        if name == '35':
            break
    return features

def initial(content_tensor):
    B, C, H, W = content_tensor.shape
    return torch.randn(C, H, W).mul(0.001).unsqueeze(0).to(device).requires_grad_(True)

def perform_style_transfer(content_img, style_img, socketio, output_dir="static/results", session_id="", content_weight=5e0, style_weight=1e2, preserve_color=False, num_iterations=500):
    initialize_global_variables(content_weight, style_weight, preserve_color, num_iterations)
    os.makedirs(output_dir, exist_ok=True)
    content_tensor = itot(content_img).to(device)
    style_tensor = itot(style_img).to(device)
    g = initial(content_tensor)
    optimizer = optim.Adam([g], lr=ADAM_LR)

    c_feat = get_features(model, content_tensor)
    s_feat = get_features(model, style_tensor)

    content_layers = ['relu4_2']
    style_layers = ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3', 'relu5_3']

    for i in range(NUM_ITER):
        optimizer.zero_grad()
        g_feat = get_features(model, g)

        c_loss = sum(content_loss(g_feat[l], c_feat[l]) for l in content_layers) * CONTENT_WEIGHT
        s_loss = sum(style_loss(g_feat[l], s_feat[l]) for l in style_layers) * STYLE_WEIGHT
        t_loss = TV_WEIGHT * tv_loss(g.clone().detach())
        total_loss = c_loss + s_loss + t_loss

        total_loss.backward()
        optimizer.step()

        # Emit intermediate results
        if (i + 1) % SHOW_ITER == 0:
            img = ttoi(g.clone().detach())
            if PRESERVE_COLOR == True:
                img = transfer_color(content_img, img)
            img_path = os.path.join(output_dir, f"stylized_{i+1}.jpg")
            cv2.imwrite(img_path, img)
            print(f"Iteration {i+1}: Saved image at {img_path}")

            # Emit update to frontend
            socketio.emit("image_update", {"image_url": "/" + img_path, "epoch": i + 1, "session_id": session_id})

    final_img = ttoi(g.clone().detach())
    if PRESERVE_COLOR:  # Apply color transfer for the final output
        final_img = transfer_color(content_img, final_img)
    final_path = os.path.join(output_dir, "final_output.jpg")
    cv2.imwrite(final_path, final_img)
    return final_path
