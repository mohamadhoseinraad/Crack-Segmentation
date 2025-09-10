import os
import time
from datetime import datetime

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import cv2
from Dataset import make_dataloader

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 8
INPUT_SIZE = (512,512)
MODEL_PATH = 'final_model_resnet.pth'

def setmodel(model_path = 'final_model_basic_aug.pth' , model_type='basic'):
    if(model_type == 'basic'):
        from basicmodel.model import CustomCrackSegModel  
        model = CustomCrackSegModel(in_channels=3, out_channels=1)
    elif(model_type == 'resnet'):
        from resnet.model_transfer import PreTrainedEncoderCrackSegModel  
        model = PreTrainedEncoderCrackSegModel(in_channels=3, out_channels=1, pretrained=True)
    elif(model_type == 'dropout'):
        from dropoutnormal.model import CustomCrackSegModel
        model = CustomCrackSegModel(in_channels=3, out_channels=1)
    else:
        raise ValueError("Invalid model type. Choose from 'basic', 'resnet', or 'dropout'.")
    
    model.load_state_dict(torch.load(model_path, map_location=f'cuda:{DEVICE.index}'))
    # state_dict = torch.load(model_path, map_location=f'cuda:{DEVICE.index}')
    model.to(DEVICE)
    model.eval()
    return model

def calculate_metrics(predictions, targets, threshold=0.5):
    predictions = (torch.sigmoid(predictions) > threshold).float()
    targets = targets.float()

    if predictions.dim() == 4 and targets.dim() == 3:
        targets = targets.unsqueeze(1)

    predictions_flat = predictions.view(-1)
    targets_flat = targets.view(-1)

    tp = (predictions_flat * targets_flat).sum()
    fp = predictions_flat.sum() - tp
    fn = targets_flat.sum() - tp

    smooth = 1e-6
    union = predictions_flat.sum() + targets_flat.sum() - tp
    iou = (tp + smooth) / (union + smooth)
    precision = (tp + smooth) / (tp + fp + smooth)
    recall = (tp + smooth) / (tp + fn + smooth)
    f1 = 2 * precision * recall / (precision + recall)

    return {'iou': iou.item(), 'precision': precision.item(), 'recall': recall.item(), 'f1': f1.item()}


def test_model(model, test_loader):
    model.eval()
    test_metrics = {'iou': 0, 'precision': 0, 'recall': 0, 'f1': 0}
    with torch.no_grad():
        for images, masks in test_loader:
            images, masks = images.to(DEVICE), masks.to(DEVICE)
            outputs = model(images)
            batch_metrics = calculate_metrics(outputs, masks)
            for k in test_metrics:
                test_metrics[k] += batch_metrics[k]

    avg_test_metrics = {k: v / len(test_loader) for k, v in test_metrics.items()}
    # print("Test Results:", avg_test_metrics)
    return avg_test_metrics

if __name__ == "__main__":

    model_path = MODEL_PATH
    print(f"Loading model from {model_path}")
    model_path ='final_model_resnet.pth'
    model = setmodel(model_path, model_type = 'resnet')
    est_loader = make_dataloader('test', batch_size=BATCH_SIZE, shuffle=False)

    
    
    if torch.cuda.is_available():
        # state_dict = torch.load(model_path)
        state_dict = torch.load(model_path, map_location=f'cuda:{DEVICE.index}')

    else:
        state_dict = torch.load(model_path, map_location='cpu')

    test_loader = make_dataloader('test', batch_size=BATCH_SIZE, shuffle=False)
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    print("Test Results:", test_model(model, test_loader))