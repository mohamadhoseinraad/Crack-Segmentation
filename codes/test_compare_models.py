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

def test_resnet(model_path):

    from resnet.model_transfer import PreTrainedEncoderCrackSegModel  
    model = PreTrainedEncoderCrackSegModel(in_channels=3, out_channels=1, pretrained=True)

    test_loader = make_dataloader('test', batch_size=BATCH_SIZE, shuffle=False)

    
    
    if torch.cuda.is_available():
        state_dict = torch.load(model_path, map_location=f'cuda:{DEVICE.index}')
    else:
        state_dict = torch.load(model_path, map_location='cpu')
    
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    return test_model(model, test_loader)

def test_dropout(model_path):

    # model_path = 'final_model_dropout.pth' 
    from dropoutnormal.model import CustomCrackSegModel  
    model = CustomCrackSegModel(in_channels=3, out_channels=1)

    test_loader = make_dataloader('test', batch_size=BATCH_SIZE, shuffle=False)

    
    
    if torch.cuda.is_available():
        # state_dict = torch.load(model_path)
        state_dict = torch.load(model_path, map_location=f'cuda:{DEVICE.index}')

    else:
        state_dict = torch.load(model_path, map_location='cpu')
    
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    return test_model(model, test_loader)


def test_basic(model_path):
    

    # model_path = 'final_model_dropout.pth' 
    from basicmodel.model import CustomCrackSegModel  
    model = CustomCrackSegModel(in_channels=3, out_channels=1)

    test_loader = make_dataloader('test', batch_size=BATCH_SIZE, shuffle=False)

    
    
    if torch.cuda.is_available():
        # state_dict = torch.load(model_path)
        state_dict = torch.load(model_path, map_location=f'cuda:{DEVICE.index}')

    else:
        state_dict = torch.load(model_path, map_location='cpu')
    
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    return test_model(model, test_loader)

def plot_comparison(results_dict):
    
    metrics = ['iou', 'precision', 'recall', 'f1']
    models = list(results_dict.keys())
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    width = 0.2
    
    r = np.arange(len(metrics))
    
    for i, model in enumerate(models):
        values = [results_dict[model][metric] for metric in metrics]
        pos = [p + width*i for p in r]
        ax.bar(pos, values, width=width, label=model)
    
    ax.set_ylabel('Score')
    ax.set_title('Model Performance Comparison')
    ax.set_xticks([r + width*len(models)/2 for r in range(len(metrics))])
    ax.set_xticklabels(metrics)
    ax.legend()
    
    for i, model in enumerate(models):
        for j, metric in enumerate(metrics):
            value = results_dict[model][metric]
            ax.text(r[j] + width*i, value + 0.01, f'{value:.3f}', 
                   ha='center', va='bottom', rotation=0)
    
    plt.tight_layout()
    plt.savefig('model_comparison.png')
    plt.show()

def compare_models():
    print("Starting model tests...")
    print(f"Using device: {DEVICE}")
    print("Testing ResNet model...")
    resnet_results = test_resnet('final_model_resnet.pth')
    print("\nTesting ResNet with Augmentation model...")
    resnet_aug_results = test_resnet('final_model_resnet_aug.pth')

    print("\nTesting Dropout model...")
    dropout_results = test_dropout('final_model_dropout.pth')
    
    print("\nTesting Dropout with Augmentation model...")
    dropout_aug_results = test_dropout('final_model_dropout_aug.pth')
    
    print("\nTesting Basic model...")
    basic_results = test_basic('final_model_basic.pth')
    
    print("\nTesting Basic with Augmentation model...")
    basic_aug_results = test_basic('final_model_basic_aug.pth')
    
  
    
    all_results = {
        'ResNet': resnet_results,
        'ResNet Aug': resnet_aug_results,
        'Dropout': dropout_results,
        'Dropout Aug': dropout_aug_results,
        'Basic': basic_results,
        'Basic Aug': basic_aug_results
    }
    
    print("\n\n Model Comparison Results =====")
    metrics = ['iou', 'precision', 'recall', 'f1']
    header = "Model".ljust(15) + "".join(metric.ljust(12) for metric in metrics)
    print(header)
    print("-" * len(header))
    
    for model_name, metrics_dict in all_results.items():
        row = model_name.ljust(15)
        for metric in metrics:
            row += f"{metrics_dict[metric]:.6f}".ljust(12)
        print(row)
    
    # Plot comparison
    print("\nGenerating comparison plot...")
    plot_comparison(all_results)


if __name__ == "__main__":

    compare_models()
    