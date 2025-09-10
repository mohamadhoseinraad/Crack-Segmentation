import os
import time
from datetime import datetime

import matplotlib.pyplot as plt
from tensorboard import summary
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

def test_image(model, src_path):
    model.eval()
    img = cv2.imread(src_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  
    
    # Store original dimensions
    original_h, original_w = img.shape[:2]

    target_h, target_w = INPUT_SIZE
    img_resized = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

    img_np = img_resized.astype(np.float32)
    img_np = img_np / 255.0
    
    
    img_tensor = torch.from_numpy(img_np).permute(2, 0, 1)  #  HWC to CHW 

    with torch.no_grad():
        img_tensor = img_tensor.to(DEVICE)
        output = model(img_tensor.unsqueeze(0))
        pred_mask = torch.sigmoid(output) > 0.5
        
        pred_mask_np = pred_mask[0, 0].cpu().numpy()
        
        pred_mask_resized = cv2.resize(pred_mask_np.astype(np.uint8), (original_w, original_h), interpolation=cv2.INTER_NEAREST)
        
        img_normalized = img.astype(np.float32) / 255.0
        img_normalized = (img_normalized - img_normalized.min()) / (img_normalized.max() - img_normalized.min())

        overlay = img_normalized.copy()
        
        alpha = 0.5  
        overlay[pred_mask_resized > 0, 0] = alpha * 1.0 + (1 - alpha) * overlay[pred_mask_resized > 0, 0]
        overlay[pred_mask_resized > 0, 1] *= 0.6  
        overlay[pred_mask_resized > 0, 2] *= 0.6  
        overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
        
        print(overlay.shape)
    return overlay
    

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

if __name__ == "__main__":

    model_path ='final_model_dropout_aug.pth'
    print(f"Loading model from {model_path}")

    model = setmodel(model_path, model_type = 'dropout')
    # infer_ex = { 'file_name' : "image.jpg" , 'out_name' : "result.jpg"}
    infer_ex = { 'file_name' : "5.jpg" , 'out_name' : "5out.jpg"}
    print(f"Processing image: {infer_ex['file_name']}")
    result = test_image(model, infer_ex['file_name'])

    #  float array to uint8
    result_uint8 = (result * 255).astype(np.uint8)
    cv2.imwrite(infer_ex['out_name'], result_uint8)
    # model = setmodel('final_model_dropout.pth', model_type='dropout')
    # def print_model_structure(model):
    #     print("Model Structure:")
    #     print("=" * 50)
    #     for name, module in model.named_modules():
    #         if len(list(module.children())) == 0:  # فقط لایه‌های پایه
    #             print(f"{name:30s} - {module.__class__.__name__}")
    #     print("=" * 50)
    # from torchsummary import summary
    # summary(model, input_size=(3, 512, 512), device=DEVICE.type)
    # print_model_structure(model)