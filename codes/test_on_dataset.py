import torch
import matplotlib.pyplot as plt
from Dataset import make_dataloader

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def visualize_predictions_with_overlay(model, test_loader, num_samples=3):
    
    
    dataiter = iter(test_loader)
    fig, axes = plt.subplots(num_samples, 4, figsize=(20, 5*num_samples))
    
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    with torch.no_grad():
        for i in range(num_samples):
            try:
                images, masks = next(dataiter)
                image = images[0].to(DEVICE)
                mask = masks[0].to(DEVICE)
                
                
                output = model(image.unsqueeze(0))
                pred_mask = torch.sigmoid(output) > 0.5
                
                
                image_np = image.cpu().permute(1, 2, 0).numpy()
                mask_np = mask.cpu().numpy()
                pred_mask_np = pred_mask[0, 0].cpu().numpy()
        
                image_np = (image_np - image_np.min()) / (image_np.max() - image_np.min())
                
                # Create overlay image
                overlay = image_np.copy()
                
                overlay[pred_mask_np > 0, 0] = 1.0  # Red channel
                overlay[pred_mask_np > 0, 1] *= 0.5  # Reduce green
                overlay[pred_mask_np > 0, 2] *= 0.5  # Reduce blue
                
                # Plot
                axes[i, 0].imshow(image_np)
                axes[i, 0].set_title('Original Image')
                axes[i, 0].axis('off')
                
                axes[i, 1].imshow(mask_np, cmap='gray')
                axes[i, 1].set_title('Ground Truth Mask')
                axes[i, 1].axis('off')
                
                axes[i, 2].imshow(pred_mask_np, cmap='gray')
                axes[i, 2].set_title('Predicted Mask')
                axes[i, 2].axis('off')
                
                axes[i, 3].imshow(overlay)
                axes[i, 3].set_title('Overlay on Image')
                axes[i, 3].axis('off')
                
            except StopIteration:
                break
    
    plt.tight_layout()
    plt.savefig('_infer_samples.png')
    
    return fig

def visualize_saved_model(model, test_loader, num_samples=3):
    
    visualize_predictions_with_overlay(model, test_loader, num_samples)

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
    
    
    test_loader = make_dataloader('test', batch_size=1, shuffle=True)
    model = setmodel(model_path ='final_model_resnet.pth', model_type = 'resnet')
    visualize_saved_model(model, test_loader, 5)