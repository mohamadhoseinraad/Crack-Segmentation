import os
import time
from datetime import datetime

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from model import CustomCrackSegModel  
from Dataset import make_dataloader

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
MODEL_NAME = 'dropout'
TENSORBOARD_LOGS = f'runs/crack_segmentation_{MODEL_NAME}'
CHECKPOINT_DIR = f'checkpoints/{MODEL_NAME}'
TRAIN_DIR = 'train'
VALID_DIR = 'val'
TEST_DIR = 'test'
FINAL_MODEL_PATH = f'final_model_{MODEL_NAME}.pth'
BATCH_SIZE = 16
NUM_EPOCHS = 20
LEARNING_RATE = 1e-4


class CombinedLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, smooth=1e-6,
                 w_wce=0.9, w_focal=20.0, w_dice=0.8):
        super(CombinedLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.smooth = smooth
        self.w_wce = w_wce
        self.w_focal = w_focal
        self.w_dice = w_dice

    def weighted_cross_entropy(self, inputs, targets, weight_positive=2.0):
        if inputs.dim() == 4 and targets.dim() == 3:
            targets = targets.unsqueeze(1)
        targets = targets.float()
        inputs = torch.sigmoid(inputs)

        loss = - (weight_positive * targets * torch.log(inputs + self.smooth) +
                  (1 - targets) * torch.log(1 - inputs + self.smooth))
        return loss.mean()

    def dice_loss(self, inputs, targets):
        if inputs.dim() == 4 and targets.dim() == 3:
            targets = targets.unsqueeze(1)
        targets = targets.float()
        inputs = torch.sigmoid(inputs)

        inputs = inputs.view(-1)
        targets = targets.view(-1)
        intersection = (inputs * targets).sum()
        dice = (2. * intersection + self.smooth) / (inputs.sum() + targets.sum() + self.smooth)
        return 1 - dice

    def focal_loss(self, inputs, targets):
        if inputs.dim() == 4 and targets.dim() == 3:
            targets = targets.unsqueeze(1)
        targets = targets.float()

        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        return (self.alpha * (1 - pt) ** self.gamma * BCE_loss).mean()

    def forward(self, inputs, targets):
        if inputs.dim() == 4 and targets.dim() == 3:
            targets = targets.unsqueeze(1)

        wce_loss = self.weighted_cross_entropy(inputs, targets, weight_positive=3.0)
        focal_loss = self.focal_loss(inputs, targets)
        dice_loss = self.dice_loss(inputs, targets)

        total_loss = (self.w_wce * wce_loss +
                      self.w_focal * focal_loss +
                      self.w_dice * dice_loss)

        return total_loss, {'wce': wce_loss.item(), 'focal': focal_loss.item(), 'dice': dice_loss.item()}


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


def train_model(model, train_loader, val_loader, num_epochs=50, learning_rate=1e-4 , save_interval = 5):
    model = model.to(DEVICE)

    criterion = CombinedLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=NUM_EPOCHS,  
        eta_min=1e-6  
    )
    # TensorBoard writer
    writer = SummaryWriter(f'runs-Aug/crack_segmentation_{datetime.now().strftime("%Y%m%d_%H%M%S")}')

    history = {
        'train_loss': [], 'val_loss': [],
        'train_iou': [], 'val_iou': [],
        'train_precision': [], 'val_precision': [],
        'train_recall': [], 'val_recall': [],
        'train_f1': [], 'val_f1': [],
        'wce_loss': [], 'focal_loss': [], 'dice_loss': []
    }

    best_val_f1 = 0.0
    os.makedirs('checkpoints-Aug', exist_ok=True)

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_metrics = {'iou': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
        batch_count = 0

        for batch_idx, (images, masks) in enumerate(train_loader):
            images = images.to(DEVICE)
            masks = masks.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images)

            total_loss, loss_components = criterion(outputs, masks)
            metrics = calculate_metrics(outputs, masks)

            total_loss.backward()
            optimizer.step()

            train_loss += total_loss.item()
            for key in train_metrics:
                train_metrics[key] += metrics[key]
            batch_count += 1

            if batch_idx % 10 == 0:
                print(f'Epoch [{epoch + 1}/{num_epochs}], Batch [{batch_idx}/{len(train_loader)}], '
                      f'Loss: {total_loss.item():.4f}, IoU: {metrics["iou"]:.4f}, F1: {metrics["f1"]:.4f}')

        avg_train_loss = train_loss / batch_count
        avg_train_metrics = {k: v / batch_count for k, v in train_metrics.items()}

        model.eval()
        val_loss = 0.0
        val_metrics = {'iou': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
        val_batch_count = 0

        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(DEVICE)
                masks = masks.to(DEVICE)

                outputs = model(images)
                total_loss, loss_components = criterion(outputs, masks)
                metrics = calculate_metrics(outputs, masks)

                val_loss += total_loss.item()
                for key in val_metrics:
                    val_metrics[key] += metrics[key]
                val_batch_count += 1

        avg_val_loss = val_loss / val_batch_count
        avg_val_metrics = {k: v / val_batch_count for k, v in val_metrics.items()}

        scheduler.step()

        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['train_iou'].append(avg_train_metrics['iou'])
        history['val_iou'].append(avg_val_metrics['iou'])
        history['train_precision'].append(avg_train_metrics['precision'])
        history['val_precision'].append(avg_val_metrics['precision'])
        history['train_recall'].append(avg_train_metrics['recall'])
        history['val_recall'].append(avg_val_metrics['recall'])
        history['train_f1'].append(avg_train_metrics['f1'])
        history['val_f1'].append(avg_val_metrics['f1'])
        history['wce_loss'].append(loss_components['wce'])
        history['focal_loss'].append(loss_components['focal'])
        history['dice_loss'].append(loss_components['dice'])

        # TensorBoard logging
        writer.add_scalar('Loss/Train', avg_train_loss, epoch)
        writer.add_scalar('Loss/Validation', avg_val_loss, epoch)
        writer.add_scalar('IoU/Train', avg_train_metrics['iou'], epoch)
        writer.add_scalar('IoU/Validation', avg_val_metrics['iou'], epoch)
        writer.add_scalar('Precision/Train', avg_train_metrics['precision'], epoch)
        writer.add_scalar('Precision/Validation', avg_val_metrics['precision'], epoch)
        writer.add_scalar('Recall/Train', avg_train_metrics['recall'], epoch)
        writer.add_scalar('Recall/Validation', avg_val_metrics['recall'], epoch)
        writer.add_scalar('F1/Train', avg_train_metrics['f1'], epoch)
        writer.add_scalar('F1/Validation', avg_val_metrics['f1'], epoch)
        writer.add_scalar('Loss/WCE', loss_components['wce'], epoch)
        writer.add_scalar('Loss/Focal', loss_components['focal'], epoch)
        writer.add_scalar('Loss/Dice', loss_components['dice'], epoch)
        writer.add_scalar('Learning Rate', optimizer.param_groups[0]['lr'], epoch)

        if avg_val_metrics['f1'] > best_val_f1:
            best_val_f1 = avg_val_metrics['f1']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_iou': avg_val_metrics['iou'],
                'val_f1': best_val_f1,
                'loss': avg_val_loss,
            }, 'checkpoints/best_model.pth')

        if (epoch + 1) % save_interval == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_iou': avg_val_metrics['iou'],
                'val_f1': avg_val_metrics['f1'],
                'loss': avg_val_loss,
            }, f'checkpoints/epoch_{epoch + 1}.pth')

        print(f'Epoch [{epoch + 1}/{num_epochs}]')
        print(f'Train Loss: {avg_train_loss:.4f}, Train IoU: {avg_train_metrics["iou"]:.4f}')
        print(f'Val Loss: {avg_val_loss:.4f}, Val IoU: {avg_val_metrics["iou"]:.4f}')
        print(f'Train F1: {avg_train_metrics["f1"]:.4f}, Val F1: {avg_val_metrics["f1"]:.4f}')
        print(f'Train Precision: {avg_train_metrics["precision"]:.4f}, Train Recall: {avg_train_metrics["recall"]:.4f}')
        print(f'Val Precision: {avg_val_metrics["precision"]:.4f}, Val Recall: {avg_val_metrics["recall"]:.4f}')
        print(f'Learning Rate: {optimizer.param_groups[0]["lr"]:.2e}')
        print('-' * 50)

    writer.close()
    return history, model



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
    print("Test Results:", avg_test_metrics)
    return avg_test_metrics

def plot_training_history(history):
    """Plot training and validation metrics"""
    plt.figure(figsize=(15, 15))

    # Plot loss
    plt.subplot(3, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # Plot IoU
    plt.subplot(3, 2, 2)
    plt.plot(history['train_iou'], label='Train IoU')
    plt.plot(history['val_iou'], label='Validation IoU')
    plt.title('Training and Validation IoU')
    plt.xlabel('Epoch')
    plt.ylabel('IoU')
    plt.legend()
    plt.grid(True)

    # Plot F1 Score
    plt.subplot(3, 2, 3)
    plt.plot(history['train_f1'], label='Train F1')
    plt.plot(history['val_f1'], label='Validation F1')
    plt.title('Training and Validation F1 Score')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.legend()
    plt.grid(True)

    # Plot Precision
    plt.subplot(3, 2, 4)
    plt.plot(history['train_precision'], label='Train Precision')
    plt.plot(history['val_precision'], label='Validation Precision')
    plt.title('Training and Validation Precision')
    plt.xlabel('Epoch')
    plt.ylabel('Precision')
    plt.legend()
    plt.grid(True)

    # Plot Recall
    plt.subplot(3, 2, 5)
    plt.plot(history['train_recall'], label='Train Recall')
    plt.plot(history['val_recall'], label='Validation Recall')
    plt.title('Training and Validation Recall')
    plt.xlabel('Epoch')
    plt.ylabel('Recall')
    plt.legend()
    plt.grid(True)

    # Plot individual loss components
    plt.subplot(3, 2, 6)
    plt.plot(history['wce_loss'], label='WCE Loss', alpha=0.7)
    plt.plot(history['focal_loss'], label='Focal Loss', alpha=0.7)
    plt.plot(history['dice_loss'], label='Dice Loss', alpha=0.7)
    plt.title('Individual Loss Components')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    

    model = CustomCrackSegModel(in_channels=3, out_channels=1, dropout_p=0.3)

    train_loader = make_dataloader(f'{TRAIN_DIR}', batch_size=BATCH_SIZE, shuffle=True)
    val_loader = make_dataloader(f'{VALID_DIR}', batch_size=BATCH_SIZE, shuffle=False)
    test_loader = make_dataloader(f'{TEST_DIR}', batch_size=BATCH_SIZE, shuffle=False)

    print(f"Training on device: {DEVICE}")
    history, trained_model = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        save_interval = 5
    )
   
    plot_training_history(history)
    test_model(trained_model, test_loader)
    torch.save(trained_model.state_dict(), f'{FINAL_MODEL_PATH}')
    print("Training complete")