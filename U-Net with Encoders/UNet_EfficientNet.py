import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import segmentation_models_pytorch as smp
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm

torch.backends.cudnn.benchmark = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 54


def load_dataset(dataset_dir):
    dataset_list = [cv2.imread(str(image_file), cv2.IMREAD_GRAYSCALE) 
                    for image_file in sorted(Path(dataset_dir).glob("*.png"),
                                             key=lambda f: int(f.stem.split("_")[-1]))]
    return np.stack(dataset_list, axis=0)


class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, preds, targets):
        preds = torch.softmax(preds, dim=1)
        one_hot_targets = torch.nn.functional.one_hot(targets, num_classes=NUM_CLASSES + 1).permute(0, 3, 1, 2).float()
        intersection = (preds * one_hot_targets).sum(dim=(2, 3))
        dice = (2. * intersection + self.smooth) / (preds.sum(dim=(2, 3)) + one_hot_targets.sum(dim=(2, 3)) + self.smooth)
        return 1 - dice.mean()


class_weights = torch.tensor([1.0] + [5.0] * NUM_CLASSES).to(device)

def combined_loss(preds, targets):
    return 0.5 * nn.CrossEntropyLoss(weight=class_weights)(preds, targets) + 0.5 * DiceLoss()(preds, targets)


def unet_model(num_classes=NUM_CLASSES + 1):
    model = smp.Unet(
        encoder_name="timm-efficientnet-b4",
        encoder_weights=None,
        in_channels=1,
        classes=num_classes
    ).to(device)
    return model

def calculate_dice_score(preds, targets, num_classes=NUM_CLASSES + 1):
    preds_softmax = torch.softmax(preds, dim=1)
    preds_argmax = torch.argmax(preds_softmax, dim=1)
    dice_scores = []
    for class_id in range(1, num_classes):
        pred_mask = (preds_argmax == class_id).float()
        target_mask = (targets == class_id).float()
        intersection = (pred_mask * target_mask).sum()
        union = pred_mask.sum() + target_mask.sum()
        dice_score = (2. * intersection + 1e-7) / (union + 1e-7)
        dice_scores.append(dice_score.item())
    return np.mean(dice_scores)



def train_model(model, train_loader, val_loader, num_epochs=200, lr=1e-3, patience=5):
    print(f"Training on {device}")
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
    best_val_loss = float('inf')
    patience_counter = 0
    scaler = torch.amp.GradScaler()

    for epoch in tqdm(range(num_epochs)):
        model.train()
        total_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            with torch.amp.autocast('cuda'):
                preds = model(x)
                loss = combined_loss(preds, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()

        val_loss = 0
        model.eval()
        dice_scores = []
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                preds = model(x)
                loss = combined_loss(preds, y)
                val_loss += loss.item()
                if epoch % 10 == 0:
                    dice_score = calculate_dice_score(preds, y)
                    dice_scores.append(dice_score)

        val_loss /= len(val_loader)
        print(f"Epoch {epoch+1}, Training Loss: {total_loss / len(train_loader)}, Validation Loss: {val_loss}")

        if epoch % 10 == 0 and dice_scores:
            avg_dice_score = np.mean(dice_scores)
            print(f"Epoch {epoch+1}, Average Dice Score: {avg_dice_score:.4f}")

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), "unet.pth")
            print("Model saved as unet.pth")
        else:
            patience_counter += 1
        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

    return model
