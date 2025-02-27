import pandas as pd
from pathlib import Path
import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from PIL import Image
import torch.utils.data
from tqdm import tqdm
from sklearn.model_selection import train_test_split

torch.backends.cudnn.benchmark = True  # Optimize GPU performance

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 54

def load_dataset(dataset_dir):
    dataset_list = [cv2.imread(str(image_file), cv2.IMREAD_GRAYSCALE) for image_file in sorted(Path(dataset_dir).glob("*.png"), key=lambda f: int(f.stem.split("_")[-1]))]
    return np.stack(dataset_list, axis=0)

def dice_image(prediction, ground_truth):
    intersection = torch.sum(prediction * ground_truth)
    if torch.sum(prediction) == 0 and torch.sum(ground_truth) == 0:
        return float('nan')
    return 2 * intersection / (torch.sum(prediction) + torch.sum(ground_truth))

def dice_multiclass(prediction, ground_truth):
    dices = [dice_image((prediction == i).float(), (ground_truth == i).float()) for i in range(1, NUM_CLASSES + 1)]
    return torch.tensor(dices, device=device)

def dice_pandas(y_true_df, y_pred_df):
    y_pred_df, y_true_df = torch.tensor(y_pred_df.values, device=device), torch.tensor(y_true_df.values, device=device)
    dices = [dice_multiclass(y_true_df[row_index].view(-1), y_pred_df[row_index].view(-1)) for row_index in range(y_true_df.shape[0])]
    return torch.nanmean(torch.stack(dices)).item()

def mask_rcnn_model():
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights=torchvision.models.detection.MaskRCNN_ResNet50_FPN_Weights.DEFAULT)
    model.to(device).eval()
    return model

def apply_mask_rcnn(model, images):
    images = torch.stack([T.ToTensor()(image).to(device) for image in images])
    with torch.no_grad():
        outputs = model(images)
    masks = [(out["masks"][0, 0] > 0.5).float() if len(out["masks"]) > 0 else torch.zeros_like(images[0][0]) for out in outputs]
    return torch.stack(masks).cpu().numpy()

def deeplabv3_plus_model(num_class=NUM_CLASSES+1):
    model = torchvision.models.segmentation.deeplabv3_resnet101(weights=torchvision.models.segmentation.DeepLabV3_ResNet101_Weights.DEFAULT)
    model.classifier[-1] = nn.Conv2d(model.classifier[-1].in_channels, num_class, kernel_size=(1, 1))
    return model.to(device)

class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth
    def forward(self, preds, targets):
        preds = torch.softmax(preds, dim=1)
        one_hot_targets = torch.nn.functional.one_hot(targets, num_classes=NUM_CLASSES+1).permute(0, 3, 1, 2).float()
        intersection = (preds * one_hot_targets).sum(dim=(2,3))
        dice = (2. * intersection + self.smooth) / (preds.sum(dim=(2,3)) + one_hot_targets.sum(dim=(2,3)) + self.smooth)
        return 1 - dice.mean()

def combined_loss(preds, targets):
    return 0.5 * nn.CrossEntropyLoss()(preds, targets) + 0.5 * DiceLoss()(preds, targets)

def train_model(model, train_loader, val_loader, num_epochs=10, lr=1e-4, patience=3, resume_training=False):
    print(f"Training on {device}")

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
    best_val_loss, patience_counter = float('inf'), 0
    scaler = torch.cuda.amp.GradScaler()

    # Load the best model if resume_training is True
    if resume_training and Path("best_model.pth").exists():
        print("Resuming training from best_model.pth")
        model.load_state_dict(torch.load("best_model.pth"))

    for epoch in tqdm(range(num_epochs)):
        model.train()
        total_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                preds = model(x)["out"]
                loss = combined_loss(preds, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()

        val_loss = sum(combined_loss(model(x.to(device))["out"], y.to(device)).item() for x, y in val_loader) / len(val_loader)
        print(f"Epoch {epoch+1}, Training Loss: {total_loss / len(train_loader)}, Validation Loss: {val_loss}")
        scheduler.step(val_loss)

        if val_loss < best_val_loss * 1.01:
            best_val_loss, patience_counter = val_loss, 0
            torch.save(model.state_dict(), "best_model.pth")
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print("Early stopping triggered")
            model.load_state_dict(torch.load("best_model.pth"))
            break

    return model


class Dataset(torch.utils.data.Dataset):
    def __init__(self, X, Y, transform=None):
        self.X, self.Y, self.transform = X, Y, transform
        self.mask_rcnn_model = mask_rcnn_model()
    def __len__(self): return len(self.X)
    def __getitem__(self, index):
        x, y = Image.fromarray(self.X[index]).convert("RGB"), torch.tensor(self.Y.iloc[index].values.reshape((256, 256))).long()
        x = np.array(x) * np.expand_dims(apply_mask_rcnn(self.mask_rcnn_model, [x])[0], axis=-1)
        return (self.transform(image=x)["image"] if self.transform else T.ToTensor()(x)), y

def main():
    labels_train = pd.read_csv("y_train.csv", index_col=0).T
    data_train, data_test = load_dataset("train-images"), load_dataset("test-images")
    valid_indices = [i for i in range(len(labels_train)) if np.any(labels_train.iloc[i].values > 0)]
    data_train_split, data_val_split, labels_train_split, labels_val_split = train_test_split(data_train[valid_indices], labels_train.iloc[valid_indices], train_size=0.8, random_state=42)
    transform = A.Compose([A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5), A.RandomRotate90(p=0.5), A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), ToTensorV2()])
    train_loader = torch.utils.data.DataLoader(Dataset(data_train_split, labels_train_split, transform), batch_size=16, shuffle=True pin_memory=True)
    val_loader = torch.utils.data.DataLoader(Dataset(data_val_split, labels_val_split, transform), batch_size=16, shuffle=False, pin_memory=True)
    continue_training = True  # Set to False if you want to start training from scratch
    train_model(deeplabv3_plus_model(), train_loader, val_loader, num_epochs=150, lr=1e-5, patience=25, resume_training=continue_training)


if __name__ == "__main__": main()