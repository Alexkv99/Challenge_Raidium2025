import pandas as pd
from pathlib import Path
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from augmentations import MyDataset, UnlabeledDataset
from UNet_EfficientNet import unet_model, train_model, load_dataset
from torch.utils.data import ConcatDataset

torch.backends.cudnn.benchmark = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 54

def main():
    # Read the data and keep first 800 rows
    labels = pd.read_csv('y_train.csv', index_col=0).T.iloc[:800]

    # Create a new DataFrame with unique values and their counts for each row
    result = labels.apply(
        lambda row: pd.Series({
            'num_unique': len(row.unique())
        }),
        axis=1
    )

    condition = result['num_unique'].apply(lambda x: x > 1)
    labels_train = labels[condition]
    print(f"Number of labeled images {len(labels_train)}")
    
    labeled_indices_array = np.where(condition)[0]
    
    data_dir = Path("./")
    data_train = load_dataset(data_dir / "train-images")
    data_test = load_dataset(data_dir / "test-images")

    ## MIN MAX NORMALIZATION
    data_train = (data_train - np.min(data_train)) / (np.max(data_train) - np.min(data_train))
    data_test = (data_test - np.min(data_test)) / (np.max(data_test) - np.min(data_test))

    data_train_labeled = torch.from_numpy(data_train[labeled_indices_array]).type(torch.float32).reshape((-1, 1, 256, 256))
    data_test = torch.from_numpy(data_test).type(torch.float32).reshape((-1, 1, 256, 256))
    labels_train = torch.from_numpy(labels_train.to_numpy()).reshape((-1, 256, 256))

    ## Split for train and test data
    X_train, X_valid, y_train, y_valid = train_test_split(data_train_labeled, labels_train, test_size=0.02, random_state=0)
    
    print(f"Number of training images {len(X_train)}")
    print(f"Number of validation images {len(X_valid)}")

    # Create the training and validation datasets
    train_dataset_aug = MyDataset(X_train, y_train, augment=True)
    train_dataset = MyDataset(X_train, y_train, augment=False)
    valid_dataset = MyDataset(X_valid, y_valid)

    final_train_dataset = ConcatDataset([train_dataset_aug, train_dataset])

    # Create DataLoader for training and validation
    train_loader = DataLoader(final_train_dataset, batch_size=8, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=8, shuffle=False)

    # Model training
    model = unet_model()
    train_model(model, train_loader, valid_loader, num_epochs=30, lr=1e-3, patience=15)


if __name__ == "__main__":
    main()