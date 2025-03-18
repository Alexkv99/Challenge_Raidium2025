# Deep Learning Challenge - ENS ChallengeData Platform

## Challenge: **Weakly Supervised Semantic Segmentation by Raidium**

This repository contains our submission for the [Weakly Supervised Semantic Segmentation Challenge](https://challengedata.ens.fr/participants/challenges/165/), organized as part of the *Sciences des Données* course at Collège de France, chaired by [Stéphane Mallat](https://www.college-de-france.fr/fr/chaire/stephane-mallat-sciences-des-donnees-chaire-statutaire). This project is undertaken as part of the IASD curriculum at Université Paris Dauphine-PSL.

## Team Members
- **Alexandros Kouvatseas**
- **Elhadi Chiter**

## Project Overview
The challenge focuses on weakly supervised semantic segmentation, where the goal is to segment objects in images using limited annotations. We developed a deep learning-based approach leveraging convolutional neural networks (CNNs) and attention mechanisms to improve segmentation quality while maintaining efficiency.

# Data Download Instructions

To download and extract the dataset, run the following commands in a terminal:

```bash
mkdir -p Data
cd Data
wget https://challengedata.ens.fr/media/public/train-images.zip
wget https://challengedata.ens.fr/media/public/test-images.zip
wget https://challengedata.ens.fr/media/public/label_Hnl61pT.csv -O y_train.csv

unzip -q train-images.zip
unzip -q test-images.zip
```

These commands will:
- Create a `Data` directory if it does not already exist.
- Download the dataset files into the `Data` directory.
- Extract the `train-images.zip` and `test-images.zip` archives.

Ensure you have `wget` and `unzip` installed on your system. If not, install them using:
- **Ubuntu/Linux**: `sudo apt install wget unzip`
- **MacOS (Homebrew)**: `brew install wget unzip`


## Approach
### 1. Data Preprocessing
- What are we exactly supposed to do here ? 

### 2. Model Architecture
- TODO

### 3. Training Strategy
- TODO

### 4. Evaluation Metrics
- TODO

## Results
- TODO
## Installation & Usage
### Prerequisites
- TODO
### Setup
```bash
# Clone the repository
TODO

# Install dependencies
pip install -r requirements.txt
```

### Training the Model
```bash
TODO
```

### Running Inference
```bash
TODO
```

## Acknowledgments
We would like to thank **Collège de France** and **École Normale Supérieure** for organizing this challenge. Special thanks to **Stéphane Mallat** and the IASD program for providing this opportunity to work on advanced deep learning tasks.
