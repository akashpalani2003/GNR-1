from __future__ import print_function
from random import shuffle
import os
import argparse
import pickle
from glob import glob
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder

from get_image_paths import get_image_paths
from get_tiny_images import get_tiny_images
from build_vocabulary import build_vocabulary
from get_bags_of_sifts import get_bags_of_sifts
from sklearn.model_selection import KFold
import joblib
from resize_linearise import resize_and_linearize_images
import numpy as np

# Step 0: Set up parameters, category list, and image paths.
parser = argparse.ArgumentParser()
parser.add_argument('--classifier', help='classifier', type=str, default='mlp')
args = parser.parse_args()

DATA_PATH = '../data/UCMerced_LandUse/Images/'
CATEGORIES = ['agricultural', 'airplane', 'baseballdiamond', 'beach', 'buildings',
              'chaparral', 'denseresidential', 'forest', 'freeway', 'golfcourse',
              'harbor', 'intersection', 'mediumresidential', 'mobilehomepark',
              'overpass', 'parkinglot', 'river', 'runway', 'sparseresidential',
              'storagetanks', 'tenniscourt']

CATE2ID = {v: k for k, v in enumerate(CATEGORIES)}

ABBR_CATEGORIES = ['agr', 'air', 'base', 'bea', 'bui', 'cha', 'den', 'for', 'fre',
                   'gol', 'har', 'int', 'med', 'mob', 'ove', 'par', 'riv', 'run',
                   'spa', 'sto', 'ten']
NUM_TRAIN_PER_CAT=70


CLASSIFIER = args.classifier

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, activation,num_classes=len(CATEGORIES)):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, num_classes)

        self.activation=activation 
        
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.sigmoid1=nn.Sigmoid()
        self.sigmoid2=nn.Sigmoid()
        self.tanh1 = nn.Tanh()
        self.tanh2 = nn.Tanh()

    def forward(self, x):

        if(self.activation=='relu'):
            x = self.relu1(self.fc1(x))
            x = self.relu2(self.fc2(x))
            x = self.fc3(x)
        
        if(self.activation=='linear'):
            x = self.fc1(x)
            x = self.fc2(x)
            x = self.fc3(x)

        if(self.activation=='sigmoid'):
            x = self.sigmoid1(self.fc1(x))
            x = self.sigmoid1(self.fc2(x))
            x = self.fc3(x)
        
        if(self.activation == 'tanh'):
            x = self.tanh1(self.fc1(x))  # Apply tanh on first layer
            x = self.tanh2(self.fc2(x))  # Apply tanh on second layer
            x = self.fc3(x)

        return x


def train_mlp(train_feats, train_labels, num_epochs=200, lr=0.001):
    le = LabelEncoder()
    train_labels = le.fit_transform(train_labels)
    joblib.dump(le, "label_encoder.pkl")
    train_feats=np.array(train_feats)
    train_feats = torch.tensor(train_feats, dtype=torch.float32)
    train_labels = torch.tensor(train_labels, dtype=torch.long)
   

    train_dataset = TensorDataset(train_feats, train_labels)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    HiddenLayers=[(512, 256), (1024,512), (2048,1024)]
    Activations=['relu','tanh','linear']

    for i in HiddenLayers:
        for j in Activations:

            model = MLP(input_size=train_feats.shape[1], hidden_size1=i[0], hidden_size2=i[1],activation=j)
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=lr)

            print(f"Training MLP for Hidden layer {i} and activation {j}...")
            for epoch in range(num_epochs):
                for features, labels in train_loader:
                    optimizer.zero_grad()
                    outputs = model(features)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

                print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}')
            os.makedirs('../store', exist_ok=True)
            torch.save(model.state_dict(), f'../store/mlp_model_{i[0]}-{j}.pth')

def train_mlp1(train_feats, train_labels, num_epochs=200, lr=0.001):
    le = LabelEncoder()
    train_labels = le.fit_transform(train_labels)
    joblib.dump(le, "label_encoder.pkl")  # Save label encoder
    
    # Convert features and labels to tensors (without requires_grad)
    train_feats = torch.tensor(train_feats, dtype=torch.float32)
    train_labels = torch.tensor(train_labels, dtype=torch.long)

    # Check if tensors require gradients
    print(f"train_feats requires_grad: {train_feats.requires_grad}")
    print(f"train_labels requires_grad: {train_labels.requires_grad}")

    # Create a DataLoader for the training data
    train_dataset = TensorDataset(train_feats, train_labels)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    # Define device (GPU or CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    HiddenLayers = [(512, 256), (256,128), (1024,512)]
    Activations = ['relu', 'linear', 'sigmoid', 'tanh']

    for i in HiddenLayers:
        for j in Activations:
            model = MLP(input_size=train_feats.shape[1], hidden_size1=i[0], hidden_size2=i[1], activation=j)
            
            # Check model parameters for requires_grad
            for name, param in model.named_parameters():
                print(f'{name} requires grad: {param.requires_grad}')

            criterion = nn.CrossEntropyLoss()  # Loss function
            optimizer = optim.Adam(model.parameters(), lr=lr)  # Optimizer

            # Move model to the correct device
            model = model.to(device)

            print(f"Training MLP for Hidden layer {i} and activation {j}...")

            for epoch in range(num_epochs):
                for features, labels in train_loader:
                    optimizer.zero_grad()

                    # Move data to the correct device
                    features, labels = features.to(device), labels.to(device)
                    
                    # Forward pass
                    outputs = model(features)
                    loss = criterion(outputs, labels)

                    # Backward pass
                    loss.backward()

                    # Optimizer step
                    optimizer.step()

                print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}')

            # Save the model
            os.makedirs('../store', exist_ok=True)  # Ensure the directory exists
            torch.save(model.state_dict(), f'../store/mlp_model_{i[0]}-{j}.pth')


   
from sklearn.model_selection import KFold

def cross_validate_mlp(val_feats, val_labels, num_folds=5, num_epochs=200, lr=0.001):
    
    le = joblib.load("label_encoder.pkl")
    val_labels = le.transform(val_labels)

    val_feats = torch.tensor(val_feats, dtype=torch.float32)
    val_labels = torch.tensor(val_labels, dtype=torch.long)
    
    HiddenLayers=[(512, 256), (256,128), (1024,512)]
    Activations=['relu','linear','sigmoid']

    accuracies = {}
    best_acc=0
    best_pair,layer,act=' ',' ',' '
   
    for i in HiddenLayers:
        for j in Activations:

            store=f'{i[0]}-{j}'

            model=torch.load(f'../store/mlp_model_{store}.pth')
            model.eval()
            with torch.no_grad():
                predictions = model(val_feats).argmax(dim=1)
                accuracy = (predictions == val_labels).float().mean().item()
            
                accuracies[store]=accuracy

            print(f'Validation Accuracy (MLP) for pair {store}: {accuracy:.4f}')

            if(accuracy>best_acc):
                best_acc=accuracy
                best_pair=store 
                layer=i
                act=j
    
    print(f'Best Validation Accuracy shows for {best_pair} pair = {best_acc}')
    print('STARTING TEST ON BEST PARAMETERS')

def cross_validate_mlp1(val_feats, val_labels, num_folds=5, num_epochs=200, lr=0.001):
    
    # Load the label encoder
    le = joblib.load("label_encoder.pkl")
    val_labels = le.transform(val_labels)
    val_feats=np.array(val_feats)
    val_feats = torch.tensor(val_feats, dtype=torch.float32)
    val_labels = torch.tensor(val_labels, dtype=torch.long)

    # Define hyperparameters for hidden layers and activation functions
    HiddenLayers=[(512, 256), (1024,512), (2048,1024)]
    Activations=['relu','tanh','linear']

    accuracies = {}
    best_acc = 0
    best_pair, layer, act = ' ', ' ', ' '

    for i in HiddenLayers:
        for j in Activations:
            store = f'{i[0]}-{j}'

            # Re-initialize the model first with the same architecture
            model = MLP(input_size=val_feats.shape[1], hidden_size1=i[0], hidden_size2=i[1], activation=j)

            # Load the state dict (weights) into the model
            model.load_state_dict(torch.load(f'../store/mlp_model_{store}.pth'))

            # Set the model to evaluation mode
            model.eval()

            # Move the model to the correct device (CUDA or CPU)
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model = model.to(device)

            # Evaluate the model
            with torch.no_grad():
                predictions = model(val_feats.to(device)).argmax(dim=1)
                accuracy = (predictions == val_labels.to(device)).float().mean().item()

            accuracies[store] = accuracy

            # Print accuracy for this configuration
            print(f'Validation Accuracy (MLP) for pair {store}: {accuracy:.4f}')

            # Update the best accuracy and corresponding parameters
            if accuracy > best_acc:
                best_acc = accuracy
                best_pair = store
                layer = i
                act = j

    # Print the best validation accuracy and corresponding model parameters
    print(f'Best Validation Accuracy shows for {best_pair} pair = {best_acc}')
    print(f'Best Pair = {best_pair}, layer = {layer}, act = {act}')
    print('STARTING TEST ON BEST PARAMETERS')
    return best_pair, layer, act

def test_mlp(test_feats, test_labels, best_model_path, hidden_size1, hidden_size2, activation ):
    # Load the best model configuration from cross-validation
    #best_model_path = '../store/mlp_model_512-relu.pth'  # Example model path for best configuration
    
    model = MLP(input_size=test_feats.shape[1], hidden_size1=hidden_size1, hidden_size2=hidden_size2, activation=activation)

    # Load the state dict (weights)
    model.load_state_dict(torch.load(best_model_path))

    # Set the model to evaluation mode
    model.eval()

    # Move the model to the appropriate device (CUDA or CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # Convert test data to tensor and move to the same device
    test_feats = torch.tensor(test_feats, dtype=torch.float32)
    test_labels = torch.tensor(test_labels, dtype=torch.long)
   


    # Evaluate the model
    with torch.no_grad():
        predictions = model(test_feats).argmax(dim=1)  # Get the class with the highest probability
        accuracy = (predictions == test_labels).float().mean().item()  # Compute accuracy

    print(f'Test Accuracy: {accuracy:.4f}')

    # Optionally, save the predictions for further analysis
    # torch.save(predictions, 'test_predictions.pth')

    return accuracy
from sklearn.preprocessing import LabelEncoder
import torch

def test_mlp1(test_feats, test_labels, model_path, hidden_size1, hidden_size2, activation):
    # Load the saved LabelEncoder used in training
    le = joblib.load("label_encoder.pkl")

    # Convert the string labels to numerical values
    test_labels = le.transform(test_labels)

    # Convert features and labels to tensors
    test_feats = torch.tensor(test_feats, dtype=torch.float32).to(device)
    test_labels = torch.tensor(test_labels, dtype=torch.long).to(device)

    # Initialize the model
    model = MLP(input_size=test_feats.shape[1], hidden_size1=hidden_size1, hidden_size2=hidden_size2, activation=activation)
    
    # Load the model's trained weights
    model.load_state_dict(torch.load(model_path))
    
    # Set the model to evaluation mode
    model.eval()

    # Perform inference
    with torch.no_grad():
        outputs = model(test_feats)
        predictions = outputs.argmax(dim=1)

        # Calculate accuracy
        accuracy = (predictions == test_labels).float().mean().item()
        print(f'Test Accuracy: {accuracy:.4f}')

def test_mlp2(test_feats, test_labels, model_path, hidden_size1, hidden_size2, activation):
    # Set device to GPU if available, otherwise use CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the saved LabelEncoder used in training
    le = joblib.load("label_encoder.pkl")

    # Convert the string labels to numerical values
    test_labels = le.transform(test_labels)

    # Convert features and labels to tensors and move to the chosen device (CPU or GPU)
    test_feats=np.array(test_feats)
    test_feats = torch.tensor(test_feats, dtype=torch.float32).to(device)
    test_labels = torch.tensor(test_labels, dtype=torch.long).to(device)

    # Initialize the model
    model = MLP(input_size=test_feats.shape[1], hidden_size1=hidden_size1, hidden_size2=hidden_size2, activation=activation)
    
    # Load the model's trained weights
    model.load_state_dict(torch.load(model_path))
    
    # Move model to the selected device
    model.to(device)
    
    # Set the model to evaluation mode
    model.eval()

    # Perform inference
    with torch.no_grad():
        outputs = model(test_feats)
        predictions = outputs.argmax(dim=1)

        # Calculate accuracy
        accuracy = (predictions == test_labels).float().mean().item()
        print(f'Test Accuracy: {accuracy:.4f}')


def main():
    print("Getting paths and labels for all train and test data")
    train_image_paths, test_image_paths, val_image_paths, train_labels, test_labels, val_labels = \
        get_image_paths(DATA_PATH, CATEGORIES, NUM_TRAIN_PER_CAT)
    print(len(train_image_paths))

    train_image_feats = resize_and_linearize_images(train_image_paths)
    val_image_feats = resize_and_linearize_images(val_image_paths)
    test_image_feats = resize_and_linearize_images(test_image_paths)
    

    store = {}
    if CLASSIFIER == 'mlp':
        train_mlp(train_image_feats,train_labels)
        a,b,c=cross_validate_mlp1(val_image_feats,val_labels)
        test_mlp2(test_image_feats, test_labels, '../store/mlp_model_' + a + '.pth', b[0], b[1],c)


if __name__ == '__main__':
    main()