# GNR 638 Assignment-1,2

This is the submission for GNR638 Assignment 1,2

### Group members

- Akash Sansugu Palaniswami (21D171001)
- Rahul B (22B3976)

# Assignment 1
## Overview
In this project we made use of an existing github repo using tiny images and bag of sift features, along with nearest neighbor and SVM classifiers to perform classification. We used the UCMerced Land Use dataset which contained 2100 images of 21 different categories. Images were divided as 70% for training, 20% for testing and 10% for validation. The size of vocabulary was varied in bag of sifts to find the one with maximum accuracy

## Steps to run code

```bash
python proj3.py  -- feature tiny_image --classifier nearest_neighbor
```

Change the feature to either tiny_image or bag_of_sift and classifier to either `nearest_neighbor` or `support_vector_machine` as required 

## Results
### Confusion Matrix and Accuracy for Tiny Images with Nearest Neighbor Classifier
![TINY IMAGES](./results/Tiny_image.png)
![Accuracy for Tiny Images](./results/acc_tiny.png)

### Confusion Matrix and Accuracy for Bag of Sift with Nearest Neighbor Classifier
![Bag](./results/bag_sift_nearest.png)
![Accuracy for bag nearest](./results/acc_nearest.png)

### Confusion Matrix and Accuracy for Bag of Sift with SVM Classifier
![Bag svm](./results/bag_sift_svm.png)
![Accuracy for bag svm](./results/acc_svm.png)

### Variation of Accuracy with Change in Vocabulary size
![val nearest](./results/val_sift_nearest.png)
![val svm](./results/val_sift_svm.png)

### t-SNE Visualisation
![SNE 1](./results/t-SNE-agriculture.png)
![SNE 2](./results/t-SNE-buildings.png)
![SNE 3](./results/t-SNE-forest.png)
![SNE 4](./results/t-SNE-golfcourse.png)
![SNE 5](./results/t-SNE-overpass.png)
![SNE 6](./results/t-SNE-runway.png)


# Assignment 2
## Overview
The goal of this assignment was to use the UCMerced Land dataset for classification problems, using a multi layered perceptron. In the first part we used the features from bag of sift, while in the second we resized the images to 72 by 72 then linearised it and used it as features. The MLP used contains 2 hidden layers which we vary in size

## Part 1
File mlp_test.py is to be run for this part. The features used were taken as the same features that were obtained from bag of sift in the first assignment. Vocab size of 600 was used as logically the higher amount of features should give us better results. MLP models are to be tested for the following pairs of hidden layer values: [(512, 256), (256,128), (1024,512), (2048,1024)]. The different activation functions tested are: ['relu','tanh','linear','sigmoid']
This gives us a total of 16 different combinations to test, to determine which gives the best output using validation. A cross vaidation technique was used to determine which of the above combinations proved to be best. The accuracy came out to be between 60-70% for most of them. When running this code, an input as --classifier mlp should be given along with the run command. 
```bash
python mlp_test.py --classifier mlp
```

### Important Parts of Code
We define a class called MLP in order to adjust and tamper with parameters such as activation and hidden layers of the model. Each activation function that we test on must be defined here
```bash
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
```
Train and crossvalidation functions are defined such that they are put in a for loop until all 16 combination of size and activation are tried. All features are transformed into tensors to be used in MLP 
```bash
train_feats = torch.tensor(train_feats, dtype=torch.float32)
    train_labels = torch.tensor(train_labels, dtype=torch.long)
   

    train_dataset = TensorDataset(train_feats, train_labels)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    HiddenLayers=[(512, 256), (256,128), (1024,512), (2048,1024)]
    Activations=['relu','tanh','linear','sigmoid']

    for i in HiddenLayers:
        for j in Activations:
```
Model with best value of accuracy from cross validation is then used for testing

## Results
![Ass2Part1](./results/Assignment2_Part1.png)

The best validation accuracy is shown for the maximum hidden layers, which is as expected and the activation function is linear. When tested with this same setup of the MLP we obtain an accuracy of 67.14%, which is around 2% less than what we achieved in validation.




