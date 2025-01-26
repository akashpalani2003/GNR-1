# GNR 638 Assignment-1

This is the submission for GNR638 Assignment 1,

### Group members

- Akash Sansugu Palaniswami (21D171001)
- Rahul B (22B3976)


## Overview
In this project we made use of an existing github repo using tiny images and bag of sift features, along with nearest neighbor and SVM classifiers to perform classification. We used the UCMerce Land Use dataset which contained 2100 images of 21 different categories. Images were divided as 70% for training, 20% for testing and 10% for validation. The size of vocabulary was varied in bag of sifts to find the one with maximum accuracy

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









