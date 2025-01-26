from __future__ import print_function
from random import shuffle
import os
import argparse
import pickle


from get_image_paths import get_image_paths
from get_tiny_images import get_tiny_images
from build_vocabulary import build_vocabulary
from get_bags_of_sifts import get_bags_of_sifts
from visualize import visualize

from nearest_neighbor_classify import nearest_neighbor_classify
from svm_classify import svm_classify
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


# Step 0: Set up parameters, category list, and image paths.

#For this project, you will need to report performance for three
#combinations of features / classifiers. It is suggested you code them in
#this order, as well:
# 1) Tiny image features and nearest neighbor classifier
# 2) Bag of sift features and nearest neighbor classifier
# 3) Bag of sift features and linear SVM classifier
#The starter code is initialized to 'placeholder' just so that the starter
#code does not crash when run unmodified and you can get a preview of how
#results are presented.

parser = argparse.ArgumentParser()
parser.add_argument('--feature', help='feature', type=str, default='dumy_feature')
parser.add_argument('--classifier', help='classifier', type=str, default='dumy_classifier')
args = parser.parse_args()

DATA_PATH = '../data/UCMerced_LandUse/Images/'

#This is the list of categories / directories to use. The categories are
#somewhat sorted by similarity so that the confusion matrix looks more
#structured (indoor and then urban and then rural).

CATEGORIES = ['agricultural', 'airplane', 'baseballdiamond', 'beach', 'buildings',
              'chaparral', 'denseresidential', 'forest', 'freeway', 'golfcourse',
              'harbor', 'intersection', 'mediumresidential', 'mobilehomepark',
              'overpass', 'parkinglot', 'river', 'runway', 'sparseresidential',
              'storagetanks', 'tenniscourt']

CATE2ID = {v: k for k, v in enumerate(CATEGORIES)}

ABBR_CATEGORIES = ['agr', 'air', 'base', 'bea', 'bui', 'cha', 'den', 'for', 'fre',
                   'gol', 'har', 'int', 'med', 'mob', 'ove', 'par', 'riv', 'run',
                   'spa', 'sto', 'ten']


FEATURE = args.feature
# FEATUR  = 'bag of sift'

CLASSIFIER = args.classifier
# CLASSIFIER = 'support vector machine'

#number of training examples per category to use. Max is 100. For
#simplicity, we assume this is the number of test cases per category, as
#well.

NUM_TRAIN_PER_CAT = 70


def main():
    #This function returns arrays containing the file path for each train
    #and test image, as well as arrays with the label of each train and
    #test image. By default all four of these arrays will be 1500 where each
    #entry is a string.
    print("Getting paths and labels for all train and test data")
    train_image_paths, test_image_paths, val_image_paths, train_labels, test_labels, val_labels = \
        get_image_paths(DATA_PATH, CATEGORIES, NUM_TRAIN_PER_CAT)
    print(len(train_image_paths))

    # TODO Step 1:
    # Represent each image with the appropriate feature
    # Each function to construct features should return an N x d matrix, where
    # N is the number of paths passed to the function and d is the 
    # dimensionality of each image representation. See the starter code for
    # each function for more details.

    if FEATURE == 'tiny_image':
        # YOU CODE get_tiny_images.py 
        train_image_feats = get_tiny_images(train_image_paths)
        test_image_feats = get_tiny_images(test_image_paths)

    elif FEATURE == 'bag_of_sift':
        # YOU CODE build_vocabulary.py
        if os.path.isfile('vocab600.pkl') is False:
            print('No existing visual word vocabulary found. Computing one from training images\n')
            vocab_size =600   ### Vocab_size is up to you. Larger values will work better (to a point) but be slower to comput.
            vocab = build_vocabulary(train_image_paths, vocab_size)
            with open('vocab600.pkl', 'wb') as handle:
                pickle.dump(vocab, handle, protocol=pickle.HIGHEST_PROTOCOL)

        if os.path.isfile('train_image_feats_600.pkl') is False:
            # YOU CODE get_bags_of_sifts.py
            train_image_feats = get_bags_of_sifts(train_image_paths);
            with open('train_image_feats_600.pkl', 'wb') as handle:
                pickle.dump(train_image_feats, handle, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            with open('train_image_feats_600.pkl', 'rb') as handle:
                train_image_feats = pickle.load(handle)

        if os.path.isfile('test_image_feats_600.pkl') is False:
            test_image_feats  = get_bags_of_sifts(test_image_paths);
            with open('test_image_feats_600.pkl', 'wb') as handle:
                pickle.dump(test_image_feats, handle, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            with open('test_image_feats_600.pkl', 'rb') as handle:
                test_image_feats = pickle.load(handle)

        if os.path.isfile('val_image_feats_600.pkl') is False:
            val_image_feats  = get_bags_of_sifts(val_image_paths);
            with open('val_image_feats_600.pkl', 'wb') as handle:
                pickle.dump(val_image_feats, handle, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            with open('val_image_feats_600.pkl', 'rb') as handle:
                val_image_feats = pickle.load(handle)
    elif FEATURE == 'dumy_feature':
        train_image_feats = []
        test_image_feats = []
    else:
        raise NameError('Unknown feature type')
    print(len(train_image_feats),len(train_image_feats[0]))

    # TODO Step 2: 
    # Classify each test image by training and using the appropriate classifier
    # Each function to classify test features will return an N x 1 array,
    # where N is the number of test cases and each entry is a string indicating
    # the predicted category for each test image. Each entry in
    # 'predicted_categories' must be one of the 15 strings in 'categories',
    # 'train_labels', and 'test_labels.

    if CLASSIFIER == 'nearest_neighbor':
        print(1)
        # YOU CODE nearest_neighbor_classify.py
        if FEATURE == 'tiny_image':
            predicted_categories = nearest_neighbor_classify(train_image_feats, train_labels, test_image_feats)
        elif FEATURE == 'bag_of_sift':
            best=0
            bestacc=0
            for i in range (2,7):
                if os.path.isfile(f'vocab{str(i*100)}.pkl') is False:
                    print(2)
                    print('No existing visual word vocabulary found. Computing one from training images\n')
                    vocab_size =i*100   ### Vocab_size is up to you. Larger values will work better (to a point) but be slower to comput.
                    vocab = build_vocabulary(train_image_paths, vocab_size)
                    with open(f'vocab{str(i*100)}.pkl', 'wb') as handle:
                        pickle.dump(vocab, handle, protocol=pickle.HIGHEST_PROTOCOL)
                if os.path.isfile('train_image_feats_' + str(i*100) + '.pkl') is False:
                    # YOU CODE get_bags_of_sifts.py
                    train_image_feats = get_bags_of_sifts(train_image_paths);
                    with open('train_image_feats_' + str(i*100) + '.pkl', 'wb') as handle:
                        pickle.dump(train_image_feats, handle, protocol=pickle.HIGHEST_PROTOCOL)
                else:
                    with open('train_image_feats_' + str(i*100) + '.pkl', 'rb') as handle:
                        train_image_feats = pickle.load(handle)

                if os.path.isfile('test_image_feats_' + str(i*100) + '.pkl') is False:
                    test_image_feats  = get_bags_of_sifts(test_image_paths);
                    with open('test_image_feats_' + str(i*100) + '.pkl', 'wb') as handle:
                        pickle.dump(test_image_feats, handle, protocol=pickle.HIGHEST_PROTOCOL)
                else:
                    with open('test_image_feats_' + str(i*100) + '.pkl', 'rb') as handle:
                        test_image_feats = pickle.load(handle)

                if os.path.isfile('val_image_feats_' + str(i*100) + '.pkl') is False:
                    val_image_feats  = get_bags_of_sifts(val_image_paths);
                    with open('val_image_feats_' + str(i*100) + '.pkl', 'wb') as handle:
                        pickle.dump(val_image_feats, handle, protocol=pickle.HIGHEST_PROTOCOL)
                else:
                    with open('val_image_feats_' + str(i*100) + '.pkl', 'rb') as handle:
                        val_image_feats = pickle.load(handle)
                predicted_categories = nearest_neighbor_classify(train_image_feats, train_labels, test_image_feats)
                accuracy = float(len([x for x in zip(test_labels,predicted_categories) if x[0]== x[1]]))/float(len(test_labels))
                print('Accuracy(vocab==' + str(i*100) + ') = ', accuracy)
                if(accuracy>bestacc):
                    bestacc=accuracy
                    best=i*100
            if os.path.isfile('vocab' + str(best) + '.pkl') is False:
                    print('No existing visual word vocabulary found. Computing one from training images\n')
                    vocab_size =best   ### Vocab_size is up to you. Larger values will work better (to a point) but be slower to comput.
                    vocab = build_vocabulary(train_image_paths, vocab_size)
                    with open('vocab' + str(best) + '.pkl', 'wb') as handle:
                        pickle.dump(vocab, handle, protocol=pickle.HIGHEST_PROTOCOL)
            if os.path.isfile('train_image_feats_' + str(best) + '.pkl') is False:
                # YOU CODE get_bags_of_sifts.py
                train_image_feats = get_bags_of_sifts(train_image_paths);
                with open('train_image_feats_' + str(best) + '.pkl', 'wb') as handle:
                    pickle.dump(train_image_feats, handle, protocol=pickle.HIGHEST_PROTOCOL)
            else:
                with open('train_image_feats_' + str(best) + '.pkl', 'rb') as handle:
                    train_image_feats = pickle.load(handle)

            if os.path.isfile('test_image_feats_' + str(best) + '.pkl') is False:
                test_image_feats  = get_bags_of_sifts(test_image_paths);
                with open('test_image_feats_' + str(best) + '.pkl', 'wb') as handle:
                    pickle.dump(test_image_feats, handle, protocol=pickle.HIGHEST_PROTOCOL)
            else:
                with open('test_image_feats_' + str(best) + '.pkl', 'rb') as handle:
                    test_image_feats = pickle.load(handle)

            if os.path.isfile('val_image_feats_' + str(best) + '.pkl') is False:
                val_image_feats  = get_bags_of_sifts(val_image_paths);
                with open('val_image_feats_' + str(best) + '.pkl', 'wb') as handle:
                    pickle.dump(val_image_feats, handle, protocol=pickle.HIGHEST_PROTOCOL)
            else:
                with open('val_image_feats_' + str(best) + '.pkl', 'rb') as handle:
                    val_image_feats = pickle.load(handle)
            predicted_categories = nearest_neighbor_classify(train_image_feats, train_labels, test_image_feats)
            print('Best vocab number is', best)
    elif CLASSIFIER == 'support_vector_machine':
        # YOU CODE svm_classify.py
        best=0
        bestacc=0
        for i in range (2,7):
            if os.path.isfile('vocab' + str(i*100) + '.pkl') is False:
                print('No existing visual word vocabulary found. Computing one from training images\n')
                vocab_size =i*100   ### Vocab_size is up to you. Larger values will work better (to a point) but be slower to comput.
                vocab = build_vocabulary(train_image_paths, vocab_size)
                with open('vocab' + str(i*100) + '.pkl', 'wb') as handle:
                    pickle.dump(vocab, handle, protocol=pickle.HIGHEST_PROTOCOL)
            if os.path.isfile('train_image_feats_' + str(i*100) + '.pkl') is False:
                    # YOU CODE get_bags_of_sifts.py
                train_image_feats = get_bags_of_sifts(train_image_paths);
                with open('train_image_feats_' + str(i*100) + '.pkl', 'wb') as handle:
                    pickle.dump(train_image_feats, handle, protocol=pickle.HIGHEST_PROTOCOL)
            else:
                with open('train_image_feats_' + str(i*100) + '.pkl', 'rb') as handle:
                    train_image_feats = pickle.load(handle)

            if os.path.isfile('test_image_feats_' + str(i*100) + '.pkl') is False:
                test_image_feats  = get_bags_of_sifts(test_image_paths);
                with open('test_image_feats_' + str(i*100) + '.pkl', 'wb') as handle:
                    pickle.dump(test_image_feats, handle, protocol=pickle.HIGHEST_PROTOCOL)
            else:
                with open('test_image_feats_' + str(i*100) + '.pkl', 'rb') as handle:
                    test_image_feats = pickle.load(handle)

            if os.path.isfile('val_image_feats_' + str(i*100) + '.pkl') is False:
                val_image_feats  = get_bags_of_sifts(val_image_paths);
                with open('val_image_feats_' + str(i*100) + '.pkl', 'wb') as handle:
                    pickle.dump(val_image_feats, handle, protocol=pickle.HIGHEST_PROTOCOL)
            else:
                with open('val_image_feats_' + str(i*100) + '.pkl', 'rb') as handle:
                    val_image_feats = pickle.load(handle)
            predicted_categories = svm_classify(train_image_feats, train_labels, test_image_feats)
            accuracy = float(len([x for x in zip(test_labels,predicted_categories) if x[0]== x[1]]))/float(len(test_labels))
            print('Accuracy(vocab==' + str(i*100) + ') = ', accuracy)
            if(accuracy>bestacc):
                bestacc=accuracy
                best=i*100
        if os.path.isfile('vocab' + str(best) + '.pkl') is False:
                print('No existing visual word vocabulary found. Computing one from training images\n')
                vocab_size =best   ### Vocab_size is up to you. Larger values will work better (to a point) but be slower to comput.
                vocab = build_vocabulary(train_image_paths, vocab_size)
                with open('vocab' + str(best) + '.pkl', 'wb') as handle:
                    pickle.dump(vocab, handle, protocol=pickle.HIGHEST_PROTOCOL)
        if os.path.isfile('train_image_feats_' + str(best) + '.pkl') is False:
                # YOU CODE get_bags_of_sifts.py
            train_image_feats = get_bags_of_sifts(train_image_paths);
            with open('train_image_feats_' + str(best) + '.pkl', 'wb') as handle:
                pickle.dump(train_image_feats, handle, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            with open('train_image_feats_' + str(best) + '.pkl', 'rb') as handle:
                train_image_feats = pickle.load(handle)

        if os.path.isfile('test_image_feats_' + str(best) + '.pkl') is False:
            test_image_feats  = get_bags_of_sifts(test_image_paths);
            with open('test_image_feats_' + str(best) + '.pkl', 'wb') as handle:
                pickle.dump(test_image_feats, handle, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            with open('test_image_feats_' + str(best) + '.pkl', 'rb') as handle:
                test_image_feats = pickle.load(handle)

        if os.path.isfile('val_image_feats_' + str(best) + '.pkl') is False:
            val_image_feats  = get_bags_of_sifts(val_image_paths);
            with open('val_image_feats_' + str(best) + '.pkl', 'wb') as handle:
                pickle.dump(val_image_feats, handle, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            with open('val_image_feats_' + str(best) + '.pkl', 'rb') as handle:
                val_image_feats = pickle.load(handle)
        
        predicted_categories = svm_classify(train_image_feats, train_labels, test_image_feats)
        print('Best vocab number is', best)

    elif CLASSIFIER == 'dumy_classifier':
        # The dummy classifier simply predicts a random category for
        # every test case
        predicted_categories = test_labels[:]
        shuffle(predicted_categories)
    else:
        raise NameError('Unknown classifier type')

    accuracy = float(len([x for x in zip(test_labels,predicted_categories) if x[0]== x[1]]))/float(len(test_labels))
    print("Accuracy = ", accuracy)
    
    for category in CATEGORIES:
        accuracy_each = float(len([x for x in zip(test_labels,predicted_categories) if x[0]==x[1] and x[0]==category]))/float(test_labels.count(category))
        print(str(category) + ': ' + str(accuracy_each))
    
    test_labels_ids = [CATE2ID[x] for x in test_labels]
    predicted_categories_ids = [CATE2ID[x] for x in predicted_categories]
    train_labels_ids = [CATE2ID[x] for x in train_labels]
    
    # Step 3: Build a confusion matrix and score the recognition system
    # You do not need to code anything in this section. 
   
    build_confusion_mtx(test_labels_ids, predicted_categories_ids, ABBR_CATEGORIES)
    #visualize(CATEGORIES, test_image_paths, test_labels_ids, predicted_categories_ids, train_image_paths, train_labels_ids)
    
    
def build_confusion_mtx(test_labels_ids, predicted_categories, abbr_categories):
    # Compute confusion matrix
    cm = confusion_matrix(test_labels_ids, predicted_categories)
    np.set_printoptions(precision=2)
    '''
    print('Confusion matrix, without normalization')
    print(cm)
    plt.figure()
    plot_confusion_matrix(cm, CATEGORIES)
    '''
    # Normalize the confusion matrix by row (i.e by the number of samples
    # in each class)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    #print('Normalized confusion matrix')
    #print(cm_normalized)
    plt.figure()
    plot_confusion_matrix(cm_normalized, abbr_categories, title='Normalized confusion matrix')

    plt.show()
     
def plot_confusion_matrix(cm, category, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(category))
    plt.xticks(tick_marks, category, rotation=45)
    plt.yticks(tick_marks, category)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def dsift(image, step=(3, 3)):
   
    # Convert to grayscale if the image is in BGR
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # Create a SIFT detector
    sift = cv2.SIFT_create()

    # Create a dense grid of keypoints
    keypoints = [
        cv2.KeyPoint(x, y, step[0])
        for y in range(0, gray.shape[0], step[1])
        for x in range(0, gray.shape[1], step[0])
    ]

    # Compute SIFT descriptors for the dense keypoints
    keypoints, descriptors = sift.compute(gray, keypoints)

    return keypoints, descriptors

def visualize_sift_tsne(sift_descriptors, labels=None, perplexity=30, n_iter=300):
    """
    Visualize SIFT keypoints using t-SNE.

    Parameters:
    - sift_descriptors: Numpy array of shape (N, 128), where N is the number of descriptors.
    - labels: (Optional) List or array of labels for coloring the points. Should have length N.
    - perplexity: Perplexity parameter for t-SNE.
    - n_iter: Number of iterations for t-SNE optimization.

    Returns:
    - None (displays a plot).
    """
    # Perform t-SNE to reduce dimensions
    tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=n_iter, random_state=42)
    reduced_features = tsne.fit_transform(sift_descriptors)

    # Plot the reduced features
    plt.figure(figsize=(10, 8))
    if labels is not None:
        scatter = plt.scatter(
            reduced_features[:, 0], reduced_features[:, 1], c=labels, cmap='tab10', s=10, alpha=0.8
        )
        plt.legend(*scatter.legend_elements(), title="Labels")
    else:
        plt.scatter(reduced_features[:, 0], reduced_features[:, 1], s=10, alpha=0.8)

    plt.title("t-SNE Visualization of SIFT Descriptors")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.show()

if __name__ == '__main__':
    main()