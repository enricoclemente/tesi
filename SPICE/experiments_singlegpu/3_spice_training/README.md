# 3rd phase for SPICE: spice training

<tr>
<td><img  height="300px" src="../figures/SPICE_framework_ab.png"></td>
</tr>

In the previous phase we trained the self-supervised model (MoCo) in order to have an Encoder capable to extract features of images (a)
In this phase we have to train the SPICE model for image clustering (b). 
SPICE model is composed with the Encoder with pretrained weights and a Clustering Head which is composed of several MLP.

The workflow of this phase is the following:
- Expectation (E) step
    - first branch takes original images and extract features from them
    - second branch takes weakly augmented images, extract features and obtain the probability of every image to belong a cluster, using a softmax
    - features of original images and probabilites of weakly agumented are used to create a prototype labeling: 
        - using probabilities, the best are selected for every cluster and so the features of the relative orginal images are used as cluster center (if there are more images, mean is computed obtaining a unique feature)
        - using clustering centers, cosine similarity is computed to retrieve which image belongs to a certain cluster --> prototype labels (ground truth)

- Maximization (M) step


## Script
train_spice.py

## Description


## Arguments
