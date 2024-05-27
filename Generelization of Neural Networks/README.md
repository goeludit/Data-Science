## Exploring Generalization of Neural Networks

This project implements a convolutional neural network (CNN) for image classification on the CIFAR-10 dataset, exploring the impact of label noise on model generalization. It aims to replicate and extend the experiments from the paper "Understanding Deep Learning (Still) Requires Rethinking Generalization" by Chiyuan Zhang, Samy Bengio, and Moritz Hardt. (https://dl.acm.org/doi/pdf/10.1145/3446776)

### Objective
The primary objective is to investigate how label noise affects the performance and generalization capabilities of a neural network (in this case CNN is used).


Understanding the impact of different levels of label noise: 
Examining the model's accuracy and robustness as the percentage of corrupted labels increases.

Analyzing the effect of data augmentation: Evaluating how data augmentation techniques can mitigate the negative effects of label noise.

Comparing the performance of different architectures: Exploring the effectiveness of various CNN architectures in handling noisy labels.


### Datasets 

#### MNIST Dataset Summary
The MNIST dataset consists of 70,000 28x28 grayscale images of handwritten digits (0-9), with 60,000 images for training and 10,000 for testing. It is widely used for training and testing image processing systems and machine learning models, particularly in image classification tasks. MNIST is popular due to its simplicity, balanced classes, and the availability of extensive literature and pre-trained models, making it an excellent benchmark for evaluating algorithm performance.

#### CIFAR-10 Dataset Summary
The CIFAR-10 dataset contains 60,000 32x32 color images divided into 10 classes (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck), with 50,000 images for training and 10,000 for testing. It is extensively used in machine learning for image classification and object recognition tasks, providing a more challenging benchmark than MNIST due to its higher variability and inclusion of color images. CIFAR-10 is crucial for testing the capabilities of advanced models, particularly convolutional neural networks (CNNs).

Model: A CNN architecture with multiple convolutional layers, max pooling, batch normalization, and dropout.
Training: The model is trained using the Adam optimizer with a learning rate scheduler and early stopping.
Data Augmentation: The code includes data augmentation techniques like rotation, shifting, flipping, and zooming to improve generalization.

Evaluation: Model performance is evaluated using accuracy on the clean test set.


### The experiment: Randomized labels,  Corrupted Labels and Shuffled Pixels
1. Introduces random label noise to a portion of the training data.
2. Randomly shuffles pixels in a portion of the training images.
3. Corrupt the labels of the images 

Post all these changes, the CNN model is trained on the training data and checked for accuracy against the test data.

### Conclusion
#### Advantages:
 • The study provides insight into the robustness of neural networks against
 data corruption, suggesting practical applications in scenarios with noisy
 or incomplete data.

• It highlights the importance of considering training algorithms as potential
 regularizes, which could lead to more efficient training strategies that rely
 less on explicit regularization techniques.
#### Disadvantages:
 • The lack of theoretical grounding for the observed phenomena means that
 the findings are mainly empirical, which may limit their applicability with
out further validation.

 • The experiments do not conclusively determine whether the ability to fit
 noise translates to true learning capability or merely memorization, an
 area that requires deeper investigation.


In conclusion, the study **effectively challenges traditional notions of neural net
work generalization** and opens up new avenues for research into the implicit
capabilities of training algorithms. However, it also underscores the need for a
better theoretical understanding of why these models perform as they do, which
remains a significant challenge in the field of machine learning.

![alt text](https://github.com/goeludit/Data-Science/blob/main/Generelization%20of%20Neural%20Networks/Images/Graphs_1.png)
![alt text](https://github.com/goeludit/Data-Science/blob/main/Generelization%20of%20Neural%20Networks/Images/Graphs_2.png)
