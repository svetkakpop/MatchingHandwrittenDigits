# Report on Neural Network Implementation

# Introduction

The field of machine learning has revolutionized the way we approach complex problems, particularly in the realm of image recognition. A brave description can be find in this article about [image recognition and how it works]([url](https://dev.to/frosnerd/handwritten-digit-recognition-using-convolutional-neural-networks-11g0)). One of the most prominent applications of this technology is in the recognition of handwritten digits, which has significant implications for various industries, including banking, postal services, and automated data entry systems. For instance, automated check processing in banks relies on accurate digit recognition to streamline transactions and reduce human error. Similarly, postal services utilize digit recognition to sort mail efficiently, ensuring timely delivery.

The **MNIST dataset** -- a well-known benchmark in the machine learning community, serves as a foundational resource for training models to recognize handwritten digits. This dataset comprises thousands of labeled images, providing a robust platform for developing and testing neural network architectures. The primary goal of this report is to implement a simple neural network that can effectively learn from the MNIST dataset, enabling it to accurately classify handwritten digits. By training the model on a substantial amount of data and evaluating its performance on unseen test data, we aim to demonstrate the capabilities and effectiveness of neural networks in solving real-world classification problems.

# Description of the Model

The neural network implemented in this report is designed for multi-class classification, specifically for recognizing handwritten digits from the MNIST dataset. Multi-class classification is a type of classification task where the goal is to categorize input data into one of three or more classes. In this case, the model aims to classify images of handwritten digits into one of ten possible classes (0 through 9). This model consists of three layers: an input layer, a hidden layer, and an output layer. Each layer plays a distinct role in the classification process, contributing to the overall functionality of the network.

1. **Input Layer**: The input layer contains 784 neurons, each representing a pixel in a 28x28 grayscale image of a handwritten digit. The pixel values are normalized to a range between 0.01 and 0.99 to facilitate the training process. This normalization helps the model converge more quickly and effectively during training. The input layer serves as the entry point for the data, where the raw pixel values are fed into the network for processing.

2. **Hidden Layer**: The hidden layer consists of 200 neurons. Each neuron in this layer processes the inputs it receives from the input layer and contributes to the learning of complex patterns and features from the data. The hidden layer acts as an intermediary between the input and output layers, enabling the network to build a hierarchical representation of the data. This layer is crucial for feature extraction, allowing the model to identify relevant characteristics of the handwritten digits.

3. **Output Layer**: The output layer contains 10 neurons, corresponding to the 10 possible digit classes (0 through 9). Each output neuron produces a value that indicates the likelihood of the input image belonging to a specific digit class. The neuron with the highest value is selected as the predicted class for the input image. The output layer is responsible for generating the final classification result, indicating which digit the model believes is represented in the input image.

During the training process, the model adjusts the weights of the connections between the layers based on the errors in its predictions. This iterative process allows the neural network to learn from the training data and improve its classification accuracy over time. The classification of neurons into input, hidden, and output layers is essential for structuring the network in a way that facilitates learning and decision-making, making it a powerful tool for image classification tasks.

## Model Representation

The neural network implemented in the provided task is a multi-layer perceptron (MLP) designed for classifying handwritten digits (0-9). It consists of an input layer, one hidden layer, and an output layer. The model receives input data as normalized pixel values from images, which are processed through the hidden layer using weighted sums and the sigmoid activation function. This function transforms the weighted sums into probabilities, allowing the network to learn patterns in the data through backpropagation, which adjusts the weights based on the error between predicted and actual outputs. Finally, the output layer produces a probability distribution over the digit classes, and the class with the highest probability is selected as the predicted label. In essence, this neural network effectively learns to classify images by capturing the relationships between input features and output classes.

## Mathematical Functions in Neural Networks

In the context of neural networks, several mathematical functions and operations are crucial for the model's performance. Below, we detail these functions and their roles in the training and operation of neural networks.

### 1. Sigmoid Function

The **sigmoid function** is a widely used activation function in neural networks, particularly in the hidden and output layers. It transforms the input \(x\) into a value between 0 and 1, making it suitable for interpreting outputs as probabilities. The mathematical representation of the sigmoid function is given by:

\[
\sigma(x) = \frac{1}{1 + e^{-x}}
\]

In Python, this function is implemented using `scipy.special.expit(x)`, which efficiently computes the sigmoid for the input \(x\). The output of the sigmoid function is particularly useful in binary classification tasks, where the output can be interpreted as the probability of a certain class.

### 2. Linear Combinations

In neural networks, the input data is combined with weights through **matrix multiplication** to form linear combinations. This operation is essential for propagating inputs through the network. The linear combination is computed as follows:

\[
\text{hidden\_inputs} = \text{np.dot}(\text{wih}, \text{inputs})
\]

Here, `wih` represents the weights between the input and hidden layers, and `inputs` are the input data. The result, `hidden_inputs`, serves as the input to the activation function (e.g., sigmoid) in the hidden layer.

### 3. Weight Updates

To improve the model's accuracy, the weights in the neural network must be updated based on the errors calculated during training. The weight update rule is derived from the **gradient descent** algorithm, which minimizes the loss function. The update for the weights between the hidden and output layers is expressed as:

\[
\text{self.who} += \text{self.lr} \cdot \text{np.dot}((\text{output\_errors} \cdot \text{final\_outputs} \cdot (1.0 - \text{final\_outputs})), \text{np.transpose}(\text{hidden\_outputs}))
\]

In this formula:
- `self.who` represents the weights between the hidden and output layers.
- `self.lr` is the learning rate, which controls how much to adjust the weights.
- `output_errors` is the difference between the predicted and actual outputs.
- `final_outputs` are the outputs from the output layer after applying the activation function.

This update rule allows the model to learn from its mistakes by adjusting the weights based on how much the output deviates from the target values.

### 4. Normalization

**Normalization** of input data is a critical preprocessing step that enhances the learning process. It scales the input values to a specific range, which helps prevent issues such as vanishing gradients. The normalization formula used is:

\[
\text{inputs} = \left(\frac{\text{np.asarray(all\_values[1:], dtype=np.float32)}}{255.0} \cdot 0.99\right) + 0.01
\]

In this equation:
- `all_values[1:]` represents the input data (excluding the label).
- The values are divided by 255.0 to scale them to the range [0, 1].
- The multiplication by 0.99 and addition of 0.01 shifts the range to [0.01, 1.0].

This normalization process accelerates convergence and improves the overall performance of the neural network.

### 5. Summation

To evaluate the performance of the neural network, the **accuracy** of the model is calculated using the summation of correct predictions. The formula for calculating accuracy is:

\[
\text{accuracy} = \frac{\text{scorecard\_array.sum()}}{\text{scorecard\_array.size}}
\]

In this formula:
- `scorecard_array` is an array that contains binary values (1 for correct predictions and 0 for incorrect ones).
- The sum of the array gives the total number of correct predictions.
- Dividing by `scorecard_array.size` provides the proportion of correct predictions, which is the accuracy of the model.

This metric is essential for assessing how well the neural network performs on classification tasks, allowing for adjustments and improvements in the model as needed.


## Loss Function

The loss function is not explicitly defined, but it is implied through the calculation of errors at the output layer. The error, obtained as the difference between the target values and the actual outputs of the neural network, serves as the basis for weight updates. For example, if the target value for a given sample is 0.99 (which corresponds to the positive class), and the actual output of the model is 0.80, the error will be:

\[
\text{output\_errors} = \text{targets} - \text{final\_outputs} = 0.99 - 0.80 = 0.19
\]

The smaller this error, the better the model predicts the outcomes.

We aim to minimize this error, which reflects the ability of the neural network to correctly classify the input data. If the model predicts outputs close to 0.99 for the positive class and 0.01 for the negative class, the error will be low. For example, if for another sample the model predicts 0.95 for the positive class and 0.05 for the negative class, the error will be:

\[
\text{output\_errors} = 0.99 - 0.95 = 0.04
\]

In this case, the error is significantly smaller, indicating a more accurate prediction.

Conversely, if the model makes incorrect predictions, the error increases, guiding the training process to adjust the weights between layers. For instance, if the model predicts 0.20 for the positive class when the actual value is 0.99, the error will be:

\[
\text{output\_errors} = 0.99 - 0.20 = 0.79
\]


## Problem Description

The **MNIST dataset** is a widely used benchmark in the field of machine learning and computer vision, particularly for handwritten digit recognition.

**Objective:** Classify images of handwritten digits (0-9) into their corresponding numerical categories.

**Number of Objects:** The dataset contains 70,000 samples, consisting of 60,000 training images and 10,000 test images.

**Number of Features:** Each image is represented by 784 features, corresponding to a 28x28 pixel grid. Each pixel value ranges from 0 to 255, representing the grayscale intensity of the pixel.

**Target Labels:** The target variable is encoded in integer format, where each digit (0-9) corresponds to a unique label. For example, the digit '3' is represented as the label 3.

The MNIST dataset serves as a fundamental resource for developing and evaluating image classification algorithms. The goal is to accurately predict the digit represented in each image based on the pixel intensity features. This dataset is particularly valuable for testing various machine learning models and techniques, including neural networks, due to its simplicity and the clear structure of the data.

# Workflow Overview

1. **Initialization**: Create an instance of the `NeuralNetwork` class with specified parameters.
2. **Training**: The model is trained on the training data for 5 epochs. Input data is normalized and converted into target values.
3. **Testing**: After training, the model is tested on the test data, and the results are compared with the correct answers.

# Conclusion

The implemented neural network successfully trains on the MNIST dataset and demonstrates high accuracy on test data. The testing results yield an efficiency that can be expressed as the percentage of correct answers. This model can be improved by increasing the number of epochs, modifying the network architecture, or applying other regularization methods.
