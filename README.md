# CIFAR-10 Image Classification with Convolutional Neural Networks

This project demonstrates how to build and train a Convolutional Neural Network (CNN) using Keras and TensorFlow to classify images from the CIFAR-10 dataset. The CIFAR-10 dataset consists of 60,000 32x32 color images in 10 different classes.

## Table of Contents

- [Installation](#installation)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Evaluation](#evaluation)
- [Visualization](#visualization)
- [Results](#results)
- [License](#license)

## Installation

To run this project, you need to install the following dependencies:

```bash
pip install tensorflow keras matplotlib
```

## Dataset

The CIFAR-10 dataset is automatically downloaded using Keras datasets API. It contains 60,000 32x32 color images in 10 classes, with 6,000 images per class. There are 50,000 training images and 10,000 test images.

## Model Architecture

The model is a Convolutional Neural Network (CNN) built using the Keras Sequential API. The architecture consists of the following layers:

1. **Conv2D Layer:** 32 filters, 3x3 kernel, ReLU activation
2. **MaxPooling2D Layer:** 2x2 pool size
3. **Conv2D Layer:** 64 filters, 3x3 kernel, ReLU activation
4. **MaxPooling2D Layer:** 2x2 pool size
5. **Conv2D Layer:** 64 filters, 3x3 kernel, ReLU activation
6. **Flatten Layer**
7. **Dense Layer:** 64 neurons, ReLU activation
8. **Dense Layer:** 10 neurons (output layer)

## Training

The model is compiled with the Adam optimizer, sparse categorical crossentropy loss, and accuracy metric. It is trained for 10 epochs with a validation split using the test dataset.

## Evaluation

The model's performance is evaluated on the test dataset. The accuracy and loss are computed and displayed.

## Visualization

The first 25 training images are displayed along with their corresponding class labels. The training and validation accuracy are plotted over the epochs to visualize the training progress.

## Results

The trained model achieves a test accuracy which is displayed at the end of the training process.



## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
