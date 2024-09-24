# Dog vs Cat Classification using Transfer Learning | Deep Learning
This project is focused on building a deep learning model to classify images of dogs and cats using a dataset from Kaggle.

## Table of Contents
- [Project Overview](#project-overview)
- [Problem Statement](#problem-statement)
- [Dataset](#dataset)
- [Deep Learning](#deep-learning)
    - [Dataset Preparation](#dataset-preparation)

## Project Overview
This project focuses on building an image classification model using deep learning to distinguish between images of dogs and cats. Leveraging the Dogs vs Cats dataset from Kaggle, the project uses a pre-trained MobileNetV2 model for transfer learning. The images are preprocessed, labeled, and then fed into the model, which is trained to classify the images with high accuracy. The model is also capable of predicting whether a given image is a dog or a cat.

## Problem Statement
The goal of this project is to develop an accurate and efficient machine learning model that can automatically classify images as either dogs or cats. With large volumes of image data becoming increasingly available, manual classification is time-consuming and prone to errors. By leveraging deep learning and transfer learning techniques, this project aims to create a model that can classify these images with high accuracy, providing a scalable solution to the problem of image-based animal classification. The challenge lies in processing and differentiating subtle features in images while ensuring generalizability across diverse data.

## Dataset
The **Dogs vs Cats** dataset, originally provided by Kaggle, contains 25,000 labeled images of dogs and cats in a balanced distribution (12,500 dog images and 12,500 cat images). The images vary in size, quality, and orientation, offering a realistic challenge for image classification models.
- _Training Data:_ The dataset includes 25,000 images for training, with filenames indicating the label (e.g., dog.123.jpg or cat.456.jpg).
- _Test Data:_ A separate test set is available for evaluating model performance.

Each image is labeled as either a dog or a cat, making this a binary classification problem. The dataset is well-suited for applying deep learning models, particularly convolutional neural networks (CNNs), due to its large size and visual complexity. For this project, the images were resized to a standard size (224x224 pixels) for consistent input into the neural network model.

## Deep Learning
### Dataset Preparation
