# Caltech-101 Image Classification with CNN

## Introduction
The Caltech-101 dataset is a well-known benchmark for image classification tasks. It consists of 101 object categories plus a background category, making it an ideal candidate for testing deep learning models. This project addresses the problem of accurately classifying images from this dataset using a Convolutional Neural Network (CNN).


## Project Overview
This project utilizes TensorFlow's deep learning capabilities to preprocess, augment, and classify images from the Caltech-101 dataset. A MobileNetV2-based transfer learning model is fine-tuned to achieve high accuracy while maintaining computational efficiency.


## Data Description
- Dataset: Caltech-101.
- Categories: 101 distinct object categories and one background category.
- Dataset Size: Approximately 9,000 images.
- Image Resolution: Varies, resized to 128x128 pixels for this project.
- Train/Test Split: Automatically handled by TensorFlow Datasets.


## Features
- Input images resized to a uniform size of 128x128 pixels.
- Normalized pixel values for faster convergence.
- Data augmentation with random flips, rotations, and zooms to improve generalization.
- Exploratory Data Analysis (EDA)


## Architecture
The model is based on the MobileNetV2 architecture with transfer learning:
- Base Model: MobileNetV2 pre-trained on ImageNet.
- Global Average Pooling: Reduces feature maps into a single vector.
- Dropout Layer: Prevents overfitting by randomly disabling neurons.
- Output Layer: Dense layer with softmax activation for 102 classes.


## Repository Structure
```
├── README.md               # Documentation for the project
├── Dockerfile              # Instructions to containerize the application
├── notebook.ipynb          # Jupyter notebook for dataset loading, feature engineering, EDA, and preparation
├── train_model.ipynb       # Script for training the model (jupyter-notebook)
├── train_model.py          # Script for training the model
├── predict_model.ipynb     # Script for making predictions (jupyter-notebook)
├── predict_model.py        # Script for making predictions
├── data/                   # Directory for storing datasets (optional, not included in version control)
├── model/                  # Directory for saving trained models
└── output/                 # Directory for logs, visualizations, and other output files
```

## Installation and Deployment

### Local Server
- Clone this repository
  ```
  git clone https://github.com/ketutgarjitao/caltech-101-classification-model.git  
  cd caltech-101-classification-model 
  ```  
- Install library dependences
  ```
  pip install flask tensorflow pillow tensorflow_datasets matplotlib
  ```
- Train model
  ```
  python train_model.py  
  ```
- Predict model
  ```
  python predict_model.py
  ```
- Test model prediction
  ```
  ./curl.sh
  ```
- Check prediction result on output/ dorectory
  
- Stop predict model session
  ```
  Ctrl-C
  ```
  

### Docker
- Build the Docker image
  ```
  docker build -t caltech101-service .
  ```
- Run the container
  ```
  docker run -d -p 8000:5800 --name caltech101-app caltech101-service 
  ```
  
  _Note: Libraries dependecies are included in Dockerfile script_:

  _RUN pip install flask tensorflow pillow tensorflow_datasets matplotlib_
  

### AWS Cloud

- Add rule on "Edit inbound rules" in EC2 Security Groups
  - Type: Custom TCP
  - Port range: 5000
  - Source: Anywhere-IPv4

- Create directories

  Connect to EC2 instance
  ```
  cd
  mkdir code output model
  ```
  
- Copy files from local to AWS EC2
  ```
  scp -i "~/.ssh/ml-zoomcamp-gar-key.pem" Dockerfile ubuntu@ec2-108-137-82-199.ap-southeast-3.compute.amazonaws.com:/home/ubuntu/caltech-101
  scp -i "~/.ssh/ml-zoomcamp-gar-key.pem" code/train_model.py ubuntu@ec2-108-137-82-199.ap-southeast-3.compute.amazonaws.com:/home/ubuntu/caltech-101/code  
  ```
  
- Deploy on AWS Linux
  - Train model 
    ```
    # Train model
    python train_model.py
    ```
  - Test API access from browser
    http://Public IPv4 address:5000
  - Stop Flask Service
    ```
    Ctrl-C
    ```
  
- Deploy on AWS running Docker
  - Build image and run container
    ```
    cd /home/ubuntu
    docker build -t caltech101-service .
    docker run -d -p 80:5000 --name caltech101-app caltech101-service
    ```
    
    _Note: Libraries dependecies are included in Dockerfile script_:
 
    RUN pip install flask tensorflow pillow
  
  - Train model
    ```
    docker exec -it caltech101-app bash
    cd code
    python train_model.py
    ```
  - Test API access from browser
    http://Public IPv4 address:5000
 
  - Stop Flask Service
    ```
    Ctrl-C
    ```
  - exit from container

## Outputs
- Trained Model: Saved as caltech101_cnn_model.keras.
- Metrics: Test accuracy: ~XX%.
- Visualizations: Training and validation accuracy/loss plots.

## Model

The model uses transfer learning for computational efficiency. The base MobileNetV2 layers are frozen, and only the top layers are trained on the Caltech-101 dataset.

## Prediction
- Run the prediction script:
```
python predict.py --image path_to_image.jpg
```

This will output the predicted category.

## Visualization

Example training history:

## Challenges and Considerations
- Dataset Imbalance: Some categories have fewer images. This was mitigated using data augmentation.
- Overfitting: Handled with dropout and early stopping.
- Computational Resources: Transfer learning significantly reduced training time and resource requirements.

## Suggestions for Improvement
- Experiment with fine-tuning the base model layers for potentially higher accuracy.
- Incorporate additional data augmentation techniques.
- Deploy the model as an API for real-time predictions.

## Acknowledgments
- TensorFlow team for the datasets and tools.
- Caltech for providing the dataset
