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
├── build_docker.txt        # Build docker and run container script
├── notebook.ipynb          # Jupyter notebook for dataset loading, feature engineering, EDA, and preparation
├── train_model.ipynb       # Script for training the model (jupyter-notebook)
├── train_model.py          # Script for training the model
├── predict_model.ipynb     # Script for making predictions (jupyter-notebook)
├── predict_model.py        # Script for making predictions
├── data/                   # Directory for storing datasets (optional, not included in version control)
├── model/                  # Directory for saving trained model
├── output/                 # Directory visualizations, and other output files
└── images/                 # Notebook images screenshot
```

## Notebook

  1. Load Dataset

     [Load Dataset](https://github.com/ketut-garjita/caltech-101-classification-model/blob/main/images/load_dataset.png)
     
     ![image](https://github.com/user-attachments/assets/db7b0ece-d311-4fea-9700-13a09417d148)

  1. Explore Dataset Information

     [Explore Dataset Information](https://github.com/ketut-garjita/caltech-101-classification-model/blob/main/images/Explore%20Dataset%20Information.png)
     
     ![image](https://github.com/user-attachments/assets/493258b7-c511-4c27-a206-a3de6e801e88)

  3. Convert Dataset to DataFrame for Exploration

     [Convert Dataset to DataFrame for Exploration](https://github.com/ketut-garjita/caltech-101-classification-model/blob/main/images/Convert%20Dataset%20to%20DataFrame%20for%20Exploration.png)
     
     ![image](https://github.com/user-attachments/assets/1b580b05-d4ea-4b65-a697-a70b64044853)

  5. Visualize Sample Data

     [Visualize Sample Data](https://github.com/ketut-garjita/caltech-101-classification-model/blob/main/images/Visualize%20Sample%20Data.png)
     
     ![image](https://github.com/user-attachments/assets/aefcaff0-3baf-47f0-af17-82230297a434)

  7. Data Preprocessing and Augmentation with Auto Tune


## Modelling
1. Prepare model (**train_model.py**)
    - Pre-processing
         - Load Dataset
         - Explore Dataset Information
         - Convert Dataset to DataFrame for Exploration
         - Visualize Sample Data
         - Data Preprocessing and Augmentation
    - Create model
    - Compile model
    - Train and tune model
    - Evaluate and save the model
    - Visualize training history

2. Predict model (**predict_model.py**)
   ```
   python predict_model.py
   ```

   ![image](https://github.com/user-attachments/assets/3bab78a1-aa46-422d-98d0-fffd9dd66f52)

   
4. Test model prediction (**curl.sh**)
   ```
   curl "http://localhost:5001/visualize_predictions?num_images=12" --output outputs/prediction.png
   ```
   _We can modify curl.sh script for example by replacing num_image parameter such as from 16 to 12._

## Installation and Deployment

### 1. Local Server
- Open port 5000 (default Flask port)
  ```
  sudo ufw enable
  sudo ufw allow 5000
  sudo ufw status
  ```
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
- Check prediction result on output/ directory
  
- Stop predict model session
  ```
  Ctrl-C
  ```
  

### 2. Docker
- Build the Docker image
  ```
  docker build -t caltech101-cnn-model .
  ```
- Run the container
  ```
  docker run -d -p 5000:5000 --name caltech101-cnn-model-service caltech101-cnn-model
  ```
  
  _Note: Libraries dependecies are included in Dockerfile script_:

  _RUN pip install flask tensorflow pillow tensorflow_datasets matplotlib_
  
- Install curl package
  ```
  docker exec -it caltech101-cnn-model-service bash
  apt update
  apt install curl
  ```
- Test model prediction
  ```
  ./curl.sh
  ```
  Or could be run from terminal command prompt:
  ```
  docker exec -it caltech101-cnn-model-service bash /app/curl.sh
  ```
- Check prediction result on output/ directory
  ```
  ls output/Visualize_Prediction.png
  ```
  Or from terminal command prompt:
  ```
  docker exec -it caltech101-cnn-model-service bash -c "ls /app/output/Visualize_Prediction.png"
  ```
- Stop docker container
  ```
  docker stop caltech101-cnn-model-service
  ```
  
### 3. AWS Cloud

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

### 4. DockerHub 


## Outputs
- Trained Model: Saved as caltech101_cnn_model.keras.
- Metrics: Test accuracy: ~XX%.
- Visualizations
  - Training and validation accuracy/loss plots (output of predict_model.py):
    
    [Training and validation accuracy](https://github.com/ketut-garjita/caltech-101-classification-model/blob/main/output/Training_Validation_Accuracy.png)
    
    ![image](https://github.com/user-attachments/assets/d23674c3-132e-41de-90c9-3c80f9fae73a)
    
    [Training and validation loss](https://github.com/ketut-garjita/caltech-101-classification-model/blob/main/output/Training_Validation_Loss.png)
    
    ![image](https://github.com/user-attachments/assets/2e053933-452b-428d-ae96-84253df49eee)

  - True vs Prediction (output of curl.sh):
    
    [Visualize_Prediction](https://github.com/ketut-garjita/caltech-101-classification-model/blob/main/output/Visualize_Prediction.png)

    ![image](https://github.com/user-attachments/assets/7cc66253-e8e0-4b3f-a432-e3de126ed3e2)   


## Model

The model uses transfer learning for computational efficiency. The base MobileNetV2 layers are frozen, and only the top layers are trained on the Caltech-101 dataset.


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
