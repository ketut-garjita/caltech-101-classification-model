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
└── images/                 # Notebook images screenshots
```

**Note:**

On AWS Lambda, we can only write to the /tmp directory, so all files: model, outputs and data are stored in the /tmp directory.


## Notebook

  [notebook.ipynb](https://github.com/ketut-garjita/caltech-101-classification-model/blob/a2089e3adf20a98453f5ec089d0f3bae366b7505/notebook.ipynb)
  
  1. Load Dataset
  
   [Load Dataset](https://github.com/ketut-garjita/caltech-101-classification-model/blob/main/images/load_dataset.png)
    
   ![image](https://github.com/user-attachments/assets/db7b0ece-d311-4fea-9700-13a09417d148)
  
  2. Explore Dataset Information
  
   [Explore Dataset Information](https://github.com/ketut-garjita/caltech-101-classification-model/blob/main/images/Explore%20Dataset%20Information.png)
    
   ![image](https://github.com/user-attachments/assets/493258b7-c511-4c27-a206-a3de6e801e88)
  
  3. Convert Dataset to DataFrame for Exploration
  
   [Convert Dataset to DataFrame for Exploration](https://github.com/ketut-garjita/caltech-101-classification-model/blob/main/images/Convert%20Dataset%20to%20DataFrame%20for%20Exploration.png)
    
   ![image](https://github.com/user-attachments/assets/1b580b05-d4ea-4b65-a697-a70b64044853)
  
  4. Visualize Sample Data
  
  [Visualize Sample Data](https://github.com/ketut-garjita/caltech-101-classification-model/blob/main/images/Visualize%20Sample%20Data.png)
    
   ![image](https://github.com/user-attachments/assets/aefcaff0-3baf-47f0-af17-82230297a434)
  
  5. Data Preprocessing and Augmentation with Auto Tune


## Modelling Modules

1. Prepare model (**train_model.py**)

   [train_model.py](https://github.com/ketut-garjita/caltech-101-classification-model/blob/3026834a2fcd82ef0ea8de1a7db6102efb417b5c/train_model.py)
    
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

   [predict_model.py](https://github.com/ketut-garjita/caltech-101-classification-model/blob/f41711f98d8f38aad1def3cd6d2be1b2e7c1a9ef/predict_model.py)
   
3. Test model prediction (**curl.sh**)

   [curl.sh](https://github.com/ketut-garjita/caltech-101-classification-model/blob/4acbd1cd747e9a0e03c8e033aed1096648da2e69/curl.sh)

   ```
   curl "http://localhost:5001/visualize_predictions?num_images=16" --output outputs/prediction.png
   ```
   
   
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
  git clone https://github.com/ketut-garjita/caltech-101-classification-model.git  
  cd caltech-101-classification-model 
  ```  
- Install library dependencies
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
  ![image](https://github.com/user-attachments/assets/3bab78a1-aa46-422d-98d0-fffd9dd66f52)
  
- Test model prediction

  Open new terminal session.
  
  ```
  ./curl.sh
  ```
  
- Check prediction result on output/ directory

  ```
  ls -l output/prediction.png
  ```
  
- Stop predict model session
  ```
  Ctrl-C
  ```  

### 2. Docker

- Build the Docker image
  ```
  docker build -t caltech101-cnn-lambda .
  ```
- Run the container
  ```
  docker run -d -p 5000:5000 --name caltech101-cnn-lambda-service caltech101-cnn-lambda
  ```
  
    _Note: Libraries dependecies are included in Dockerfile script_:
  
    _RUN pip install flask tensorflow pillow tensorflow_datasets matplotlib_
  
 - Test model prediction
     
     - Connect to container
       ```
       docker exec -it caltech101-cnn-lambda-service bash
       ```
     - Install curl package (this etsp only need once)
       ```
       apt update
       apt install curl
       ```
     - Add an execute mode of the curl.sh (this step only need once)
       ```
       chmod +x curl.sh
       ```
     - Run curl.sh
       ```
       ./curl.sh
       ```
     - Check output
       ```
       ls -l output/prediction.png
       ```
     - Exit from container
       ```
       exit
       ```
  - Stop container
     ```
     docker stop caltech101-cnn-lambda-service
     ```

     
### 3. DockerHub 

- Create DockerHub Repository

  [https://hub.docker.com](https://hub.docker.com/)

  Repository Name: ketutgarjita/caltech101-cnn-lambda

  Make public

- Push local image to DockerHub
  ```
  docker tag caltech101-cnn-lambda:latest ketutgarjita/caltech101-cnn-lambda:latest
  docker login
     Username: 
     Password:
  docker push ketutgarjita/caltech101-cnn-lambda:latest
  ```
  
### 4. AWS Cloud

#### Option 1: AWS Lambda
   - Authenticate Docker to AWS ECR, use the AWS CLI to authenticate Docker client
     ```
     aws ecr get-login-password --region ap-southeast-3 | docker login --username AWS --password-stdin 734800375959.dkr.ecr.ap-southeast-3.amazonaws.com 
     ```
   - Create repository
     ```
     aws ecr create-repository --repository-name caltech101-cnn-lambda
     ```
   - Tag image to AWS
     ```
     docker tag caltech101-cnn-lambda 734800375959.dkr.ecr.ap-southeast-3.amazonaws.com/caltech101-cnn-lambda:latest
     ```
   - Push image
     ```
     docker push 734800375959.dkr.ecr.ap-southeast-3.amazonaws.com/caltech101-cnn-lambda:latest
     ```
   - Create Lambda Role

     - Create Trust Policy for AWS Lambda (**trust-policy.json**)
       ```
       {
          "Version": "2012-10-17",
          "Statement": {
            "Effect": "Allow",
            "Principal": {
              "Service": "lambda.amazonaws.com"
            },
            "Action": "sts:AssumeRole"
          }
        }
       ```
     - Create role
       ```
       aws iam create-role \
       --role-name caltech101-cnn-model-role \
       --assume-role-policy-document file://trust-policy.json
       ```

     - Attach policy to role
       ```
       aws iam attach-role-policy \
       --role-name caltech101-cnn-model-role \
       --policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole
       ```
       
     - Verify role and policy
       ```
       aws iam get-role --role-name caltech101-cnn-model-role
       aws iam list-attached-role-policies --role-name caltech101-cnn-model-role

  - Create AWS Lambda Function:
    ```
    aws lambda create-function \
    --function-name caltech-101-predictor \
    --package-type Image \
    --role arn:aws:iam::734800375959:role/caltech101-cnn-model-role \
    --code ImageUri=734800375959.dkr.ecr.ap-southeast-3.amazonaws.com/caltech101-cnn-lambda:latest
    ```
    
   - Create API Gateway

     ![image](https://github.com/user-attachments/assets/6c4bf1ed-549e-4e53-b304-140d42652e76)

  - Test Using Curl - AWS Lambda - API Gateway

    [curl_app.py](https://github.com/ketut-garjita/caltech-101-classification-model/blob/main/curl_app.py)

    ```
    python curl_app.py
    ```
   
    ![image](https://github.com/user-attachments/assets/9eb88a9f-6743-40ab-ac5a-0c3c81541289)

    ![image](https://github.com/user-attachments/assets/087eca28-ba93-4d1a-bd73-e245ad8483fb)


#### Option 2: Install on EC2 Instance Virtual Environment

   - Start AWS EC2 Instance
   - Connect to Instance
   - Create the virtual environment
     
      ```
      sudo apt update 
      sudo apt install python3.12-venv
      source myenv/bin/activate
      ```
   -  Clone this repository     
      ```
      git clone https://github.com/ketut-garjita/caltech-101-classification-model.git  
      cd caltech-101-classification-model 
      ```  
   - Install library dependencies
     
     ```
     pip install flask tensorflow pillow tensorflow_datasets matplotlib
     ```
   - Train model
     
     ```
     python train_model.py  
     ```
     [aws_train_model_log](https://github.com/ketut-garjita/caltech-101-classification-model/blob/main/output/aws_train_model_log.txt)
     
   - Predict model
      
     ```
     python predict_model.py
     ```
     [aws_predict_model_log](https://github.com/ketut-garjita/caltech-101-classification-model/blob/main/output/aws_predict_model_log.txt)
     
   - Test model prediction
   
     Open new terminal session.
     
     ```
     ./curl.sh
     ```
          
   - Check prediction result on output/ directory
   
     ```
     ls -l output/prediction.png
     ```

     Output should be like this: [prediction.png](https://github.com/ketut-garjita/caltech-101-classification-model/blob/main/output/prediction.png)
     
   - Stop predict model session
     ```
     Ctrl-C
     ```

#### Option 3: Using (Pull) DockerHub Image

   - Pull Image
     
     ```
     docker pull ketutgarjita/caltech101-cnn-lambda:latest
     ```
   - Start and Run Container
     
     ```
     docker run -d -p 5000:5000 --name caltech101-cnn-lambda-service ketutgarjita/caltech101-cnn-lambda
     ```
   - Test model prediction
     
     - Connect to container
       ```
       docker exec -it caltech101-cnn-lambda-service bash
       ```
     - Install curl package (this etsp only need once)
       ```
       apt update
       apt install curl
       ```
     - Add an execute mode of the curl.sh (this step only need once)
       ```
       chmod +x curl.sh
       ```
     - Run curl.sh
       ```
       ./curl.sh
       ```
     - Check output
       ```
       ls -l output/prediction.png
       ```
     - Exit from container
       ```
       exit
       ```
  - Stop container
     ```
     docker stop caltech101-cnn-lambda-service
     ```
  

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
    
    [Visualize_Prediction](https://github.com/ketut-garjita/caltech-101-classification-model/blob/main/output/prediction.png)

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
