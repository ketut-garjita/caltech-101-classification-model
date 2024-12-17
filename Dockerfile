# Use AWS Lambda base image for Python 3.11
FROM public.ecr.aws/lambda/python:3.11

# Set working directory
WORKDIR /app

# Create necessary directories
RUN mkdir -p /app/model /tmp/output

# Copy application files
COPY predict_model.py /app/predict_model.py
COPY model/caltech101_cnn_model.keras /app/model/caltech101_cnn_model.keras
COPY curl.sh /app/curl.sh
COPY curl_app.py /app/curl_app.py

# Install dependencies
RUN pip install flask tensorflow tensorflow_datasets pillow matplotlib boto3
RUN yum install -y curl

# Expose the port (only for local testing, Lambda won't need this)
EXPOSE 5000

# Command to run the Flask application
CMD ["predict_model.handler"]
