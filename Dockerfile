# Base Image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Create directories
RUN mkdir /app/model /app/data /app/output

# Copy files
COPY predict_model.py /app
COPY data /app/data
COPY model/caltech101_cnn_model.keras /app/model
COPY curl.sh /app

# Install dependencies
RUN pip install flask tensorflow pillow tensorflow_datasets matplotlib

# Expose port
EXPOSE 5000

# Command to run the application
CMD ["python", "predict_model.py"]
