# Base Image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy files
COPY web_service.py /app/
COPY model/caltech101_cnn_model.keras /app/

# Install dependencies
RUN pip install flask tensorflow pillow tensorflow_datasets matplotlib

# Expose port
EXPOSE 5000

# Command to run the application
CMD ["python", "predict_model.py"]
