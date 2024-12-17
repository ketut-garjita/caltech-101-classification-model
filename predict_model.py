from flask import Flask, request, jsonify, send_file
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import os
import boto3
import matplotlib.pyplot as plt

# Environment variable for TensorFlow (disabling GPU usage in Lambda)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"

# Initialize Flask app
app = Flask(__name__)

# Load model from S3 (or local if small enough)
def load_model_from_s3(bucket_name, model_key, local_path):
    s3 = boto3.client('s3')
    s3.download_file(bucket_name, model_key, local_path)
    model = tf.keras.models.load_model(local_path)
    return model

# Load dataset from S3 and save locally to /tmp
def download_data_from_s3(bucket_name, data_key, local_path):
    s3 = boto3.client('s3')
    s3.download_file(bucket_name, data_key, local_path)
    return local_path

def _parse_function(proto):
    # Schema definition
    feature_description = {
        "image": tf.io.FixedLenFeature([], tf.string),
        "label": tf.io.FixedLenFeature([], tf.int64)
    }
    parsed_features = tf.io.parse_single_example(proto, feature_description)
    # Decode image
    image = tf.io.decode_jpeg(parsed_features["image"], channels=3)
    label = parsed_features["label"]
    return image, label
    
# Load model
model_path = '/tmp/caltech101_cnn_model.keras'
model = load_model_from_s3('dtcml-outputs', 'caltech101_cnn_model.keras', model_path)

# Dataset (full) and test_data
tfds_dir = "/tmp/data/"
dataset, info = tfds.load("caltech101", as_supervised=True, with_info=True, data_dir=tfds_dir)
test_data = dataset["test"].map(lambda img, lbl: (tf.image.resize(img, (128, 128)) / 255.0, lbl)).batch(32)

# Prediction Function
def visualize_predictions(dataset, model, num_images=16, output_file="/tmp/Visualize_Prediction.png"):
    class_names = info.features['label'].names
    for images, labels in dataset.take(1):
        preds = model.predict(images)
        pred_labels = np.argmax(preds, axis=1)

        plt.figure(figsize=(8, 8))
        for i in range(min(num_images, images.shape[0])):
            plt.subplot(4, 4, i + 1)
            plt.imshow(images[i].numpy())
            plt.title(f"True: {class_names[labels[i]]}\nPred: {class_names[pred_labels[i]]}")
            plt.axis('off')
        plt.tight_layout()
        plt.savefig(output_file)
        plt.close()
        
# Flask Route: Visualize Predictions
@app.route('/visualize_predictions', methods=['GET'])
def handle_visualize_predictions():
    try:
        num_images = int(request.args.get('num_images', 16))
        if num_images <= 0:
            raise ValueError("num_images must be a positive integer.")
        
        output_file = "/tmp/Visualize_Prediction.png"
        visualize_predictions(test_data, model, num_images, output_file)
        
        return send_file(output_file, mimetype='image/png')
    
    except ValueError as ve:
        return jsonify({"error": f"Invalid input: {str(ve)}"}), 400
    except Exception as e:
        return jsonify({"error": f"Error processing request: {str(e)}"}), 500

# Lambda handler function
def handler(event, context):
    with app.app_context():
        return app.full_dispatch_request()

# Start the Flask app (for local testing)
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
