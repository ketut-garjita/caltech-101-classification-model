from flask import Flask, request, jsonify, send_file
import tensorflow as tf
import numpy as np
import tensorflow_datasets as tfds
import os

# Set Matplotlib cache directory to 
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"

# Import Matplotlib after configuration
import matplotlib.pyplot as plt

# Load model and dataset
model_path = 'model/caltech101_cnn_model.keras'
model = tf.keras.models.load_model(model_path)

# Load dataset
#tfds_dir = "data/"
tfds_dir = "data/"

dataset, info = tfds.load("caltech101", as_supervised=True, with_info=True, data_dir=tfds_dir)
test_data = dataset["test"].map(lambda img, lbl: (tf.image.resize(img, (128, 128)) / 255.0, lbl)).batch(32)

# Flask App
app = Flask(__name__)

# Prediction Function
def visualize_predictions(dataset, model, num_images=16, output_file="output/Visualize_Prediction.png"):
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

@app.route('/visualize_predictions', methods=['GET'])
def handle_visualize_predictions():
    try:
        num_images = int(request.args.get('num_images', 16))
        output_file = "output/Visualize_Prediction.png"
        
        visualize_predictions(test_data, model, num_images, output_file)
        
        return send_file(output_file, mimetype='image/png')
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
