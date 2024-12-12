"""
====================================================
About Dataset:
The Caltech101 dataset contains images from 101 object categories (e.g., “helicopter”, “elephant” and “chair” etc.) and 
a background category that contains the images not from the 101 object categories. 
For each object category, there are about 40 to 800 images, while most classes have about 50 images. 
The resolution of the image is roughly about 300×200 pixels.

Steps:
    1. Load Dataset
    2. Explore Dataset Information*
    3. Convert Dataset to DataFrame for Exploration
    4. Visualize Sample Data
    5. Data Preprocessing and Augmentation
    6. Create the Model*
    7. Compile the Model
    8. Add Early Stopping
    9. Train the Model*
    10.Evaluate and Save the Model
    11.Visualize Training History
====================================================
"""
import tensorflow_datasets as tfds
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# **Step 1: Load Dataset**
print("\nStep 1: Load Dataset")
tfds_dir = "data/" 
dataset, info = tfds.load(
    "caltech101", as_supervised=True, with_info=True, data_dir=tfds_dir
)

train_data, test_data = dataset["train"], dataset["test"]

# **Step 2: Explore Dataset Information**
print("\nStep 2: Explore Dataset Information*")
print(info)

label_names = info.features["label"].names
print(f"Label Names: {label_names}")

# **Step 3: Convert Dataset to DataFrame for Exploration**
print("\nStep 3: Convert Dataset to DataFrame for Exploration")
def tfds_to_dataframe(tf_dataset, num_samples=None):
    data = []
    for image, label in tf_dataset.take(num_samples or -1):
        image_array = image.numpy()
        label_value = label.numpy()
        data.append({
            "label": label_value,
            "image_shape": image_array.shape,
            "image_data": image_array
        })
    return pd.DataFrame(data)

train_df = tfds_to_dataframe(train_data, num_samples=10)
train_df["label_name"] = train_df["label"].apply(lambda x: label_names[x])
print(train_df.head())

# **Step 4: Visualize Sample Data**
print("\nStep 4: Visualize Sample Data")
plt.imshow(train_df.loc[5, "image_data"])
plt.title(f"Label: {train_df.loc[5, 'label_name']}")
plt.axis("off")
plt.savefig("output/image_sample.png")      

# **Step 5: Data Preprocessing and Augmentation**
print("\nStep 5: Data Preprocessing and Augmentation")
IMG_SIZE = 128
BATCH_SIZE = 32

def preprocess(image, label):
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    image = image / 255.0
    return image, label

# Data Augmentation
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
])

# Apply preprocessing and augmentation
train_data = train_data.map(preprocess).map(
    lambda x, y: (data_augmentation(x), y)
).shuffle(1000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
test_data = test_data.map(preprocess).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# **Step 6: Create the Model**
print("\nStep 6: Create the Model")
def create_model():
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3), include_top=False, weights="imagenet"
    )
    base_model.trainable = False  # Freeze initial layers for fine-tuning

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.5),
        layers.Dense(info.features["label"].num_classes, activation="softmax")
    ])
    return model

model = create_model()
model.summary()

# **Step 7: Compile the Model**
print("\nStep 7: Compile the Model")
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# **Step 8: Add Early Stopping**
print("\nStep 8: Add Early Stopping")
early_stopping = EarlyStopping(
    monitor="val_loss", patience=5, restore_best_weights=True
)

# **Step 9: Train the Model**
print("\nStep 9: Train the Model")
history = model.fit(
    train_data,
    validation_data=test_data,
    epochs=20,
    callbacks=[early_stopping]
)

# **Step 10: Evaluate and Save the Model**
print("\nStep 10: Evaluate and Save the Model")
test_loss, test_accuracy = model.evaluate(test_data)
print(f"Test Accuracy: {test_accuracy}")

# model.save("caltech101_model.keras")
model.save('model/caltech101_cnn_model.keras')

# **Step 11: Visualize Training History**
print("\nStep 11: Visualize Training History")

# Training and Validation Accuracy
plt.figure(figsize=(6, 4))
plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.legend()
plt.title("Training and Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.savefig("output/Training_Validation_Accuracy.png")

# Training and Validation Loss
plt.figure(figsize=(6, 4))
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.legend()
plt.title("Training and Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.savefig("output/Training_Validation_Loss.png")
