import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pandas as pd
import requests
import numpy as np
from PIL import Image
from io import BytesIO
from sklearn.model_selection import train_test_split
import os

# Load and prepare data
df = pd.read_csv('poster_genres_binarized.csv')

IMAGE_PATH_COL = 'Id'
LOCAL_IMAGES_FOLDER = 'Images'
GENRES = df.columns[3:].tolist()
NUM_CLASSES = len(GENRES)

# Split data
train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42)
valid_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

# Custom data generator
class URLImageGenerator(tf.keras.utils.Sequence):
    def __init__(self, dataframe, batch_size=32, img_size=(182,268), shuffle=True):
        self.dataframe = dataframe
        self.batch_size = batch_size
        self.img_size = img_size
        self.shuffle = shuffle
        self.indexes = np.arange(len(self.dataframe))
        self.on_epoch_end()
        
    def __len__(self):
        return int(np.ceil(len(self.dataframe) / self.batch_size))
    
    def __getitem__(self, index):
        batch_indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        batch_df = self.dataframe.iloc[batch_indexes]
        
        images = []
        labels = []
        
        for _, row in batch_df.iterrows():
            local_img_path = os.path.join(LOCAL_IMAGES_FOLDER, str(row[IMAGE_PATH_COL])+ '.jpg')
            if not os.path.exists(local_img_path):
                # Skip if file not found (or handle differently if you prefer)
                continue

            try:
                # Load image from local path
                img = Image.open(local_img_path).convert('RGB')
                img = img.resize(self.img_size)
                
                img_array = tf.keras.preprocessing.image.img_to_array(img)
                img_array /= 255.0  # Normalize to [0,1]
                
                images.append(img_array)
                
                # Convert the row's genre columns to float32
                labels.append(row[GENRES].values.astype(np.float32))
            except:
                # If any error occurs reading the image, skip it
                continue
        
        return np.array(images), np.array(labels)
    
    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)

# Create data generators
BATCH_SIZE = 32
IMG_SIZE = (182, 268)

train_generator = URLImageGenerator(train_df, batch_size=BATCH_SIZE, img_size=IMG_SIZE)
valid_generator = URLImageGenerator(valid_df, batch_size=BATCH_SIZE, img_size=IMG_SIZE, shuffle=False)
test_generator = URLImageGenerator(test_df, batch_size=BATCH_SIZE, img_size=IMG_SIZE, shuffle=False)

# Build Sequential model
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),
    layers.MaxPooling2D(2,2),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),
    layers.Conv2D(128, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(NUM_CLASSES, activation='sigmoid')  # Multi-label output
])

# Compile model
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=[
        tf.keras.metrics.BinaryAccuracy(),
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall')
    ]
)

# Add callbacks
callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=3, monitor='val_loss'),
    tf.keras.callbacks.ModelCheckpoint('best_model.h5', save_best_only=True)
]

# Train model
history = model.fit(
    train_generator,
    validation_data=valid_generator,
    epochs=20,
    callbacks=callbacks
)

# Evaluate on test set
test_loss, test_acc, test_precision, test_recall = model.evaluate(test_generator)
print(f"Test Accuracy: {test_acc:.2f}, Precision: {test_precision:.2f}, Recall: {test_recall:.2f}")