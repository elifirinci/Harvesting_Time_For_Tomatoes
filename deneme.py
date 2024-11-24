# -*- coding: utf-8 -*-
"""CNN_deneme.ipynb"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# TensorFlow versiyonu
print(tf.__version__)

# Dataset yolu
# - dataset yolu :  /content/drive/MyDrive/dataset_tomatoes
# - dataset train klasörü : /content/drive/MyDrive/dataset_tomatoes/train
# - dataset test klasörü : /content/drive/MyDrive/dataset_tomatoes/test
# - dataset single pred : /content/drive/MyDrive/dataset_tomatoes/single_pred

# Train dataset'i alıyoruz ve işliyoruz
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

training_set = train_datagen.flow_from_directory(
    '/content/drive/MyDrive/dataset_tomatoes/train',
    target_size=(64, 64),
    batch_size=32,
    class_mode='categorical'
)

# Test datasını işliyoruz
test_datagen = ImageDataGenerator(rescale=1.0 / 255)
test_set = test_datagen.flow_from_directory(
    '/content/drive/MyDrive/dataset_tomatoes/test',
    target_size=(64, 64),
    batch_size=32,
    class_mode='categorical'
)

# Eğitim ve test verisi sayısı
print(f"Training set sample count: {training_set.n}")
print(f"Test set sample count: {test_set.n}")

# CNN oluşturuyoruz
cnn = tf.keras.models.Sequential()
# İlk convolutional katman ve pooling
cnn.add(tf.keras.layers.Input(shape=(64, 64, 3)))
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation="relu"))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

# İkinci convolutional katman ve pooling
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation="relu"))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

# Veriyi düzleştirme (Flatten)
cnn.add(tf.keras.layers.Flatten())

# Fully connected katmanlar
cnn.add(tf.keras.layers.Dense(units=128, activation="relu"))
cnn.add(tf.keras.layers.Dense(units=64, activation="relu"))
cnn.add(tf.keras.layers.Dense(units=4, activation="softmax"))

# Optimizasyon ve kayıp fonksiyonu
cnn.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# Early stopping
from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True
)

# Modeli eğitiyoruz
cnn.fit(
    x=training_set,
    validation_data=test_set,
    epochs=20,
    callbacks=[early_stopping]
)

# Test için tek bir resmi tahmin etme
test_image = image.load_img("/content/drive/MyDrive/dataset_tomatoes/single_pred/u (2).jfif", target_size=(64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)
test_image = test_image / 255.0

result = cnn.predict(test_image)

# Sınıf indekslerini ve isimlerini elde etme
class_indices = training_set.class_indices
class_names = {v: k for k, v in class_indices.items()}  # İsimleri ters çevirme

# Tahmin edilen sınıf
predicted_class_index = np.argmax(result)
predicted_class = class_names[predicted_class_index]
print(f"Predicted class: {predicted_class}")

# Karışıklık matrisi hesaplama ve görselleştirme
test_labels = test_set.classes  # Test setindeki gerçek sınıflar
test_predictions = cnn.predict(test_set)
predicted_classes = np.argmax(test_predictions, axis=1)

# Karışıklık matrisi
cm = confusion_matrix(test_labels, predicted_classes)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(class_names.values()))
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.show()
