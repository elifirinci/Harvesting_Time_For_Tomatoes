# -*- coding: utf-8 -*-
"""CNN_deneme.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1hi11IPy704ARbPb7QJVPieAPo8TXqsWD
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image

tf.__version__

# dataset yolu :  /content/drive/MyDrive/dataset_tomatoes
# dataset train klasörü : /content/drive/MyDrive/dataset_tomatoes/train
# dataset test klasörü : /content/drive/MyDrive/dataset_tomatoes/test
# dataset single pred : /content/drive/MyDrive/dataset_tomatoes/single_pred

#train datayı alıyoruz ve işliyoruz
train_datagen = ImageDataGenerator(
    rescale = 1./255,
    shear_range = 0.2,
    zoom_range = 0.2,
    horizontal_flip = True)
training_set = train_datagen.flow_from_directory('/content/drive/MyDrive/dataset_tomatoes/train',
                  target_size = (64, 64),
                  batch_size = 32,
                  class_mode = 'categorical')

# shear_range = 0.2 : Görüntüyü yatay veya dikey eksende hafifçe kaydırır.
# zoom_range = 0.2 : Görüntüleri %20 oranında rastgele yakınlaştırır.
#test datasını işliyoruz bu data shear_range falan almıyor sadece eğitirken fotoları çeşitlendiriyoruz
test_datagen = ImageDataGenerator(rescale = 1./255)
test_set = test_datagen.flow_from_directory('/content/drive/MyDrive/dataset_tomatoes/test',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'categorical')

# cnn oluşturuyoruz
cnn = tf.keras.models.Sequential()
# 64x64 hazırladık input shape :
cnn.add(tf.keras.layers.Input(shape=(64, 64, 3)),
    tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation="relu")
    #2d Convolutional katman ekliyoruz
)

#pooling yapıyoruz veri 32x32 düşecek bundan sonra
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2,strides=2))

#bir layer daha ekliyoruz
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation="relu"))
#artık ikinci layer da input_shape vermek zorunda değiliz otomatik algılıyor
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2,strides=2))

#flat hale getiriyoruz veriyi
cnn.add(tf.keras.layers.Flatten())

#düzleştirdik dense layer ekliyoruz
cnn.add(tf.keras.layers.Dense(units=128,activation="relu"))
cnn.add(tf.keras.layers.Dense(units=64,activation="relu"))
cnn.add(tf.keras.layers.Dense(units=4,activation="softmax"))

#optimizasyon,loss
cnn.compile(optimizer="adam",loss="categorical_crossentropy",metrics=["accuracy"])

from tensorflow.keras.callbacks import EarlyStopping
#overfitting tespit etmek için early stopping kullanıyoruz
early_stopping = EarlyStopping(monitor='val_loss',  # İzlenecek metrik
                               patience=3,         # 3 epoch boyunca iyileşme olmazsa durdur
                               restore_best_weights=True)  # En iyi ağırlıkları geri yükle
#fitting
cnn.fit(x = training_set, validation_data=test_set,epochs=20,callbacks=[early_stopping])

### run ETME

# Test resmini yükle
test_image = image.load_img("/content/drive/MyDrive/dataset_tomatoes/single_pred/u (2).jfif", target_size=(64,64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)  # Modelin istediği şekle getirme
test_image = test_image / 255.0

result = cnn.predict(test_image)

# Sınıf indekslerini al
class_indices = training_set.class_indices
# Sınıf isimlerine dönüştürme (sınıf ismi ve indekslerinin tersini alarak)
class_names = {v: k for k, v in class_indices.items()}

# Sonuçları elde et
predicted_class_index = np.argmax(result)  # En yüksek olasılığa sahip sınıf
predicted_class = class_names[predicted_class_index]

print(f"Predicted class: {predicted_class}")

#/content/drive/MyDrive/dataset_tomatoes/single_pred/d (362).png -- ÇÜRÜK
#/content/drive/MyDrive/dataset_tomatoes/single_pred/o (150).jpg -- ESKİ
#/content/drive/MyDrive/dataset_tomatoes/single_pred/r (2492).jpg -- OLGUN
#/content/drive/MyDrive/dataset_tomatoes/single_pred/u (2).jfif -- HAM