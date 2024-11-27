import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D
from tensorflow.keras.callbacks import EarlyStopping

# Define the paths for train and test directories
main_dir = '/content/drive/MyDrive/ddt'  # Main directory
train_dir = f'{main_dir}/train'          # Training data
test_dir = f'{main_dir}/test'            # Testing data

# Create ImageDataGenerator for training and testing
train_datagen = ImageDataGenerator(
    rescale=1.0/255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1.0/255)

# Flow data from directories
training_set = train_datagen.flow_from_directory(
    directory=train_dir,
    target_size=(224, 224),  # Adjusted for ResNet input
    batch_size=32,
    class_mode='categorical',
    shuffle=True)           # Shuffle for better training

test_set = test_datagen.flow_from_directory(
    directory=test_dir,
    target_size=(224, 224),  # Adjusted for ResNet input
    batch_size=32,
    class_mode='categorical',
    shuffle=False)          # Do not shuffle to preserve label order in evaluation

# Print class indices for reference
print(f"Class Indices: {training_set.class_indices}")


# Load pre-trained ResNet50 without the top layers
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze the base model layers to retain pre-trained features
base_model.trainable = False

# Add custom classification head
x = base_model.output
x = GlobalAveragePooling2D()(x)  # Replaces Flatten for better generalization
x = Dense(128, activation='relu')(x)
x = Dense(64, activation='relu')(x)
output_layer = Dense(4, activation='softmax')(x)  # Adjust for the number of classes

# Create the full model
model = Model(inputs=base_model.input, outputs=output_layer)

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Early stopping to avoid overfitting
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True)

# Train the model
history = model.fit(
    training_set,
    validation_data=test_set,
    epochs=20,
    callbacks=[early_stopping])

# Predict on a single test image
from tensorflow.keras.preprocessing import image
test_image = image.load_img("/content/drive/MyDrive/dataset_tomatoes/single_pred/u (2).jfif", target_size=(224, 224))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0) / 255.0

result = model.predict(test_image)

# Map predictions to class labels
class_indices = training_set.class_indices
class_names = {v: k for k, v in class_indices.items()}

predicted_class_index = np.argmax(result)
predicted_class = class_names[predicted_class_index]
print(f"Predicted class: {predicted_class}")

# Confusion Matrix
test_labels = test_set.classes
test_predictions = model.predict(test_set)
predicted_classes = np.argmax(test_predictions, axis=1)

cm = confusion_matrix(test_labels, predicted_classes)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(class_names.values()))
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.show()
