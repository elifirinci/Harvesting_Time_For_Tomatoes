import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2

# Define the paths for train and test directories
main_dir = '/content/drive/MyDrive/ddt'  # Main directory
train_dir = f'{main_dir}/train'          # Training data
test_dir = f'{main_dir}/test'            # Testing data

# Data augmentation for training and preprocessing for validation/test
train_datagen = ImageDataGenerator(
    rescale=1.0/255,
    shear_range=0.2,
    zoom_range=0.2,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    brightness_range=[0.8, 1.2],
    horizontal_flip=True,
    validation_split=0.2  # Reserve 20% of training data for validation
)

test_datagen = ImageDataGenerator(rescale=1.0/255)

# Flow data from directories
batch_size = 16
training_set = train_datagen.flow_from_directory(
    directory=train_dir,
    target_size=(224, 224),  # Adjusted for ResNet input
    batch_size=batch_size,
    class_mode='categorical',
    subset='training',
    shuffle=True)  # Shuffle for better training

validation_set = train_datagen.flow_from_directory(
    directory=train_dir,
    target_size=(224, 224),  # Adjusted for ResNet input
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation',
    shuffle=True)  # Shuffle for better validation

test_set = test_datagen.flow_from_directory(
    directory=test_dir,
    target_size=(224, 224),  # Adjusted for ResNet input
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False)  # Do not shuffle to preserve label order in evaluation

# Print class indices for reference
print(f"Class Indices: {training_set.class_indices}")

# Load pre-trained ResNet50 without the top layers
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze most of the base model layers
for layer in base_model.layers[:100]:  # Freeze the first 100 layers
    layer.trainable = False

# Add custom classification head
x = base_model.output
x = GlobalAveragePooling2D()(x)  # Replaces Flatten for better generalization
x = Dense(128, activation='relu', kernel_regularizer=l2(0.01))(x)
x = Dropout(0.5)(x)  # Dropout with 50% rate
x = Dense(64, activation='relu', kernel_regularizer=l2(0.01))(x)
x = Dropout(0.5)(x)
output_layer = Dense(4, activation='softmax')(x)  # Adjust for the number of classes

# Create the full model
model = Model(inputs=base_model.input, outputs=output_layer)

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Callbacks for training
callbacks = [
    EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6)
]

# Train the model
history = model.fit(
    training_set,
    validation_data=validation_set,
    epochs=20,
    callbacks=callbacks
)

# Predict on the test dataset
test_predictions = model.predict(test_set)
predicted_classes = np.argmax(test_predictions, axis=1)  # Predicted class indices
true_classes = test_set.classes  # Ground truth class indices from test_set

# Map class indices to class names
class_indices = test_set.class_indices
class_names = {v: k for k, v in class_indices.items()}

# Calculate the confusion matrix
cm = confusion_matrix(true_classes, predicted_classes)

# Display the confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(class_names.values()))
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.show()
