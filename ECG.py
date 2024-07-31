
from google.colab import drive
drive.mount('/content/drive')

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, EarlyStopping
from sklearn.model_selection import train_test_split
import os
import pandas as pd

file_path = '/content/drive/MyDrive/ECG/Normal'
normal = []

# def extract_data()

for filename in os.listdir(file_path):
    file_path_normal = os.path.join(file_path, filename)
    with open(file_path_normal, 'rb') as file:
      binary_data = file.read()
    for i in range(3):
      normal.append(binary_data)

print(len(normal))

file_path = '/content/drive/MyDrive/HCM'
hcm = []

for filename in os.listdir(file_path):
    file_path_normal = os.path.join(file_path, filename)
    with open(file_path_normal, 'rb') as file:
      binary_data = file.read()
    for i in range(2):
      hcm.append(binary_data)

# len(normal)

import struct
from sklearn.preprocessing import MinMaxScaler

num1 =  struct.unpack('<' + 'i' * (len(normal[0]) // 4), normal[0])
scaler = MinMaxScaler()
scaler.fit(np.array(num1).reshape(-1, 1))

def unpack(nm):
  numbers = struct.unpack('<' + 'i' * (len(nm) // 4), nm)
  return scaler.transform(np.array(numbers).reshape(-1, 1))

for i, num in enumerate(normal):
  normal[i] = unpack(num)

for i, num in enumerate(hcm):
  hcm[i] = unpack(num)

!pip install ecg-plot

print(normal[0])



tmp_normal = normal
tmp_hcm = hcm



normal = tmp_normal;
hcm = tmp_hcm;


for i, signal in enumerate(normal):
  normal[i] = signal[:10000]

for i, signal in enumerate(hcm):
  hcm[i]  = signal[:10000]

normal = np.array(normal)
hcm = np.array(hcm)

for i,signal in enumerate(normal):
  print(f'\r{i}' , end = ' ')
  normal[i] = 100*signal

for i,signal in enumerate(hcm):
  print(f'\r{i}' , end = ' ')
  hcm[i] = 100*signal

print(normal[0])

print(normal.shape)

import ecg_plot
ecg_plot.plot_1(normal[0]/100, sample_rate = 500, title = 'ECG 12')
ecg_plot.show()

!mkdir /content/data
!mkdir /content/data/normal
!mkdir /content/data/hcm

for i,signal in enumerate(normal):
  print(f'\r{i}', end = ' ')
  path = f'data/normal/normal_{i}.npy'
  file = open(path, 'wb')
  file.write(signal)
  file.close()

print('\n', end = '\n')

for i,signal in enumerate(hcm):
  print(f'\r{i}', end = ' ')
  path = f'data/hcm/hcm_{i}.npy'
  file = open(path, 'wb')
  file.write(signal)
  file.close()

# !zip -r data.zip data/ -q

from keras.utils import image_dataset_from_directory
# from keras.utils.data import random_split

"""<h2>Code Starts Here</h2>
<h3>First We will load DataSet It have been made a zip file</h3>
"""

# !unzip -d /content/ data.zip

normal[0].shape

def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]

X, y = unison_shuffled_copies(np.concatenate((normal, hcm)), np.concatenate((np.zeros(len(normal)), np.ones(len(hcm)))))

y[:10]

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers

model = Sequential()
model.add(layers.Conv1D(10, kernel_size=10, activation='relu', input_shape=(10000, 1)))
model.add(layers.AveragePooling1D(pool_size=2))
model.add(layers.Conv1D(20, kernel_size=10, activation='relu', input_shape=(10000, 1)))
model.add(layers.AveragePooling1D(pool_size=2))
# model.add(layers.Dropout(0.2))
model.add(layers.Conv1D(100, kernel_size=10, activation='relu'))
model.add(layers.AveragePooling1D(pool_size=2))
# model.add(layers.Dropout(0.2))
model.add(Flatten())
model.add(Dense(32, activation='relu'))
# model.add(layers.Dropout(0.2))
model.add(Dense(32, activation='relu'))
model.add(layers.Dropout(0.2))
model.add(Dense(2, activation='softmax'))

# model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# model.add(Dense(128, activation='softmax'))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.summary()
model.save('saved_model.keras')

print("Model saved successfully.")

X.shape

dataset_size = X.shape[0]
train_size = int(0.85 * dataset_size)
test_size = int(0.15 * dataset_size)
# test_size = dataset_size - train_size - val_size

import tensorflow as tf
from sklearn.model_selection import train_test_split

# Assume X_train, y_train, X_test, y_test are your data and labels

X_train = X[:train_size]
y_train = y[:train_size]
X_test = X[train_size:]
y_test = y[train_size:]

# Combine data and labels into a tf.data.Dataset
train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))

# Optionally, shuffle the training data
train_dataset = train_dataset.shuffle(buffer_size=len(X_train))
test_dataset = test_dataset.shuffle(buffer_size=len(X_test))

# Determine sizes of train, validation, and test sets
train_size = int(0.9 * len(X_train))
val_size = int(0.1 * len(X_train))
test_size = len(X_train) - train_size - val_size

# Split train dataset into train and validation sets
# train_dataset = train_dataset.take(train_size)
# remaining_dataset = train_dataset.skip(train_size)
# val_dataset = remaining_dataset.take(val_size)
# train_dataset = remaining_dataset.skip(val_size)

# Batch the datasets
batch_size = 64
train_dataset = train_dataset.batch(batch_size)
# val_dataset = val_dataset.batch(batch_size)
test_dataset = test_dataset.batch(batch_size)

# Optionally prefetch for better performance
train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)
# val_dataset = val_dataset.prefetch(tf.data.experimental.AUTOTUNE)
test_dataset = test_dataset.prefetch(tf.data.experimental.AUTOTUNE)

# train_data = X[:train_size]

# train_labels = y[:train_size]
# val_labels = y[train_size:train_size+val_size]
# test_labels = y[train_size+val_size:]

# dataset = tf.data.Dataset.from_tensor_slices((train_data, train_labels))
# val_dataset = tf.data.Dataset.from_tensor_slices((val_data, val_labels))
# test_dataset = tf.data.Dataset.from_tensor_slices((test_data, test_labels))

# batch_size = 64
# dataset = dataset.shuffle(buffer_size=100).batch(batch_size)
# val_dataset = val_dataset.shuffle(buffer_size=100).batch(batch_size)
# test_dataset = test_dataset.shuffle(buffer_size=100).batch(batch_size)

val_dataset

from sklearn.metrics import roc_curve, auc

import matplotlib.pyplot as plt
history = model.fit(train_dataset, epochs=10, validation_data=test_dataset)
test_loss, test_accuracy = model.evaluate(test_data, test_labels)
print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_accuracy}")

# Generate predictions on the test set
y_pred_prob = model.predict(test_data)

# Calculate ROC curve and AUC
fpr, tpr, thresholds = roc_curve(test_labels, y_pred_prob[:, 1])
roc_auc = auc(fpr, tpr)
print(f'AUC: {roc_auc:0.2f}')

# Plot training history
plt.figure(figsize=(12, 4))

# Plot loss
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()

# Plot validation accuracy
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()

plt.show()

# Plot ROC curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:0.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc='lower right')
plt.show()

test_loss, test_accuracy = model.evaluate(test_dataset)

print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_accuracy}")



"""<h2>LIME Implementation</h2>"""

from lime_explanation import segment_ecg_signal, generate_random_perturbations
from visualization import plot_segmented_ecg

instance_ecg = X_test[0]

num_slices = 40
slice_width = segment_ecg_signal(instance_ecg, num_slices)

# plot the segmented ECG signal
plot_segmented_ecg(instance_ecg, slice_width)

num_perturbations = 150
random_perturbations = generate_random_perturbations(num_perturbations, num_slices)

# Example output
print("The shape of random_perturbations array (num_perturbations, num_slices):", random_perturbations.shape)
print("Example Perturbation:", random_perturbations[-1])

from lime_explanation import apply_perturbation_to_ecg, perturb_mean
from visualization import plot_perturbed_ecg

# Choose the perturbation function
perturb_function = perturb_mean

# Apply a random perturbation to the ECG signal
perturbed_ecg_example = apply_perturbation_to_ecg(instance_ecg, random_perturbations[-1], num_slices, perturb_function)

# plot the original and perturbed ECG signals with highlighted slices and deactivated segments
plot_perturbed_ecg(instance_ecg, perturbed_ecg_example, random_perturbations[-1], num_slices, title='ECG Signal with Perturbation')

from lime_explanation import predict_perturbations

## Predict the class probabilities using the trained ECG classifier
perturbation_predictions = predict_perturbations(model, instance_ecg, random_perturbations, num_slices, perturb_mean)

from lime_explanation import calculate_cosine_distances

# Calculate cosine distances between each perturbation and the original ECG signal representation
cosine_distances = calculate_cosine_distances(random_perturbations, num_slices)
print("Shape of Cosine Distances Array:", cosine_distances.shape)

from lime_explanation import calculate_weights_from_distances

#Applying a Kernel Function to Compute Weights
kernel_width = 0.25  # This can be adjusted based on your specific needs
weights = calculate_weights_from_distances(cosine_distances, kernel_width)

# Now we have the weights for each perturbation for further analysis
print("Shape of Weights Array:", weights.shape)

from lime_explanation import fit_explainable_model
top_pred_classes = [0]
# Constructing the Explainable Model for ECG Signals
segment_importance_coefficients = fit_explainable_model(perturbation_predictions, random_perturbations, weights, target_class=top_pred_classes[0])

# The importance coefficients for each segment
print("Segment Importance Coefficients:", segment_importance_coefficients)

from lime_explanation import identify_top_influential_segments

number_of_top_features = 5
top_influential_segments = identify_top_influential_segments(segment_importance_coefficients, number_of_top_features)

# The indices of the top influential segments
print("Top Influential Signal Segments:", top_influential_segments)

from visualization import visualize_lime_explanation

visualize_lime_explanation(instance_ecg, top_influential_segments, num_slices, perturb_function=perturb_mean)
