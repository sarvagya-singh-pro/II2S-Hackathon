from tensorflow.keras.models import Sequential, save_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint
import os
import shutil
from sklearn.model_selection import train_test_split
from google.colab import drive

drive.mount('/content/drive')

checkpoint_path = "/content/drive/My Drive/checkpoints/cp-{epoch:04d}.h5"
checkpoint_dir = os.path.dirname(checkpoint_path)

data_dir = '/content/HCM'  
print(data_dir)

classes = os.listdir(data_dir)
print(classes)

data = {cls: [] for cls in classes}

for cls in classes:
    cls_path = os.path.join(data_dir, cls)
    if os.path.isdir(cls_path):
        for root, dirs, files in os.walk(cls_path)
            data[cls].extend([os.path.join(root, file) for file in files])

train_data, validation_data, test_data = {}, {}, {}
for cls, images in data.items():
    train_images, test_images = train_test_split(images, test_size=0.3, random_state=42)

    validation_images, test_images = train_test_split(test_images, test_size=0.33, random_state=42)

    train_data[cls] = train_images
    validation_data[cls] = validation_images
    test_data[cls] = test_images

for cls in classes:
    print(f"Class: {cls}")
    print(f"Training data size: {len(train_data[cls])}")
    print(f"Test data size: {len(test_data[cls])}")
    print(f"Validation data size: {len(validation_data[cls])}")
    print("---------------------------")

train_data_dir = '/content/train_data'
test_data_dir = '/content/test_data'
validation_data_dir = '/content/validation_data'
print(train_data_dir)
print(test_data_dir)
print(validation_data_dir)

# Create training and test directories
os.makedirs(train_data_dir, exist_ok=True)
os.makedirs(test_data_dir, exist_ok=True)
os.makedirs(validation_data_dir, exist_ok=True)

for cls, paths in train_data.items():
    cls_dir = os.path.join(train_data_dir, cls)
    os.makedirs(cls_dir, exist_ok=True)
    for path in paths:
        shutil.copy(path, cls_dir)

for cls, paths in test_data.items():
    cls_dir = os.path.join(test_data_dir, cls)
    os.makedirs(cls_dir, exist_ok=True)
    for path in paths:
        shutil.copy(path, cls_dir)

for cls, paths in validation_data.items():
    cls_dir = os.path.join(validation_data_dir, cls)
    os.makedirs(cls_dir, exist_ok=True)
    for path in paths:
        shutil.copy(path, cls_dir)input_shape = (150, 150, 1)  # Updated for grayscale images
batch_size = 64
epochs = 20
learning_rate = 0.001
dropout_rate = 0.5

train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(input_shape[0], input_shape[1]),
    batch_size=batch_size,
    class_mode='binary',
    color_mode='grayscale'  
)

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(input_shape[0], input_shape[1]),
    batch_size=batch_size,
    class_mode='binary',
    color_mode='grayscale'  
)

test_generator = test_datagen.flow_from_directory(
    test_data_dir,
    target_size=(input_shape[0], input_shape[1]),
    batch_size=batch_size,
    class_mode='binary',
    color_mode='grayscale'  
)

model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(dropout_rate))
model.add(Dense(1, activation='sigmoid'))

optimizer = Adam(learning_rate=learning_rate)
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

checkpoint_callback = ModelCheckpoint(
    filepath=checkpoint_path,
    save_weights_only=False, 
    verbose=1,
    save_freq='epoch' 
)

history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // batch_size,
    callbacks=[checkpoint_callback]
)


