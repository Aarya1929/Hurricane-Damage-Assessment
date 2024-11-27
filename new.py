from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import cv2
import tensorflow as tf

# Function to print file sizes in the 'damage' and 'no damage' folders
def print_file_sizes(input_path, subset):
    print(f'Files in {subset}:')
    print('-' * 50)
    
    path = os.path.join(input_path, subset)
    
    if not os.path.exists(path):
        print(f"Path '{path}' does not exist.")
        return
    
    sizes = []
    folders = ['damage', 'no damage']  # Assuming these are the subfolders
    
    for folder in folders:
        folder_path = os.path.join(path, folder)
        if not os.path.exists(folder_path):
            print(f"Folder '{folder}' does not exist in '{subset}'.")
            continue

        print(f'Files in {folder}:')
        files = os.listdir(folder_path)
        
        if not files:
            print(f'No files found in {folder}.')
            continue
        
        for f in files:
            file_path = os.path.join(folder_path, f)
            if not os.path.isdir(file_path):  # Skip directories
                file_size = os.path.getsize(file_path) / (1024 * 1024)  # Convert bytes to MB
                sizes.append(file_size)
                print(f.ljust(30) + str(round(file_size, 2)) + 'MB')

    if sizes:
        print('-' * 50)
        print(f'Total: {round(sum(sizes), 2)}MB ({len(sizes)} files)')
    else:
        print('No files found in any folder.')
    
    print('')

# Specify your dataset path
input_path = r'C:\Users\Sakshi Shewale\OneDrive\Desktop\Aarya\dataset\dataset'  # Update to your correct path

# Check files in 'train', 'validation', and 'test' subsets
print_file_sizes(input_path, 'train_another')
print_file_sizes(input_path, 'validation_another')
print_file_sizes(input_path, 'test_another')

# Load image paths into a pandas dataframe
image_df = pd.DataFrame({'path': list(Path(input_path).rglob('*.jp*g'))})

# Extract additional information from file paths
image_df['damage'] = image_df['path'].map(lambda x: x.parent.stem)
image_df['data_split'] = image_df['path'].map(lambda x: x.parent.parent.stem)
image_df['location'] = image_df['path'].map(lambda x: x.stem)
image_df['lon'] = image_df['location'].map(lambda x: float(x.split('_')[0]))
image_df['lat'] = image_df['location'].map(lambda x: float(x.split('_')[-1]))
image_df['path'] = image_df['path'].map(lambda x: str(x))  # Convert path back to string

# Plot data distribution based on split and damage status
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15, 6))

s = 10
alpha = 0.5

# Get the train-validation-test splits
image_df_train = image_df[image_df['data_split'] == 'train_another'].copy()
image_df_val = image_df[image_df['data_split'] == 'validation_another'].copy()
image_df_test = image_df[image_df['data_split'] == 'test_another'].copy()

# Sort by latitude to ensure reproducible behavior
image_df_train.sort_values('lat', inplace=True)
image_df_val.sort_values('lat', inplace=True)
image_df_test.sort_values('lat', inplace=True)

# Plot the latitude and longitude for each split
ax[0].scatter(image_df_train['lon'], image_df_train['lat'], color='C0', s=s, alpha=alpha, label='train')
ax[0].scatter(image_df_val['lon'], image_df_val['lat'], color='C1', s=s, alpha=alpha, label='validation')
ax[0].set_title('split')
ax[0].legend()
ax[0].set_xlabel('longitude')
ax[0].set_ylabel('latitude')

# Plot the damage vs no damage labels
image_df_dmg = image_df[image_df['damage'] == 'damage'].copy()
image_df_nodmg = image_df[image_df['damage'] == 'no_damage'].copy()

ax[1].scatter(image_df_dmg['lon'], image_df_dmg['lat'], color='C0', s=s, alpha=alpha, label='damage')
ax[1].scatter(image_df_nodmg['lon'], image_df_nodmg['lat'], color='C1', s=s, alpha=alpha, label='no damage')
ax[1].set_title('label')
ax[1].legend()
ax[1].set_xlabel('longitude')
ax[1].set_ylabel('latitude')

plt.show()

# Function to load images using OpenCV
def cv2_imread(path, label):
    img = cv2.imread(path.numpy().decode('utf-8'), cv2.IMREAD_UNCHANGED)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img, label

# Function to clean up images and convert to tensors
def tf_cleanup(img, label):
    img = tf.convert_to_tensor(img)
    img = tf.dtypes.cast(img, tf.uint8)
    img.set_shape((128, 128, 3))
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, [128, 128])
    label = tf.convert_to_tensor(label)
    label.set_shape(())
    return img, label

# Create datasets for train, validation, and test
train_path = image_df_train['path'].copy().values
val_path = image_df_val['path'].copy().values
test_path = image_df_test['path'].copy().values

train_labels = np.zeros(len(image_df_train), dtype=np.int8)
train_labels[image_df_train['damage'].values == 'damage'] = 1

val_labels = np.zeros(len(image_df_val), dtype=np.int8)
val_labels[image_df_val['damage'].values == 'damage'] = 1

test_labels = np.zeros(len(image_df_test), dtype=np.int8)
test_labels[image_df_test['damage'].values == 'damage'] = 1

train_ds = tf.data.Dataset.from_tensor_slices((train_path, train_labels))
val_ds = tf.data.Dataset.from_tensor_slices((val_path, val_labels))
test_ds = tf.data.Dataset.from_tensor_slices((test_path, test_labels))

AUTOTUNE = tf.data.experimental.AUTOTUNE

# Apply OpenCV image loading and TensorFlow cleanup
train_ds = train_ds.map(lambda path, label: tuple(tf.py_function(cv2_imread, [path, label], [tf.uint16, label.dtype])),
                        num_parallel_calls=AUTOTUNE)
val_ds = val_ds.map(lambda path, label: tuple(tf.py_function(cv2_imread, [path, label], [tf.uint16, label.dtype])),
                    num_parallel_calls=AUTOTUNE)
test_ds = test_ds.map(lambda path, label: tuple(tf.py_function(cv2_imread, [path, label], [tf.uint16, label.dtype])),
                      num_parallel_calls=AUTOTUNE)

train_ds = train_ds.map(tf_cleanup, num_parallel_calls=AUTOTUNE)
val_ds = val_ds.map(tf_cleanup, num_parallel_calls=AUTOTUNE)
test_ds = test_ds.map(tf_cleanup, num_parallel_calls=AUTOTUNE)

# Data augmentation functions
def rotate_augmentation(img, label):
    img = tf.image.rot90(img, tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32, seed=1111))
    return img, label

def flip_augmentation(img, label):
    img = tf.image.random_flip_left_right(img, seed=2222)
    img = tf.image.random_flip_up_down(img, seed=3333)
    return img, label

# Apply augmentations and concatenate datasets
augmented_train_ds = train_ds.map(rotate_augmentation, num_parallel_calls=AUTOTUNE)
augmented_train_ds = augmented_train_ds.map(flip_augmentation, num_parallel_calls=AUTOTUNE)
train_ds = train_ds.concatenate(augmented_train_ds)

augmented_val_ds = val_ds.map(rotate_augmentation, num_parallel_calls=AUTOTUNE)
augmented_val_ds = augmented_val_ds.map(flip_augmentation, num_parallel_calls=AUTOTUNE)
val_ds = val_ds.concatenate(augmented_val_ds)

# Shuffle datasets
n_train = len(train_labels) * 2
n_val = len(val_labels) * 2
n_test = len(test_labels)

train_ds = train_ds.shuffle(n_train, seed=2019, reshuffle_each_iteration=True)
val_ds = val_ds.shuffle(n_val, seed=2019, reshuffle_each_iteration=True)
test_ds = test_ds.shuffle(n_test, seed=2019, reshuffle_each_iteration=True)

# Batch the datasets
BATCH_SIZE = 32

train_batches_ds = train_ds.batch(BATCH_SIZE)
val_batches_ds = val_ds.batch(BATCH_SIZE)
test_batches_ds = test_ds.batch(BATCH_SIZE)

# Load pre-trained VGG16 model
IMG_SHAPE = (128, 128, 3)

pretrained_model = tf.keras.applications.VGG16(input_shape=IMG_SHAPE,
                                               include_top=False,
                                               weights='imagenet')

# Freeze the convolutional base
pretrained_model.trainable = False

# Build the model
model = tf.keras.Sequential([
    pretrained_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(train_batches_ds,
                    epochs=10,
                    validation_data=val_batches_ds)

# Evaluate the model on test data
test_loss, test_acc = model.evaluate(test_batches_ds)
print(f'Test accuracy: {test_acc}')
