import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
import glob


dataset_dir = "D:\\College\\Level 4\\First Term\\Graduation Project\\Data\\Classification\\kvasir-dataset"


def get_dataCategories(dataset_dir):
    categories = []
    for folder_name in os.listdir(dataset_dir):
        if folder_name == "normal-cecum" or folder_name == 'polyps':
            #check whether the specified path is an existing directory or not
            if os.path.isdir(os.path.join(dataset_dir, folder_name)):
                #number of iamges in each file
                nbr_files = len(
                    glob.glob(os.path.join(dataset_dir, folder_name) + "/*.jpg")
                )
                #append the folder name with number of images in eace folder
                categories.append(np.array([folder_name, nbr_files]))

    categories.sort(key=lambda a: a[0])
    cat = np.array(categories)

    return list(cat[:, 0]), list(cat[:, 1])

categories, nbr_files = get_dataCategories(dataset_dir)

# Create DataFrame
df = pd.DataFrame({"category": categories, "number of files": nbr_files})
print("number of categories: ", len(categories))
print(df)

def create_dataset(datadir, categories, img_wid, img_high):
    X, y = [], []
    for category in categories:
        path = os.path.join(datadir, category)
        class_num = categories.index(category)
        for img in os.listdir(path):
            try:
                #preprocessing 
                img_array = cv2.imread(os.path.join(path, img))
                ima_resize_rgb = cv2.resize(img_array, (img_wid, img_high))
                ima_resize_rgb=cv2.medianBlur(ima_resize_rgb, 3)
                X.append(ima_resize_rgb)
                y.append(class_num)
            except Exception as e:
                pass
       

    y = np.array(y)
    X = np.array(X)
    return X, y



img_wid, img_high = 256, 256
X, y = create_dataset(dataset_dir, categories, img_wid, img_high)

print(f"X: {X.shape}")
print(f"y: {y.shape}")

# Plot the images
plt.figure(figsize=(12, 5))
st, end = 0, 1000
for i in range(2):
    plt.subplot(1, 2, i + 1)
    idx = 0
    idx2 = 999
    if i == 0:
        plt.imshow(X[idx][:, :, ::-1])
        plt.title(f"{i}. {categories[y[idx]]}")
    else:
        plt.imshow(X[idx2][:, :, ::-1])
        plt.title(f"{i}. {categories[y[idx2]]}")
    plt.axis("off")
plt.show()

# split dataset to train and test set
X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=0.8, random_state=42
)
print(f"X_train: {X_train.shape}")
print(f"y_train: {y_train.shape}")
print(f"X_test: {X_test.shape}")
print(f"y_test: {y_test.shape}")

# defining training and test sets
x_train, x_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.3)
x_test = X_test

# Dimension of the dataset
print(f"x_train:{x_train.shape},  y_train:{y_train.shape}")
print(f"x_val:{x_val.shape},  y_val:{y_val.shape}")
print(f"x_test:{x_test.shape},  y_test:{y_test.shape}")

# One Hot Encoding
y_train = to_categorical(y_train)
y_val = to_categorical(y_val)
y_test = to_categorical(y_test)

# Verifying the dimension after one hot encoding
print(f"x_train:{x_train.shape},  y_train:{y_train.shape}")
print(f"x_val:{x_val.shape},  y_val:{y_val.shape}")
print(f"x_test:{x_test.shape},  y_test:{y_test.shape}")

# Image Data Augmentation
train_generator = ImageDataGenerator(rotation_range=20,width_shift_range=0.2, height_shift_range=0.2, horizontal_flip=True, vertical_flip=True, brightness_range=[0.3, 0.7])

val_generator = ImageDataGenerator(rotation_range=20,width_shift_range=0.2, height_shift_range=0.2, horizontal_flip=True, vertical_flip=True, brightness_range=[0.3, 0.7])

test_generator = ImageDataGenerator(rotation_range=20,width_shift_range=0.2, height_shift_range=0.2, horizontal_flip=True, vertical_flip=True, brightness_range=[0.3, 0.7])

# Fitting the augmentation defined above to the data
train_generator.fit(x_train)
val_generator.fit(x_val)
test_generator.fit(x_test)
print('finished')
