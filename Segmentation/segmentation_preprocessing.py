import os
os.environ["SM_FRAMEWORK"] = "tf.keras"
import segmentation_models as sm
import glob
import cv2
from tqdm import tqdm
import numpy as np
from matplotlib import pyplot as plt
import random
from PIL import Image
from sklearn.model_selection import train_test_split
import pandas as pd
from skimage.color import rgb2gray as rtg
from skimage.io import imread, imshow
from skimage.transform import resize
from skimage.morphology import label,square
from skimage import morphology
from skimage.filters import median
from PIL import Image, ImageDraw
from skimage import measure
from albumentations import CenterCrop, RandomRotate90, GridDistortion, HorizontalFlip, VerticalFlip

def get_paths(dataset_dir):
    images = []
    masks = []
    for directory_path in glob.glob(os.path.join(dataset_dir, "*/")):
        if directory_path != 'D:\\College\\Level 4\\First Term\\Graduation Project\\Data\\Segmentation datasets\\cvc612\\':
            for directory_path_2 in glob.glob(os.path.join(directory_path, "*\\")):
                if directory_path_2 != 'D:\\College\\Level 4\\First Term\\Graduation Project\\Data\\Segmentation datasets\\Kvasir for segmentation\\kvasir-sessile\\':
                    img_path = os.path.join(directory_path_2, 'images')
                    mask_path = os.path.join(directory_path_2, 'masks')
                    if img_path:
                        for imgs in glob.glob(os.path.join(img_path, "*jpg")):
                            images.append(imgs)
                            
                    if mask_path:
                        for msks in glob.glob(os.path.join(mask_path, "*jpg")):
                            masks.append(msks)
                            
                else:
                    pass
        else:
            for directory_path_2 in glob.glob(os.path.join(directory_path,'PNG')):
                mask_path = os.path.join(directory_path_2, 'Ground Truth')
                img_path = os.path.join(directory_path_2, 'Original')
                if img_path:
                    for imgs in glob.glob(os.path.join(img_path, "*png")):
                        images.append(imgs)
                        
                if mask_path:
                    for msks in glob.glob(os.path.join(mask_path, "*png")):
                        masks.append(msks)
                        
    images = sorted(images)
    masks = sorted(masks)
    return images, masks

def augment_data(images, masks, augment=True):
    H = 256
    W = 256
    save_images = []
    save_masks = []
    for x, y in tqdm(zip(images, masks), total=len(images)):
        """ Augmentation """
        if augment == True:

            aug = RandomRotate90(p=1.0)
            augmented = aug(image=x, mask=y)
            x2 = augmented['image']
            y2 = augmented['mask']

            aug = HorizontalFlip(p=1.0)
            augmented = aug(image=x, mask=y)
            x4 = augmented['image']
            y4 = augmented['mask']

            aug = VerticalFlip(p=1.0)
            augmented = aug(image=x, mask=y)
            x5 = augmented['image']
            y5 = augmented['mask']

            save_images.extend([x, x2, x4, x5])
            save_masks.extend([y, y2, y4, y5])

        else:
            save_images.append(x)
            save_masks.append(y)

    save_images = np.array(save_images)
    save_masks = np.array(save_masks)
    return save_images, save_masks


dataset_dir = "D:\\College\\Level 4\\First Term\\Graduation Project\\Data\\Segmentation datasets"
images, masks = get_paths(dataset_dir)

#Create array for images and masks(labels)
X = np.zeros((len(images), 256, 256, 3), dtype=np.uint8)
Y = np.zeros((len(masks), 256, 256, 1), dtype=np.float32)

#preprocessing on images
#apply on images using cv2
print("Resizing training images")
for n, id_ in tqdm(enumerate(images), total=len(images)):
    img = cv2.imread(id_)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (256, 256))
    img = cv2.medianBlur(img, 3)
    X[n] = img

#preprocessing on masks
#apply on masks using skimage
print("Resizing training masks")
for n, id_ in tqdm(enumerate(masks), total=len(masks)):
    mask = np.zeros((256, 256, 1), dtype=np.float32)
    mask = imread(id_)
    mask = rtg(mask)
    mask = (mask - np.min(mask)) / (np.max(mask) - np.min(mask)) * 255
    #Dilation increases the foreground (white) region of the image
    # Define the structuring element for dilation
    # small binary image that is used to define a neighborhood around each pixel in an input image
    selem = morphology.disk(5)
    mask = morphology.binary_dilation(mask, selem)
    mask = np.expand_dims(resize(mask, (256, 256), mode="constant", preserve_range=True), axis=-1)

    Y[n] = mask

# Plotting random images and their mask
image_x = random.randint(0, len(images))
imshow(X[image_x])
plt.show()
imshow(np.squeeze(Y[image_x]))
plt.show()

X_aug, Y_aug = augment_data(X, Y, augment=True)

# split dataset to train and test set
X_train, X_test, y_train, y_test = train_test_split(
    X_aug, Y_aug, train_size=0.8, random_state=42
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
