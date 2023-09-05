import segmentation_preprocessing as p
import segmentation_training as train
from skimage import measure
from PIL import Image, ImageDraw
import numpy as np
from keras.models import model_from_json
import os
os.environ["SM_FRAMEWORK"] = "tf.keras"
import tensorflow as tf
from keras import backend as K
import segmentation_models as sm
import cv2
import matplotlib.pyplot as plt
from keras.optimizers import Adam
import keras.metrics as t
import random

def contour(image, mask):
    #select the first mask, all rows and all columns for the first channel
    mask = mask[0][:, :, 0]

    # extract contour of white foreground in mask
    contours = measure.find_contours(mask, 0.5) # array of points
    
    # draw outline on image
    img = Image.fromarray(image)
    draw = ImageDraw.Draw(img)
    for contour in contours:
        coords = np.array(contour, dtype=np.int32)

        # transposes the coordinates
        coords[:, [0, 1]] = coords[:, [1, 0]]
        draw.line(tuple(map(tuple, coords.tolist())), fill='yellow', width=3)
    return img

json_file_seg = open("D:\\College\\Level 4\\First Term\\Graduation Project\\Models\\Segmentation\\Models\\Res-unet++\\final\\res-unet++_layers.json", 'r')
loaded_model_json_seg = json_file_seg.read()
json_file_seg.close()
model_seg = model_from_json(loaded_model_json_seg)
print("Loaded model from disk")
model_seg.load_weights("D:\\College\\Level 4\\First Term\\Graduation Project\\Models\\Segmentation\\Models\\Res-unet++\\final\\resunet++_weights.h5")

model_seg.compile(optimizer=train.optimizer, loss='binary_crossentropy', metrics=['accuracy', sm.metrics.iou_score, train.dice_coef])
# score = model_seg.evaluate(p.x_test, p.y_test)
# print("Test loss:", round(score[0], 2))
# print("Test accuracy:", round(score[1], 2))
# print("Test IOU score:", round(score[2], 2))
# print("Test Dice coefficient:", round(score[3], 2))

# Making prediction
y_pred = model_seg.predict(p.x_test)

fig, ax = plt.subplots(nrows=4, ncols=4, figsize=(15, 25))
fig.subplots_adjust(hspace=0.3, wspace=0.1)
for i in range(4):
    image_x = random.randint(0, len(x_test))
    ax[i, 0].imshow(x_test[image_x])
    ax[i, 0].set_title('The image')
    ax[i, 0].axis('off')
    ax[i, 1].imshow(y_test[image_x])
    ax[i, 1].set_title('Original Mask')
    ax[i, 1].axis('off')

    ax[i, 2].imshow(y_pred[image_x])
    ax[i, 2].set_title('Predicted Mask')
    ax[i, 2].axis('off')
    predMask = (y_pred[image_x] > 0.5).astype(np.bool_)
    annotated_image = contour(x_test[image_x], predMask)
    ax[i, 3].imshow(annotated_image)
    ax[i, 3].set_title('Location of cancer')
    ax[i, 3].axis('off')
plt.show()

