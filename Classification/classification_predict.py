from sklearn.metrics import confusion_matrix
import classification_preprocessing as p
import classification_train as train
import Resnet50 as m
from keras.optimizers import Adam
import keras.metrics as t
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2
from keras.models import model_from_json

def cm_plt(ax, cm, classes, cmap, title, normalize):

    # create a heatmap of the confusion matrix using the color map
    im = ax.imshow(cm, interpolation="nearest", cmap=cmap)
    
    #add colorbar to the plot
    ax.figure.colorbar(im, ax=ax)

    # We want to show all ticks...
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        # ... and label them with the respective list entries
        xticklabels=classes,
        yticklabels=classes,
        title=title,
        ylabel="True label",
        xlabel="Predicted label",
    )

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = ".2f" if normalize else "d"
    thresh = cm.max() / 2.0

    # loop on rows and columns of the confusion matrix
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], fmt),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    return ax


def plt_confusion_mat(cm, classes, fig_size, cmap=plt.cm.Blues):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=fig_size)
    ax1 = cm_plt(
        ax1,
        cm,
        classes,
        cmap,
        title="Confusion matrix, without normalization",
        normalize=False,
    )

    cmn = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    ax2 = cm_plt(
        ax2,
        cmn,
        classes,
        cmap,
        title="Normalized confusion matrix",
        normalize=True,
    )
    plt.show()


def predict_plot(X_test, Y_test, model):
    y_pred = model.predict(X_test)
    X_test_reshaped = X_test.reshape((X_test.shape[0], 256, 256, 3))
    fig, axis = plt.subplots(4, 4, figsize = (12, 14))

    # get theindex of the maximum value in each row
    idx_cat = np.argmax(y_pred, axis=1)[0]

    for i, ax in enumerate(axis.flat):
        true = 'Polyps' if Y_test[i].argmax() == 1 else 'Normal'
        pred = 'Polyps' if y_pred[i].argmax() == 1 else 'Normal'
        ax.imshow(X_test_reshaped[i][:, :, ::-1]) # reverse order of the color channels, imshow expect BGR
        ax.set(title=f'Real type is {true} \n Predicted type is {pred}')
    plt.show()

    
json_file = open("D:\\College\\Level 4\\First Term\\Graduation Project\\Models\\Classification\\model.json", 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
model.load_weights("D:\\College\\Level 4\\First Term\\Graduation Project\\Models\\Classification\\model_weights.h5")
print("Loaded model from disk")

model.compile(optimizer=train.adam, loss="binary_crossentropy", metrics=["accuracy", t.Precision(), t.Recall(), train.mcc])

score = model.evaluate(p.x_test, p.y_test)
print("Test loss:", round(score[0], 3))
print("Test accuracy:", round(score[1], 3))
print("Test precision:", round(score[2], 3))
print("Test recall:", round(score[3], 3))
print("mcc:", round(score[4], 3))


# Making prediction
y_pred = np.argmax(model.predict(p.x_test), axis=1)
y_true = np.argmax(p.y_test, axis=1)

# get confusion matrix
confuision_mat = confusion_matrix(y_true, y_pred)
# plot confusion_mat
plt_confusion_mat(confuision_mat, classes=p.categories, fig_size=(20, 7))

history = np.load("D:\\College\\Level 4\\First Term\\Graduation Project\\Models\\Classification\\model_history.npy", allow_pickle=True).item()
plt.plot(history['loss'], label='training loss')
plt.plot(history['accuracy'], label='training accuracy')
plt.legend()
plt.show()

predict_plot(p.x_test, p.y_test, model)
