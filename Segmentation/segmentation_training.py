from tensorflow.keras import backend as K 
from tensorflow.keras.optimizers import SGD, Adam
import os
os.environ["SM_FRAMEWORK"] = "tf.keras"
import segmentation_models as sm
import tensorflow as tf
import segmentation_preprocessing as p
import resunetplusplus as m


smooth = 1

def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)

def dice_loss(y_true, y_pred):
    return 1.0 - dice_coef(y_true, y_pred)

optimizer = tf.keras.optimizers.Nadam(learning_rate=1e-4)
epochs = 200
batch_size = 16
m.model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy', sm.metrics.iou_score, dice_coef])
callbacks=[
    tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=False, monitor="val_loss")]

# Train model
# results = m.model.fit(
#     p.x_train,
#     p.y_train,
#     epochs=epochs,
#     validation_data=(p.x_val, p.y_val),
#     callbacks=callbacks,
#     batch_size=batch_size,
#     validation_split=0.1,
#     verbose=1
# )
# m.model.load_weights("D:\\College\\Level 4\\First Term\\Graduation Project\\Models\\Segmentation\\Models\\Res-unet++\\final\\res++.h5")
# model_json = m.model.to_json()
# with open("res-unet++_layers.json", "w") as json_file:
#     json_file.write(model_json)

# model.save_weights("resunet++_weights.h5")
# print("Saved model to disk")