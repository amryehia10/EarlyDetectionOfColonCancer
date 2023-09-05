import classification_preprocessing as p
import Resnet50 as m
from keras.optimizers import Adam
import keras.metrics as t
import tensorflow as tf

def mcc(y_true, y_pred):
    # This line rounds the predicted probabilities to either 0 or 1 to make it binary predictions.
    y_pred_pos = tf.keras.backend.round(tf.keras.backend.clip(y_pred, 0, 1)) 
    
    #negation of the predicted binary values.
    y_pred_neg = 1 - y_pred_pos 

    # round ytrue
    y_pos = tf.keras.backend.round(tf.keras.backend.clip(y_true, 0, 1))
    y_neg = 1 - y_pos

    # computes true positives
    tp = tf.keras.backend.sum(y_pos * y_pred_pos)

    # computes true negatives
    tn = tf.keras.backend.sum(y_neg * y_pred_neg)

    # computes false positives
    fp = tf.keras.backend.sum(y_neg * y_pred_pos)

    #computes false negatives
    fn = tf.keras.backend.sum(y_pos * y_pred_neg)

    #Mcc formula
    numerator = (tp * tn - fp * fn)
    denominator = tf.keras.backend.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))

    return numerator / (denominator + tf.keras.backend.epsilon())

""" Initializing the hyperparameters """
batch_size = 120
epochs = 50
learn_rate = 0.0001
adam = Adam(learning_rate=learn_rate)
m.model.compile(optimizer=adam, loss="binary_crossentropy", metrics=["accuracy", t.Precision(), t.Recall(), mcc])

'''Training'''
# history = m.model.fit(
#     p.train_generator.flow(p.x_train, p.y_train, batch_size= batch_size),
#     epochs=epochs,
#     validation_data=p.val_generator.flow(p.x_val, p.y_val, batch_size=batch_size),
#     
# )
# m.model.load_weights("D:\\College\\Level 4\\First Term\\Graduation Project\\Models\\Classification\\model.h5")
# model_json = m.model.to_json()
# with open("model.json", "w") as json_file:
#     json_file.write(model_json)

# m.model.save_weights("model_weights.h5")
# print("Saved model to disk")
