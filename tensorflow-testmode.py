from __future__ import absolute_import

import multiprocessing

import tensorflow as tf
import tensorflow_addons as tfa
import typeguard
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras import losses
import PIL.Image
import os
import numpy as np
import matplotlib.pyplot as plt
import manager_of_path

def mcc_metric(y_true, y_pred):
  predicted = tf.cast(tf.greater(y_pred, 0.5), tf.float32)
  true_pos = tf.math.count_nonzero(predicted * y_true)
  true_neg = tf.math.count_nonzero((predicted - 1) * (y_true - 1))
  false_pos = tf.math.count_nonzero(predicted * (y_true - 1))
  false_neg = tf.math.count_nonzero((predicted - 1) * y_true)
  x = tf.cast((true_pos + false_pos) * (true_pos + false_neg)
      * (true_neg + false_pos) * (true_neg + false_neg), tf.float32)
  return tf.cast((true_pos * true_neg) - (false_pos * false_neg), tf.float32) / tf.sqrt(x)

def tester(lock,mp,classes):
    lock.acquire()
    batch_size = 4 # batch =divisione del dataset
    IMG_HEIGHT = 800
    IMG_WIDTH = 600
    total_test = 7200
    checkpoint_dir=os.path.dirname(mp.get_path_classes(classes)["checkpoint"])
    test_image_generator = ImageDataGenerator(rescale=1. / 255)  # Generator for our test data
    #manage gpu memory usage
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
    config.log_device_placement = True  # to log device placement (on which device the operation ran)
    sess = tf.Session(config=config)
    #tf.set_session(sess)  # set this TensorFlow session as the default session for Keras
    #model = Sequential([
    #    Conv2D(16, 3, padding='same', activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    #    MaxPooling2D(),
    #    Conv2D(32, 3, padding='same', activation='relu'),
    #    MaxPooling2D(),
    #    Conv2D(64, 3, padding='same', activation='relu'),
    #    MaxPooling2D(),
    #    Flatten(),
    #    Dense(512, activation='relu'),
    #    Dense(1)
    #])
   
    #model.summary()
    #lastest = tf.train.latest_checkpoint(checkpoint_dir)
    #print(lastest)
    #model.load_weights(lastest)
    #model.load_weights(checkpoint_dir+"training_1.index")
    
    model1=tf.keras.models.load_model(checkpoint_dir+"/model",compile=False)
    model1.summary()
    #model.summary()
    model1.compile(optimizer='adam',
                  loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
<<<<<<< HEAD
                  metrics=tfa)
=======
                  metrics=[mcc_metric])
>>>>>>> 1aae6703759e9418a6e35897b4be17f04513e897
    test_data_gen = test_image_generator.flow_from_directory(batch_size=batch_size,
                                                              directory=mp.get_path_classes("all")["train"],
                                                              shuffle=True,
                                                              target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                              class_mode='binary')

    loss_ev_a, acc_ev_a = model1.evaluate(test_data_gen,batch_size=4,verbose=1)
    print(classes)
    print("Restored model, accuracy: {:5.2f}%".format(100 * acc_ev_a))
    print("Restored model, loss: {:5.2f}%".format(100 * loss_ev_a))
    prediction = model1.predict(test_data_gen, batch_size=4, verbose=1)
    res=tf.math.confusion_matrix(test_data_gen.y,prediction,num_classes=2)
    print('Confusion_matrix: ', res)
    test_data_gen = test_image_generator.flow_from_directory(batch_size=batch_size,
                                                              directory=mp.get_path_classes(classes)["train"],
                                                              shuffle=True,
                                                              target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                              class_mode='binary')

    loss_ev_a, acc_ev_a = model1.evaluate(test_data_gen,batch_size=4,verbose=1)
    print(classes)
    print("Restored model, accuracy: {:5.2f}%".format(100 * acc_ev_a))
    print("Restored model, loss: {:5.2f}%".format(100 * loss_ev_a))
    prediction = model1.predict(test_data_gen, batch_size=4, verbose=1)
    res = tf.math.confusion_matrix(test_data_gen.y, prediction, num_classes=2)
    print('Confusion_matrix: ', res)
# predict restistuisce numpy array
    #loss_predict_a, acc_predict_a = model.predict(test_data_gen,batch_size=4,verbose=1) 
    #print("Restored model, accuracy: {:5.2f}%".format(100 * acc_ev_a))
    #print("Restored model, loss: {:5.2f}%".format(100 * loss_ev_a))
    lock.release();
 



if __name__ == "__main__":


    path_check="/home/bernabei/carla0.8.4/PythonClient/_out/"
    classes_of_modified= ["blur", "black", "brightness",  "200_death_pixels","nodemos","noise","sharpness","brokenlens","icelens","banding","greyscale","50_death_pixels","condensation","dirty_lens","chromaticaberration","rain","all"]
    multiproc=True
    lock= multiprocessing.Lock()
    if multiproc==True:
        for classes in classes_of_modified[2:3]:
            mp = manager_of_path.ManagerOfPath(path_check, classes_of_modified, True)
            p = multiprocessing.Process(target=tester, args=(lock,mp, classes));p.start();
            p.join()
    else:
        mp = manager_of_path.ManagerOfPath(path_check, classes_of_modified[9:11], True)
        tester(mp,classes_of_modified[5],)
