from __future__ import absolute_import

import multiprocessing

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import PIL.Image
import os
import numpy as np
import matplotlib.pyplot as plt
import manager_of_path


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
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=mp.get_path_classes(classes)["checkpoint"],
                                                     save_weights_only=True,
                                                     verbose=1,
                                                     period=5)

    model=tf.keras.models.load_model(checkpoint_dir + "/model")
    model.summary()
    test_data_gen = test_image_generator.flow_from_directory(batch_size=batch_size,
                                                              directory=mp.get_path_classes(classes)["train"],
                                                              shuffle=True,
                                                              target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                              class_mode='binary')

    loss, acc = model.evaluate_generator(test_data_gen, verbose=2)
    print("Restored model, accuracy: {:5.2f}%".format(100 * acc))
    print("Restored model, loss: {:5.2f}%".format(100 * loss))
    lock.release();
    #print_result(epochs, history)




if __name__ == "__main__":

    path_of_test= "/home/bernabei/carla0.8.4/PythonClient/_out/"
    path_check="/home/bernabei/carla0.8.4/PythonClient/_out/"
    classes_of_modified= ["blur", "black", "brightness",  "200_death_pixels","nodemos","noise","sharpness","brokenlens","icelens","banding","greyscale","50_death_pixels","condensation","dirty_lens","chromaticaberration","rain","all"]
    multiproc=True
    lock= multiprocessing.Lock()
    if multiproc==True:
        for classes in classes_of_modified[:]:
            mp = manager_of_path.ManagerOfPath(path_check, classes_of_modified, True)
            p = multiprocessing.Process(target=tester, args=(lock,mp, classes));p.start();print(classes);
            p.join()
    else:
        mp = manager_of_path.ManagerOfPath(path_check, classes_of_modified[9:11], True)
        tester(mp,classes_of_modified[5],)
