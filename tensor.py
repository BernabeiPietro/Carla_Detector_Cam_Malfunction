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


def classificator(lock,mp,classes,path_checkpoint):
    lock.acquire()
    batch_size = 4 # batch =divisione del dataset
    epochs = 5  # epochs= numero di volte che un dataset viene ripetuto nella rete
    IMG_HEIGHT = 800
    IMG_WIDTH = 600
    total_train = 240000
    total_val = 60000
    checkpoint_dir=os.path.dirname(mp.get_path_classes(classes)["checkpoint"]+path_checkpoint)
    train_image_generator = ImageDataGenerator(rescale=1. / 255)  # Generator for our training data
    validation_image_generator = ImageDataGenerator(rescale=1. / 255)  # Generator for our validation data
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

    model = Sequential([
        Conv2D(16, 3, padding='same', activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
        MaxPooling2D(),
        Conv2D(32, 3, padding='same', activation='relu'),
        MaxPooling2D(),
        Conv2D(64, 3, padding='same', activation='relu'),
        MaxPooling2D(),
        Flatten(),
        Dense(512, activation='relu'),
        Dense(1)
    ])
    train_data_gen = train_image_generator.flow_from_directory(batch_size=batch_size,
                                                               directory=mp.get_path_classes(classes)["train"],
                                                               shuffle=True,
                                                               target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                               class_mode='binary')
    val_data_gen = validation_image_generator.flow_from_directory(batch_size=batch_size,
                                                                  directory=mp.get_path_classes(classes)["validation"],
                                                                  target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                                  class_mode='binary')
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    model.summary()
    model.save_weights(checkpoint_dir.format(epoch=0))
    history = model.fit(
        train_data_gen,
        steps_per_epoch=total_train // batch_size,
        epochs=epochs,
        validation_data=val_data_gen,
        validation_steps=total_val // batch_size,
        callbacks=[cp_callback]
    )
    model.save(mp.get_path_classes(classes)["checkpoint"]+"model")
    lock.release();
    #print_result(epochs, history)


def print_result(epochs, history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs_range = range(epochs)
    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()

if __name__ == "__main__":

    path = "/home/bernabei/carla0.8.4/PythonClient/_out/"
    classes_of_modified= ["blur", "black", "brightness", "50_death_pixels", "200_death_pixels","nodemos","noise","sharpness","brokenlens","icelens","banding","greyscale"]
    multiproc=False
    lock= multiprocessing.Lock()
    if multiproc==True:
        for classes in classes_of_modified[5:7]:
            mp = manager_of_path.ManagerOfPath(path, classes_of_modified, True)
            path_checkpoint = "training_1/cp-{epoch:04d}.ckpt"
            p = multiprocessing.Process(target=classificator, args=(lock,mp, classes, path_checkpoint))
            p.start()
            p.join()
    else:
        mp = manager_of_path.ManagerOfPath(path, classes_of_modified[5:7], True)
        path_checkpoint = "training_1/cp-{epoch:04d}.ckpt"
        classificator(mp,classes_of_modified[5],path_checkpoint)



