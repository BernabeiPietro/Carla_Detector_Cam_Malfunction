from __future__ import absolute_import

import math
import multiprocessing
import os

import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import manager_of_path


def classificator(lock, mp, classes, path_checkpoint, f):
    lock.acquire()
    # settings
    batch_size = 4  # batch =divisione del dataset
    epochs = 5  # epochs= numero di volte che un dataset viene ripetuto nella rete
    IMG_HEIGHT = 800
    IMG_WIDTH = 600
    total_train = 240000
    total_val = 60000
    checkpoint_dir = os.path.dirname(mp.get_path_classes(classes)["checkpoint"] + path_checkpoint)
    train_image_generator = ImageDataGenerator(rescale=1. / 255)  # Generator for our training data
    validation_image_generator = ImageDataGenerator(rescale=1. / 255)  # Generator for our validation data
    test_image_generator = ImageDataGenerator(rescale=1. / 255)  # Generator for our test data
    # manage gpu memory usage
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
    config.log_device_placement = True  # to log device placement (on which device the operation ran)
    sess = tf.Session(config=config)
    # tf.set_session(sess)  # set this TensorFlow session as the default session for Keras
    # set checkpoint
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
    loss_train, acc_train, loss_val, acc_val = model.fit(
        train_data_gen,
        steps_per_epoch=total_train // batch_size,
        epochs=epochs,
        validation_data=val_data_gen,
        validation_steps=total_val // batch_size,
        callbacks=[cp_callback]
    )
    f.write(classes + ":train validation" + "\n")
    write_train_test_result(loss_train, acc_train, loss_val, acc_val, save_result_file)
    model.save(mp.get_path_classes(classes)["checkpoint"] + "model")

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                  metrics=[tf.keras.metrics.TruePositives(), tf.keras.metrics.TrueNegatives(),
                           tf.keras.metrics.FalsePositives(), tf.keras.metrics.FalseNegatives()])

    test_data_gen = test_image_generator.flow_from_directory(batch_size=batch_size,
                                                             directory=mp.get_path_classes(classes)["test_all"],
                                                             shuffle=True,
                                                             target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                             class_mode='binary')
    ls, tp, tn, fp, fn = model.evaluate(test_data_gen, verbose=0)
    f.write(classes + ":all" + "\n")
    write_cm_mcc(tp, tn, fp, fn, f)
    test_data_gen = test_image_generator.flow_from_directory(batch_size=batch_size,
                                                             directory=mp.get_path_classes(classes)["test"],
                                                             shuffle=True,
                                                             target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                             class_mode='binary')
    ls, tp, tn, fp, fn = model.evaluate(test_data_gen, verbose=0)
    f.write(classes + ":" + classes + "\n")
    write_cm_mcc(tp, tn, fp, fn, f)
    f.close()
    lock.release();


def write_train_test_result(lst, act, lsv, acv, f):
    f.write("loss_train=" + str(lst.item()) + " acc_train=" + str(act.item()) + "\n" + "loss_val " + str(
        lsv.item()) + " acc_val " + str(acv.item()))
    f.flush()


def write_cm_mcc(tp, tn, fp, fn, f):
    f.write("TruePositive=" + str(tp.item()) + " TrueNegative=" + str(tn.item()) + "\n" + "FalsePositive=" + str(
        fp.item()) + " FalseNegative=" + str(fn.item()) + "\n")
    f.write("MCC=" + str(mcc_metric(tp.item(), tn.item(), fp.item(), fn.item())) + "\n")
    f.flush()


def mcc_metric(true_pos, true_neg, false_pos, false_neg):
    x = (true_pos + false_pos) * (true_pos + false_neg) * (true_neg + false_pos) * (true_neg + false_neg)
    if x == 0:
        x = 1
    return ((true_pos * true_neg) - (false_pos * false_neg)) / math.sqrt(x)


if __name__ == "__main__":

    path = "/home/bernabei/carla0.8.4/PythonClient/_out/"
    classes_of_modified = ["blur", "black", "brightness", "200_death_pixels", "nodemos", "noise", "sharpness",
                           "brokenlens", "icelens", "banding", "greyscale", "50_death_pixels", "condensation",
                           "dirty_lens", "chromaticaberration", "rain", "all"]
    multiproc = True
    save_result_file = open(path + "result.txt", "a")
    lock = multiprocessing.Lock()

    for classes in classes_of_modified:
        mp = manager_of_path.ManagerOfPath(path, classes_of_modified, False)
        path_checkpoint = "training_1/cp-{epoch:04d}.ckpt"
        p = multiprocessing.Process(target=classificator, args=(lock, mp, classes, path_checkpoint))
        p.start();
        print(classes);
        p.join()
