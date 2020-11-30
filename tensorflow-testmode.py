from __future__ import absolute_import

import multiprocessing

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras import losses
import PIL.Image
import os
import numpy as np
import matplotlib.pyplot as plt
import manager_of_path
import math 
def mcc_metric(true_pos,true_neg,false_pos,false_neg):
  x = (true_pos + false_pos) * (true_pos + false_neg)* (true_neg + false_pos) * (true_neg + false_neg)
  if x==0:
     x=1
  return ((true_pos * true_neg) - (false_pos * false_neg))/math.sqrt(x)

def tester(lock,mp,classes,f):
    lock.acquire()
    batch_size = 4 # batch =divisione del dataset
    IMG_HEIGHT = 800
    IMG_WIDTH = 600
    total_test = 14400
    checkpoint_dir=os.path.dirname(mp.get_path_classes(classes)["checkpoint"])
    test_image_generator = ImageDataGenerator(rescale=1. / 255)  # Generator for our test data
    #manage gpu memory usage
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
    config.log_device_placement = True  # to log device placement (on which device the operation ran)
    sess = tf.Session(config=config)
   
    model1=tf.keras.models.load_model(checkpoint_dir+"/model",compile=False)
    model1.compile(optimizer='adam',
                  loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                  metrics=[tf.keras.metrics.TruePositives(),tf.keras.metrics.TrueNegatives(),tf.keras.metrics.FalsePositives(),tf.keras.metrics.FalseNegatives()])
     test_data_gen = test_image_generator.flow_from_directory(batch_size=batch_size,
                                                              directory=mp.get_path_classes(classes)["test_all"],
                                                              shuffle=True,
                                                              target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                              class_mode='binary')

    ls,tp,tn,fp,fn=model1.evaluate(test_data_gen, verbose=0)
    f.write(classes+":all"+"\n")
    write_cm_mcc(tp,tn,fp,fn,f)
    test_data_gen = test_image_generator.flow_from_directory(batch_size=batch_size,
                                                              directory=mp.get_path_classes(classes)["test"],
                                                              shuffle=True,
                                                              target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                              class_mode='binary')
    ls,tp,tn,fp,fn=model1.evaluate(test_data_gen, verbose=0)
    f.write(classes+":"+classes+"\n")
    write_cm_mcc(tp,tn,fp,fn,f)
    f.close()
    lock.release();
 
def write_cm_mcc(tp,tn,fp,fn,f):
    f.write("TruePositive="+str(tp.item())+" TrueNegative="+str(tn.item())+"\n"+"FalsePositive="+str(fp.item())+" FalseNegative="+str(fn.item())+"\n")
    f.write("MCC="+str(mcc_metric(tp.item(),tn.item(),fp.item(),fn.item()))+"\n")
    f.flush()


if __name__ == "__main__":


    path_check="/home/bernabei/carla0.8.4/PythonClient/_out/"
    classes_of_modified= ["blur", "black", "brightness",  "200_death_pixels","nodemos","noise","sharpness","brokenlens","icelens","banding","greyscale","50_death_pixels","condensation","dirty_lens","chromaticaberration","rain","all"]
    multiproc=True
    lock= multiprocessing.Lock()
    
    for classes in classes_of_modified[:]:
            save_result_file=open(path_check+"result.txt","a")
            mp = manager_of_path.ManagerOfPath(path_check, classes_of_modified, True)
            p = multiprocessing.Process(target=tester, args=(lock,mp, classes,save_result_file));p.start();
            p.join()
    
