import cv2
import os
from PIL import Image, ImageEnhance
from matplotlib import pyplot as plt
import numpy as np


def open_cv2(pathImage,name):
    array_image=[]
    for n in name:
        array_image.append(cv2.imread(pathImage+"/"+n))
    return array_image
def cv2toPIL(img):
    img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    return Image.fromarray(img)
def PIltocv2(img):

    numpy_image = np.array(img)
    numpy_image=np.uint8(img)

    return cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)

def blur(images,progressiveName,pathModified): #sistemato

    for img in images:
        blur = cv2.blur(img,(5,5))
        cv2.imwrite(pathModified+str(progressiveName)+".png",blur)
        progressiveName=progressiveName+1
    return progressiveName

def black(images,progressiveName,pathModified): #sistemato
    #metodo che necessita immagini in formato PIL
    for img in images:
        img=cv2toPIL(img)
        pixels=img.load()
        width, height = img.size
        for i in range(0, width):
            for j in range(0, height):
                pixels[i, j] = (0, 0, 0)
        pixels=PIltocv2(img)
        cv2.imwrite(pathModified + str(progressiveName) + ".png", pixels)
        progressiveName=progressiveName+1
    return progressiveName

def dead_pixel_50(images,progressiveName,pathModified):

    for img in images:
        h, w, _ = img.shape

        count = 0
        w1 = w2 = int((w - 70) / 10)  # in base a questo 10 e al 5 sotto decido quanti pixel neri inserire
        h1 = h2 = int((h - 70) / 5)
        w2 = w3 = w2 - 5
        h2 = h3 = h2 - 5
        countpixel = 0

        # naturalmente bisogna sempre controllare con un paio di prove (o calcolandolo)
        # se si esce confini dell'immagine in elaborazione
        for y in range(0, 5):
            h2 = h2 + (h1 * y)
            for x in range(0, 10):
                img[h2, w2] = (255, 0, 0)  # pixels rossi (per visibilità) in coordinate [h2,w2]
                countpixel = countpixel + 1
                w2 = w2 + w1
                if x == 9:
                    h2 = h3
                    w2 = w3
        cv2.imwrite(pathModified + str(progressiveName) + ".png", img)
        progressiveName=progressiveName+1
    return progressiveName

def birghtness(images,progressiveName,pathModified):

    for img in images:
        img= cv2toPIL(img)
        enhancer = ImageEnhance.Brightness(img)
        factor = 3.5
        img = enhancer.enhance(factor)
        img=PIltocv2(img)
        cv2.imwrite(pathModified + str(progressiveName) + ".png", img)
        progressiveName=progressiveName+1
    return progressiveName

def not_modified(images,progressiveName,pathModified):
    for img in images:
        cv2.imwrite(pathModified + str(progressiveName) + ".png", img)
        progressiveName = progressiveName + 1
    return progressiveName
