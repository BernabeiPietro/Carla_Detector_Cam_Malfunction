import cv2
import os
from PIL import Image, ImageEnhance
from matplotlib import pyplot as plt

def open_cv2(pathImage,name):
    array_image=[]
    for n in name:
        array_image.append(cv2.imread(pathImage+"/"+n))
    return array_image
def open_image(pathImage,name):
    array_image = []
    for n in name:
        array_image.append(Image.open(pathImage + "/" + n))
    return array_image

def blur(images,progressiveName,pathModified):

    for img in images:
        blur = cv2.blur(img,(5,5))
        cv2.imwrite(pathModified+str(progressiveName)+".png",blur)
        progressiveName=progressiveName+1
    return progressiveName

def black(images,progressiveName,pathModified):

    for img in images:
        pixels=img.load()
        width, height = img.size
        for i in range(0, width):
            for j in range(0, height):
                pixels[i, j] = (0, 0, 0)
        cv2.imwrite(pathModified + str(progressiveName) + ".png", pixels)
        progressiveName=progressiveName+1
    return progressiveName

def dead_pixel_50(images,progressiveName,pathModified):

    for img in images:
        pixels=img.load()
        width, height = img.size
        for i in range(0, width):
            for j in range(0, height):
                if i == 150 and j == 90:
                    pixels[i, j] = (255, 0, 0)  # rosso per notarlo meglio nell'immagine
        cv2.imwrite(pathModified + str(progressiveName) + ".png", pixels)
        progressiveName=progressiveName+1
    return progressiveName

def birghtness(images,progressiveName,pathModified):

    for img in images:
        enhancer = ImageEnhance.Brightness(img)
        factor = 3.5
        img = enhancer.enhance(factor)
        cv2.imwrite(pathModified + str(progressiveName) + ".png", img)
        progressiveName=progressiveName+1
    return progressiveName

def not_modified(images,progressiveName,pathModified):
    for img in images:
        cv2.imwrite(pathModified + str(progressiveName) + ".png", img)
        progressiveName = progressiveName + 1
    return progressiveName
