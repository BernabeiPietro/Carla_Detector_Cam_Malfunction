import copy

import cv2
import numpy as np
from PIL import Image, ImageEnhance

import chromaticaberration2
import manager_of_path


def open_cv2(pathImage, name):
    array_image = []
    for n in name:
        array_image.append(cv2.imread(pathImage + "/" + n))
    return array_image


def cv2toPIL(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(img)


def PIltocv2(img):
    numpy_image = np.array(img)
    numpy_image = np.uint8(img)

    return cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)


def run_methods(function, file, name, path):
    function = dispatcher[function]
    return function(file, name, path)


def blur(images, progressiveName, pathModified):  # sistemato

    for img in images:
        blur = cv2.blur(img, (5, 5))
        cv2.imwrite(pathModified + str(progressiveName) + ".png", blur)
        progressiveName = progressiveName + 1
    return progressiveName


def black(images, progressiveName, pathModified):  # sistemato
    # metodo che necessita immagini in formato PIL
    for img in images:
        img = cv2toPIL(img)
        pixels = img.load()
        width, height = img.size
        for i in range(0, width):
            for j in range(0, height):
                pixels[i, j] = (0, 0, 0)
        pixels = PIltocv2(img)
        cv2.imwrite(pathModified + str(progressiveName) + ".png", pixels)
        progressiveName = progressiveName + 1
    return progressiveName


def death_pixel_50(imag, progressiveName, pathModified):
    images = copy.deepcopy(imag)

    for img in images:
        h, w, _ = img.shape

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
        progressiveName = progressiveName + 1
    return progressiveName


def death_pixel_200(imag, progressiveName, pathModified):
    images = copy.deepcopy(imag)
    for img in images:
        h, w, _ = img.shape

        w1 = w2 = int(w / 20)  # in base a questo 20 e al 10 sotto decido quanti pixel neri inserire
        h1 = h2 = int(h / 10)
        w2 = w3 = w2 - 5
        h2 = h3 = h2 - 5
        countpixel = 0
        # naturalmente bisogna sempre controllare con un paio di prove (o calcolandolo)
        # se si esce confini dell'immagine in elaborazione

        for y in range(0, 10):
            h2 = h2 + (h1 * y)
            for x in range(0, 20):
                img[h2, w2] = (255, 0, 0)  # rossi per visibilità
                countpixel = countpixel + 1
                w2 = w2 + w1
                if x == 19:
                    h2 = h3
                    w2 = w3

        cv2.imwrite(pathModified + str(progressiveName) + ".png", img)
        progressiveName = progressiveName + 1
    return progressiveName


def chromaticaberration(images, progressiveName, pathModified):
    for img in images:
        img = cv2toPIL(img)
        imOriginal = img
        i = 0
        j = 0
        a = img.size[0]
        b = img.size[1]
        # Ensure width and height are odd numbers
        if (img.size[0] % 2 == 0):
            img = img.crop((0, 0, img.size[0] - 1, img.size[1]))
            img.load()
            i = 1
        if (img.size[1] % 2 == 0):
            img = img.crop((0, 0, img.size[0], img.size[1] - 1))
            img.load()
            j = 1
        # 1, no blur
        # 2, no blur
        # 1, blur
        # 2, blur
        img = chromaticaberration2.add_chromatic(img, strength=1,
                                                 no_blur=False)  # metodo da invocare per aggiungere effetto
        imOriginal.paste(img)
        pixels = PIltocv2(imOriginal)
        cv2.imwrite(pathModified + str(progressiveName) + ".png", pixels)
        progressiveName = progressiveName + 1
    return progressiveName


def condensation(images, progressiveName, pathModified):
    img = images[0]
    img = cv2toPIL(img)
    imgM = get_image_blend(img, "condensation", "RGBA")
    progressiveName = paste_image(images, imgM, pathModified, progressiveName)
    return progressiveName


def rain(images, progressiveName, pathModified):
    img = images[0]
    img = cv2toPIL(img)
    imgM = get_image_blend(img, "rain", "RGBA")
    progressiveName = paste_image(images, imgM, pathModified, progressiveName)
    return progressiveName


def dirty_lens(images, progressiveName, pathModified):
    img = images[0]
    img = cv2toPIL(img)
    imgM = get_image_blend(img, "lensDirt")
    progressiveName = overlap(images, progressiveName, pathModified, imgM, [0.5], 1.6)
    return progressiveName


def brightness(images, progressiveName, pathModified):
    for img in images:
        img = cv2toPIL(img)
        enhancer = ImageEnhance.Brightness(img)
        factor = 3.5
        img = enhancer.enhance(factor)
        img = PIltocv2(img)
        cv2.imwrite(pathModified + str(progressiveName) + ".png", img)
        progressiveName = progressiveName + 1
    return progressiveName


def greyscale(images, progressiveName, pathModified):
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(pathModified + str(progressiveName) + ".png", img)
        progressiveName = progressiveName + 1
    return progressiveName


def nodemos(images, progressiveName, pathModified):
    for img in images:
        w, h, _ = img.shape
        # Create target array, twice the size of the original image
        res_array = np.zeros((2 * w, 2 * h, 3), dtype=np.uint8)
        # Map the RGB values in the original picture according to the BGGR pattern#
        # Blue
        res_array[::2, ::2, 2] = img[:, :, 2]
        # Green (top row of the Bayer matrix)
        res_array[1::2, ::2, 1] = img[:, :, 1]
        # Green (bottom row of the Bayer matrix)
        res_array[::2, 1::2, 1] = img[:, :, 1]
        # Red
        res_array[1::2, 1::2, 0] = img[:, :, 0]
        img = cv2.cvtColor(res_array, cv2.COLOR_BGR2RGB)
        cv2.imwrite(pathModified + str(progressiveName) + ".png", img)
        progressiveName = progressiveName + 1
    return progressiveName


def noise(images, progressiveName, pathModified):
    for img in images:
        gauss = np.random.normal(0, 1, img.size)
        gauss = gauss.reshape(img.shape[0], img.shape[1], img.shape[2]).astype('uint8')
        img = img + img * gauss
        cv2.imwrite(pathModified + str(progressiveName) + ".png", img)
        progressiveName = progressiveName + 1
    return progressiveName


def sharpness(images, progressiveName, pathModified):
    for img in images:
        img = cv2toPIL(img)
        enhancer = ImageEnhance.Sharpness(img)
        factor = -3.5
        img = enhancer.enhance(factor)
        img = PIltocv2(img)
        cv2.imwrite(pathModified + str(progressiveName) + ".png", img)
        progressiveName = progressiveName + 1
    return progressiveName


def banding(images, progressiveName, pathModified):
    img = images[0]
    img = cv2toPIL(img)
    valueOfImage = [0.02, 0.05]
    imgM = get_image_blend(img, "banding")
    progressiveName = overlap(images, progressiveName, pathModified, imgM, valueOfImage)
    return progressiveName


def icelens(images, progressiveName, pathModified):
    img = images[0]
    img = cv2toPIL(img)
    imgM = get_image_blend(img, "ice", "RGBA")
    progressiveName = paste_image(images, imgM, pathModified, progressiveName)
    return progressiveName


def brokenlens(images, progressiveName, pathModified):
    img = images[0]
    img = cv2toPIL(img)
    valueOfImage = [0.5]
    imgM = get_image_blend(img, "lensBroken")
    i = 0
    progressiveName = overlap(images, progressiveName, pathModified, imgM, valueOfImage, 1.6)
    return progressiveName


def not_modified(images, progressiveName, pathModified):
    for img in images:
        cv2.imwrite(pathModified + str(progressiveName) + ".png", img)
        progressiveName = progressiveName + 1
    return progressiveName


def get_image_blend(img, name, i_mode="img.mode"):
    imgM = []
    if (i_mode == "img.mode"):
        i_mode = img.mode
    for i in manager_of_path.open_image(name):
        img2 = Image.open(i).convert(i_mode)
        imgM.append(img2.resize(img.size))
    return imgM


def overlap(images, progressiveName, pathModified, image_overlap, overlap_value, factor="null"):
    i = 0
    for img in images:
        img = cv2toPIL(img)
        img = Image.blend(img, image_overlap[i], overlap_value[i])
        if (factor != "null"):
            enhancer = ImageEnhance.Brightness(img)
            img = enhancer.enhance(factor)  # aggiungo luminosità
        img = PIltocv2(img)
        cv2.imwrite(pathModified + str(progressiveName) + ".png", img)
        progressiveName = progressiveName + 1
        i = (i + 1) % len(overlap_value)

    return progressiveName


def paste_image(images, imgM, pathModified, progressiveName):
    i = 0
    for img in images:
        img = cv2toPIL(img)
        img.paste(imgM[i], (0, 0), imgM[i])
        i = (i + 1) % len(imgM)
        img = PIltocv2(img)
        cv2.imwrite(pathModified + str(progressiveName) + ".png", img)
        progressiveName = progressiveName + 1
    return progressiveName


dispatcher = {"blur": blur, "black": black, "brightness": brightness, "200_death_pixels": death_pixel_200,
              "nodemos": nodemos, "noise": noise, "sharpness": sharpness, "brokenlens": brokenlens, "icelens": icelens,
              "banding": banding, "50_death_pixels": death_pixel_50, "greyscale": greyscale,
              "condensation": condensation, "dirty_lens": dirty_lens, "chromaticaberration": chromaticaberration,
              "rain": rain}
