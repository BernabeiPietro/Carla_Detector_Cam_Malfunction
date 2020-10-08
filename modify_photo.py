import cv2
import numpy as np
from PIL import Image, ImageEnhance
import chromaticaberration2


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


def dead_pixel_50(images, progressiveName, pathModified):
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


def dead_pixel_200(images, progressiveName, pathModified):
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
    imgM = []
    for i in range(1, 3):
        img2 = Image.open("condensation/condensation" + str(i) + ".png").convert("RGBA")
        imgM.append(img2.resize(img.shape[0:2]))
    i = 0
    for img in images:
        img = cv2toPIL(img)
        img.paste(imgM[i], (0, 0), imgM[i])
        i = (i + 1) % len(imgM)
        img = PIltocv2(img)
        cv2.imwrite(pathModified + str(progressiveName) + ".png", img)
        progressiveName = progressiveName + 1
    return progressiveName


def rain(images, progressiveName, pathModified):
    img = images[0]
    imgM = []
    for i in range(1, 5):
        img2 = Image.open("rain/rain" + str(i) + ".png").convert("RGBA")
        imgM.append(img2.resize(img.shape[0:2]))
    i = 0
    for img in images:
        img = cv2toPIL(img)
        img.paste(imgM[i], (0, 0), imgM[i])
        i = (i + 1) % len(imgM)
        img = PIltocv2(img)
        cv2.imwrite(pathModified + str(progressiveName) + ".png", img)
        progressiveName = progressiveName + 1
    return progressiveName


def dirty_lens(images, progressiveName, pathModified):
    img = images[0]
    img = cv2toPIL(img)
    imgM = []
    for i in range(1, 37):
        img2 = Image.open("lensDirt/LensDirt-" + str(i) + ".png").convert(img.mode)
        imgM.append(img2.resize(img.size))
    i = 0
    for img in images:
        img = cv2toPIL(img)
        img = Image.blend(img, imgM[i], 0.5)  # valori per immagine banding->0.02, banding1->0.05, ice->0.2
        i = (i + 1) % len(imgM)
        enhancer = ImageEnhance.Brightness(img)
        factor = 1.6  # aggiungo luminosità
        img = enhancer.enhance(factor)
        img = PIltocv2(img)
        cv2.imwrite(pathModified + str(progressiveName) + ".png", img)
        progressiveName = progressiveName + 1
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
        # img= cv2toPIL(img)
        # img=img.convert('L')
        # img=PIltocv2(img)
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


def overlap(images, progressiveName, pathModified, overlap_img_path, overlap_value):
    image_overlap = Image.open(overlap_img_path)

    for img in images:
        img = cv2toPIL(img)
        img_over = image_overlap.convert(img.mode)
        img_over = img_over.resize(img.size)
        img = Image.blend(img, img_over, overlap_value)
        img = PIltocv2(img)
        cv2.imwrite(pathModified + str(progressiveName) + ".png", img)
        progressiveName = progressiveName + 1
    return progressiveName


def not_modified(images, progressiveName, pathModified):
    for img in images:
        cv2.imwrite(pathModified + str(progressiveName) + ".png", img)
        progressiveName = progressiveName + 1
    return progressiveName
