import os
import cv2
import numpy as np
import pytesseract
from matplotlib import pyplot as plt
from PIL import Image
from pdf2image import convert_from_path


def text_recognition(lang):
    file_path = 'images/' + input('Enter file path: ')

    if file_path.endswith('.pdf'):
        pages = convert_from_path(file_path, 200)
        pages[0].save('images/out.jpg', 'JPEG')

        img = cv2.imread('images/out.jpg')
        # imgcv = cv2.cvtColor(imgcv, cv2.COLOR_BGR2RGB)

        custom_config = r'--oem 3 --psm 3'
        result = pytesseract.image_to_string(img, lang=lang, config=custom_config)

    else:
        img = cv2.imread(file_path)
        # img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        custom_config = r'--oem 3 --psm 6'
        result = pytesseract.image_to_string(img, lang=lang, config=custom_config)

    return result


def main():
    # pdf to string
    # print(text_recognition('rus'))

    # grayscale
    img = cv2.imread('images/otchet.jpg')
    gray_image = grayscale(img)
    cv2.imwrite("images/gray.jpg", gray_image)

    # color
    img2 = cv2.imread('images/gray.jpg')
    img2_rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    cv2.imwrite("images/color.jpg", img2_rgb)

    # noise remove
    img3 = cv2.imread('images/noisy_picture.jpeg')
    no_noise = noise_removal(img3)  # im_bw)
    cv2.imwrite("images/no_noise.jpg", no_noise)

    # resize
    new_img = resize_img(img, 2400, 1800)
    # cv2.imshow('Resize', new_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # rotate
    rotate_img = rotate(img, 45)
    cv2.imwrite("images/rotate.jpg", rotate_img)


def grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


# Rotate the image around its center
def rotate(image, angle: float):
    newImage = image.copy()
    (rows, cows) = newImage.shape[:2]
    center = (rows // 2, cows // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    new_img = cv2.warpAffine(newImage, M, (rows, cows))
    return new_img


def noise_removal(image):
    kernel = np.ones((1, 1), np.uint8)
    image = cv2.dilate(image, kernel, iterations=1)
    kernel = np.ones((1, 1), np.uint8)
    image = cv2.erode(image, kernel, iterations=1)
    image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    image = cv2.medianBlur(image, 3)
    return image


# a, b - size
def resize_img(image, a, b):
    img = cv2.resize(image, (a, b))
    return img


if __name__ == '__main__':
    main()