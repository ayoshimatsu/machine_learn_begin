import numpy as np
import matplotlib.pyplot as plt
import cv2

def aidemy_imshow(name, img):
    b,g,r = cv2.split(img)
    img = cv2.merge([r,g,b])
    plt.imshow(img)
    plt.show()


# overwrite ======
cv2.imshow = aidemy_imshow

img = cv2.imread("pictures/sample.JPG")
filt = np.array([[0, 1, 0],
                 [1, 0, 1],
                 [0, 1, 0]], np.uint8)
my_img = cv2.dilate(img, filt)
cv2.imshow("sample", my_img)

# remove noise =====
my_img = cv2.fastNlMeansDenoisingColored(img)

# blur =====
#my_img = cv2.GaussianBlur(img, (301, 301), 0)

# masking =====
#mask = cv2.imread("pictures/mask.png", 0)
#mask = cv2.resize(mask, (img.shape[1], img.shape[0]))
#retval, mask = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY_INV)  # invert color
#my_img = cv2.bitwise_and(img, img, mask=mask)

# Threshold =====
#retval, my_img = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY)

# chnage color =====
#my_img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

# Resize or rotate by affine =====
#mat = cv2.getRotationMatrix2D(tuple(np.array((img.shape[1], img.shape[0])) / 2), 180, 2)
#my_img = cv2.warpAffine(img, mat, (img.shape[1], img.shape[0]))

# Resize =====
#size = img.shape
#my_img = img[: size[0] // 2, : size[1] // 3]
#my_img = cv2.resize(my_img, (my_img.shape[1] * 2, my_img.shape[0] * 2))

# create a new image =====
# img_size = (512, 512)
# img = np.array([[[0, 0, 255] for _ in range(img_size[1])] for _ in range(img_size[0])], dtype="uint8")
