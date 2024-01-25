import cv2
import numpy as np
from scipy.interpolate import UnivariateSpline
def cartoonize_image(img):
    # Convert the image to grayscale
    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply median blur to reduce noise
    grey = cv2.medianBlur(grey, 5)

    # Detect edges using adaptive thresholding
    edges = cv2.adaptiveThreshold(grey, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9)

    # Cartoonize the image
    color = cv2.bilateralFilter(img, 9, 250, 250)
    cartoon = cv2.bitwise_and(color, color, mask=edges)

    return cartoon
def pixelize_image(img, pixel_size=10):
    # Get the image dimensions
    height, width = img.shape[:2]

    # Resize the image to a smaller size
    small_img = cv2.resize(img, (width // pixel_size, height // pixel_size), interpolation=cv2.INTER_LINEAR)

    # Resize the small image back to the original size
    pixelized_img = cv2.resize(small_img, (width, height), interpolation=cv2.INTER_NEAREST)

    return pixelized_img



def apply_emboss_effect(img):
    # Define the emboss kernel
    emboss_kernel = np.array([[-2, -1, 0],
                              [-1, 1, 1],
                              [0, 1, 2]])

    # Apply the convolution operation to the image
    emboss_img = cv2.filter2D(img, -1, emboss_kernel)

    return emboss_img

def adjust_lightness(image_path, daylight_factor):

    # Convert to HLS color space
    image_HLS = cv2.cvtColor(image_path, cv2.COLOR_BGR2HLS)
    image_HLS = np.array(image_HLS, dtype=np.float64)

    # Adjust lightness
    image_HLS[:, :, 1] = image_HLS[:, :, 1] * daylight_factor
    image_HLS[:, :, 1][image_HLS[:, :, 1] > 255] = 255

    # Convert back to BGR color space
    image_HLS = np.array(image_HLS, dtype=np.uint8)
    image_RGB = cv2.cvtColor(image_HLS, cv2.COLOR_HLS2BGR)

    return image_RGB

def tv60(image_path, threshold=0.8, noise_range=64):

    height, width = image_path.shape[:2]

    gray = cv2.cvtColor(image_path, cv2.COLOR_BGR2GRAY)

    for i in range(height):
        for j in range(width):
            if np.random.rand() <= threshold:
                if np.random.randint(2) == 0:
                    gray[i, j] = min(gray[i, j] + np.random.randint(0, noise_range), 255)
                else:
                    gray[i, j] = max(gray[i, j] - np.random.randint(0, noise_range), 0)

    return gray
def LookupTable(x, y):
  spline = UnivariateSpline(x, y)
  return spline(range(256))
def pencil_sketch_col(img):

    #inbuilt function to create sketch effect in colour and greyscale
    sk_gray, sk_color = cv2.pencilSketch(img, sigma_s=60, sigma_r=0.07, shade_factor=0.1)

    return  sk_color
def pencil_sketch_grey(img):

    #inbuilt function to create sketch effect in colour and greyscale
    sk_gray, sk_color = cv2.pencilSketch(img, sigma_s=60, sigma_r=0.07, shade_factor=0.1)

    return  sk_gray
def sepia(img):


    img_sepia = np.array(img, dtype=np.float64) # converting to float to prevent loss
    img_sepia = cv2.transform(img_sepia, np.matrix([[0.272, 0.534, 0.131],
                                    [0.349, 0.686, 0.168],
                                    [0.393, 0.769, 0.189]])) # multipying image with special sepia matrix
    img_sepia[np.where(img_sepia > 255)] = 255 # normalizing values greater than 255 to 255
    img_sepia = np.array(img_sepia, dtype=np.uint8)

    return img_sepia
def sharpen(img):


    kernel = np.array([[-1, -1, -1], [-1, 9.5, -1], [-1, -1, -1]])
    img_sharpen = cv2.filter2D(img, -1, kernel)

    return img_sharpen
def HDR(img):

    hdr = cv2.detailEnhance(img, sigma_s=12, sigma_r=0.15)
    return  hdr
def invert(img):

    inv = cv2.bitwise_not(img)
    return inv
def Winter(img):

    increaseLookupTable = LookupTable([0, 64, 128, 256], [0, 80, 160, 256])
    decreaseLookupTable = LookupTable([0, 64, 128, 256], [0, 50, 100, 256])
    blue_channel, green_channel,red_channel = cv2.split(img)
    red_channel = cv2.LUT(red_channel, decreaseLookupTable).astype(np.uint8)
    blue_channel = cv2.LUT(blue_channel, increaseLookupTable).astype(np.uint8)
    win= cv2.merge((blue_channel, green_channel, red_channel))

    return win
def Summer(img):

    increaseLookupTable = LookupTable([0, 64, 128, 256], [0, 80, 160, 256])
    decreaseLookupTable = LookupTable([0, 64, 128, 256], [0, 50, 100, 256])
    blue_channel, green_channel,red_channel  = cv2.split(img)
    red_channel = cv2.LUT(red_channel, increaseLookupTable).astype(np.uint8)
    blue_channel = cv2.LUT(blue_channel, decreaseLookupTable).astype(np.uint8)
    sum= cv2.merge((blue_channel, green_channel, red_channel ))
    return sum