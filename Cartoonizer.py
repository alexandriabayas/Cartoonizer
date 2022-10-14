import cv2
import numpy as np

def showImage(image):
    cv2.imshow("My Image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def edgeMask(image, blur, blockSize):
    grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    grayBlur = cv2.medianBlur(grayImage, blur)
    edges = cv2.adaptiveThreshold(grayBlur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, blockSize, blur)
    return edges

def colorQuantization(image, k):
  data = np.float32(image).reshape((-1, 3))

  criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.001)

  ret, label, center = cv2.kmeans(data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
  center = np.uint8(center)
  result = center[label.flatten()]
  result = result.reshape(image.shape)
  return result

#image = cv2.imread("C:\\Users\\abaya\\Documents\\Coding... Stuff\\OpenCV Exploration\\meandafakehead.jpg", cv2.IMREAD_COLOR)
#image = cv2.imread("C:\\Users\\abaya\\Documents\\Coding... Stuff\\OpenCV Exploration\\meandbunny.JPG", cv2.IMREAD_COLOR)
image = cv2.imread("C:\\Users\\abaya\\Documents\\Coding... Stuff\\OpenCV Exploration\\meandstuffedanimals.JPG", cv2.IMREAD_COLOR)

edges = edgeMask(image, 7, 7)
#showImage(edges)

image = colorQuantization(image, 10)
#showImage(image)

cartoon = cv2.bitwise_and(image, image, mask=edges)
showImage(cartoon)

