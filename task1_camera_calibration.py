import cv2
import os

def calibrate(directory):
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        print(filename)
        image = cv2.imread(directory + filename)
        cv2.imshow(filename, image)
        cv2.waitKey(0)
    pass