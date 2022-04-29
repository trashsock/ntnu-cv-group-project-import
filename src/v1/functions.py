import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas as pd
import cv2
from scipy.stats import stats
from PIL import Image
from PIL import ImageFilter
import random

def getIm(typ, amount, arr):
    dupes = []
    for _ in range (amount):
        getint, dupes = getUniqueRand(dupes)
        get = str(getint)
        digits = len(get)
        if digits == 5 and getint < 19379 and typ:
            get = get + "_1"
        while digits < 5:
            get = "0" + get
            digits = len(get)
        if typ:
            get = "cv2206-project/testdata/Positive/" + get + ".jpg"
        else:
            get = "cv2206-project/testdata/Negative/" + get + ".jpg"
        im = cv2.imread(get)
        arr.append(im)

#Returns a rendom number that has not yet been added to the given array
def getUniqueRand(dupes):
    dupe = True
    while dupe:
        ret = random.randint(1, 1000)
        dupe = ret in dupes
    dupes.append(ret)
    return ret, dupes

# Returns two arrays positive and negative which contains amount images each
def getRandomImages(amount, Pos, Neg):
    getIm(True, amount, Pos)
    getIm(False, amount, Neg)
    #convert grayscale here
    return Pos, Neg
def contours(img):
    imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(imgray, 127, 255, 0)
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        max_lencnt = max([len(x) for x in contours])
        return max_lencnt
    return 0

def results(images, sampleType, max = 50):
  positive = 0
  negative = 0
  for i in range(len(images)):
    cont = contours(images[i])
    if cont > max:
      positive += 1
    else:
      negative += 1
  if sampleType.lower() == "positive":
    print("Results after testing", len(images), "images that are positive of having cracks:")
    print("Total Positives: ", positive, "/", len(images))
    print("Total False-negatives: ", negative, "/", len(images))
    print("Accuracy:" , (positive/len(images))*100, "%")
 
  if sampleType.lower() == "negative":
    print("Results after testing", len(images), "images that are negative of having cracks:")
    print("Total Negatives: ", negative, "/", len(images))
    print("Total False-positives: ", positive, "/", len(images))
    print("Accuracy:" , (negative/len(images))*100, "%")   