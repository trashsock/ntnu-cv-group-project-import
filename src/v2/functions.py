import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas as pd
import cv2
from scipy.stats import stats
import matplotlib
import PIL
from PIL import Image
from PIL import ImageFilter
from scipy import signal
import math
import torch
import random
def pcacomp(img): 
  img = np.array(img)
  #print("Original image : ", img.shape)
  img_re = np.reshape(img, (225,225*3))
  #print("Reshaped image : ", img_re.shape)
  pca = PCA(100).fit(img_re)
  img_trans = pca.transform(img_re)
  #print("PCA Transformed: ", img_trans.shape)
  img_Inv = pca.inverse_transform(img_trans)
  img = np.reshape(img_Inv,(225, 225,3))
  if np.isnan(img).any() or np.isinf(img).any():
    print("NAN")
  #print("Inversed Image : ", img.shape)
  #plt.axis('off')
  #plt.imshow(img.astype('uint8'))
  #plt.show()
  return img

def grayscale(img):
    img = np.array(img)
    r,g,b = img[:,:,0],img[:,:,1],img[:,:,2]
    gamma = 2 #verdi før 1.04
    r_const, g_const, b_const = 0.2126, 0.7152, 0.0722
    grayscale_img = r_const * r ** gamma + g_const * g ** gamma + b_const * b ** gamma
    #grayscale_img = Image.fromarray(np.uint8(grayscale_img))
    return grayscale_img

def checkIfCrack(image):
  #The checkIfCrack function takes image opened through the PIL library as a parameter
  image = np.array(image)
  countTrues = 0
  countFalses = 0
  
  x, y = image.shape[0], image.shape[1]
  for x in range(x):
    for y in range(y):
      #if the gradiant orientation pixel value is 0.0, then there is a symmetry thus a crack. If its not (if its PI) then there is no crack
      
      if image[x,y] == 0.0:
        countTrues += 1
      else:
        countFalses += 1
  #print("Trues:", countTrues)
  #print("Falses:", countFalses)
  max = countFalses + countTrues
  if countTrues > 0.999*max or countFalses > 0.999*max:
    return False
  return True


def convolve(image):
    arr = np.asarray(image)
    if np.isnan(arr).any() or np.isinf(arr).any():
        print("NAN arr")
        arr = np.nan_to_num(arr)
    img = torch.from_numpy(arr)
    X = img
    Y = X.float()      
    val, vec = (torch.eig(Y, eigenvectors = True))
    mask = vec[0]
    l,b = X.size()
    mask = np.reshape(mask,(int(math.sqrt(l)),int(math.sqrt(b))))
    val = val.cpu().numpy()
    vec = vec.cpu().numpy()
    Y = Y.cpu().numpy()
    #convolution to find symmetry to find a pattern to detect cracks
    grad = signal.convolve2d(Y, mask, boundary='symm', mode='same')
    #fig, (ax_orig, ax_ang) = plt.subplots(2, 1, figsize=(6, 15))
    #ax_orig.imshow(Y)
    #ax_orig.set_title('Original')
    #ax_orig.set_axis_off()
    #ax_ang.imshow(np.angle(grad)) 
    #ax_ang.set_title('Gradient orientation')
    #ax_ang.set_axis_off()
    #fig.show()
    #print(np.angle(grad))
    return np.angle(grad)

def results(images, sampleType):
#The results function takes a list of images and a string of sampletype (either positive or negative) as parameters.
  positive = 0
  negative = 0
  for i in range(len(images)):
    images[i] = np.reshape(images[i], (225,225,3))
    grayscaleimg = grayscale(images[i]) 
    convolveImage = convolve(grayscaleimg)
    crack = checkIfCrack(convolveImage)
    if crack == True:
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
        im = PIL.Image.open(get)
        arr.append(im)
      
#Returns a rendom number that has not yet been added to the given array
def getUniqueRand(dupes):
    dupe = True
    while dupe:
        ret = random.randint(1, 1000)
        dupe = ret in dupes
    dupes.append(ret)
    return ret, dupes

#Cuts image to 225x225
def cutIm(arr):
    retArr = []
    for i in arr:
        im = i.crop((0, 0, 225, 225))
        im = pcacomp(im)
        retArr.append(im)
    return retArr

#Need getIm function from v2/functions
# Returns two arrays positive and negative which contains amount images each
def getRandom225x225(amount, Pos, Neg):
    getIm(True, amount, Pos)
    getIm(False, amount, Neg)
    Pos = cutIm(Pos)
    Neg = cutIm(Neg)
    results()
    return Pos, Neg