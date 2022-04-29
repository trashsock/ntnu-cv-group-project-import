import matplotlib.image as mpimg
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import cv2

def pcacompv2(img): 
  img = mpimg.imread()
  print("Original image : ", img.shape)
  img_re = np.reshape(img, (1207,1817*4))
  print("Reshaped image : ", img_re.shape)
  pca = PCA(20).fit(img_re)
  img_trans = pca.transform(img_re)
  print("PCA Transformed: ", img_trans.shape)
  img_Inv = pca.inverse_transform(img_trans)
  img = np.reshape(img_Inv,(1207, 1817,4))
  print("Inversed Image : ", img.shape)
  #plt.axis('off')
  #plt.imshow(img.astype('uint8'))
  #plt.show()

def pcacompv1(img):
    img = cv2.cvtColor(cv2.imread('Johan.jpg'), cv2.COLOR_BGR2RGB)
    plt.imshow(img)
    plt.show()

    img.shape

    #Splitting into channels
    blue,green,red = cv2.split(img)
    # Plotting the images
    fig = plt.figure(figsize = (15, 7.2)) 
    fig.add_subplot(131)
    plt.title("Blue Channel")
    plt.imshow(blue)
    fig.add_subplot(132)
    plt.title("Green Channel")
    plt.imshow(green)
    fig.add_subplot(133)
    plt.title("Red Channel")
    plt.imshow(red)
    plt.show()

    df_blue = blue/255
    df_green = green/255
    df_red = red/255

    pca_b = PCA(n_components=50)
    pca_b.fit(df_blue)
    trans_pca_b = pca_b.transform(df_blue)
    pca_g = PCA(n_components=50)
    pca_g.fit(df_green)
    trans_pca_g = pca_g.transform(df_green)
    pca_r = PCA(n_components=50)
    pca_r.fit(df_red)
    trans_pca_r = pca_r.transform(df_red)

    print(trans_pca_b.shape)
    print(trans_pca_r.shape)
    print(trans_pca_g.shape)

    print(f"Blue Channel : {sum(pca_b.explained_variance_ratio_)}")
    print(f"Green Channel: {sum(pca_g.explained_variance_ratio_)}")
    print(f"Red Channel  : {sum(pca_r.explained_variance_ratio_)}")

    fig = plt.figure(figsize = (15, 7.2)) 
    fig.add_subplot(131)
    plt.title("Blue Channel")
    plt.ylabel('Variation explained')
    plt.xlabel('Eigen Value')
    plt.bar(list(range(1,51)),pca_b.explained_variance_ratio_)
    fig.add_subplot(132)
    plt.title("Green Channel")
    plt.ylabel('Variation explained')
    plt.xlabel('Eigen Value')
    plt.bar(list(range(1,51)),pca_g.explained_variance_ratio_)
    fig.add_subplot(133)
    plt.title("Red Channel")
    plt.ylabel('Variation explained')
    plt.xlabel('Eigen Value')
    plt.bar(list(range(1,51)),pca_r.explained_variance_ratio_)
    plt.show()

    b_arr = pca_b.inverse_transform(trans_pca_b)
    g_arr = pca_g.inverse_transform(trans_pca_g)
    r_arr = pca_r.inverse_transform(trans_pca_r)
    print(b_arr.shape, g_arr.shape, r_arr.shape)

    img_reduced= (cv2.merge((b_arr, g_arr, r_arr)))
    print(img_reduced.shape)

    fig = plt.figure(figsize = (10, 7.2)) 
    fig.add_subplot(121)
    plt.title("Original Image")
    plt.imshow(img)
    fig.add_subplot(122)
    plt.title("Reduced Image")
    plt.imshow(img_reduced)
    plt.show()
    return img_reduced

def applyFilter(img):
    #Applying a low-pass filter to the image 
    filterImage = img.filter(PIL.ImageFilter.GaussianBlur)
    return filterImage

def imDeco(img, images):
        w, h = img.size
        i = 0
        img = grayscale(img)
        addImg(img, w, h,images)
        return images
    

def addImg(im, w, h,images):
    size = 9
    w1 = 0
    h1 = 0
    while(w1+size<w and h1+size<h):
        w2 = w1+size
        h2 = h1+size
        croppedimage = im.crop((w1, h1, w2, h2))
        images.append(croppedimage)
        w1 = w1+size
        h1 = h1+size
    return 