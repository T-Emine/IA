# from skimage.segmentation import mark_boundaries
# from skimage.segmentation import slic
# import matplotlib.pyplot as plt
# import scipy.misc as im
# from sklearn import svm
# import seaborn as sns
# import numpy as np
#
# path_img = 'image2.jpg'
#
# # charger une image
# img = im.imread(path_img)
#
# # utilisation de l'algorithme SLIC
# superpixel_labels = slic(img, n_segments=1000, compactness=10)
#
# # récupérer les canaux de couleur de l'image
# red = img[:, :, 0]
# green = img[:, :, 1]
# blue = img[:, :, 2]
#
# # recuperer le nombre de superpixels
# nb_superpixels = np.max(superpixel_labels) + 1
#
# # attribuer la couleur moyenne de chaque superpixel
# # aux pixels lui appartenant
# for label in range(nb_superpixels):
#     idx = superpixel_labels == label
#     red[idx] = np.mean(red[idx])
#     green[idx] = np.mean(green[idx])
#     blue[idx] = np.mean(blue[idx])
#
# # dessiner les bordures des superpixels en noir et blanc
# img = mark_boundaries(img, superpixel_labels, \
#                       color=(1, 1, 1), outline_color=(0, 0, 0), \
#                       mode='outer')
#
#
# # afficher le résultat
# plt.imshow(img)
# plt.show()
#
# #-----------------------------------------------------------------------------------------
#
#
# # largeur et hauteur de l'image
# width = np.shape(red)[1]
# height = np.shape(red)[0]
#
# # pour calculer la position du barycentre dans la largeur de l'image
# x_idx = np.repeat(range(width), height)
# x_idx = np.reshape(x_idx, [width, height])
# x_idx = np.transpose(x_idx)
#
# # pour calculer la position du barycentre dans la hauteur de l'image
# y_idx = np.repeat(range(height), width)
# y_idx = np.reshape(y_idx, [height, width])
#
# # extraire les caractéristiques de chaque superpixel
# feature_superpixels = []
#
# for label in range(nb_superpixels):
#     # pixels appartenant au superpixels
#     idx = superpixel_labels == label
#
#     # calcul et normalisation de la couleur moyenne
#     c1_mean = np.mean(red[idx]) / 255
#     c2_mean = np.mean(green[idx]) / 255
#     c3_mean = np.mean(blue[idx]) / 255
#
#     # calcul et normalisation de la position du barycentre
#     x_mean = np.mean(x_idx[idx]) / (width - 1)
#     y_mean = np.mean(y_idx[idx]) / (height - 1)
#
#     # constitution du vecteur à 5 dimension
#     sp = [c1_mean, c2_mean, c3_mean, x_mean, y_mean]
#     feature_superpixels.append(sp)
#
# # ----------------------------------------------------------------------------------------------------------------------
#
# path_learning = 'img_learning2.png'
# # lire l'image indiquant quels superpixels doivent être utilisés
# # pour l'apprentissage
# img_learning = im.imread(path_learning)
#
# # récupérer les canaux de couleur de l'image
# red_learning = img_learning[:, :, 0]
# green_learning = img_learning[:, :, 1]
# blue_learning = img_learning[:, :, 2]
#
# # récupérer la couleur associée à chaque classe
# class_colors = [(red_learning[y, x], green_learning[y, x], blue_learning[y, x]) for y in range(height) \
#                 for x in range(width)]
# class_colors = set(class_colors)
# class_colors.remove((0, 0, 0))
# class_colors = list(class_colors)
# # recupérer pour chaque classe tous les pixels qui lui sont attribués
# class_pixels = []
# for color in class_colors:
#     learning_pixels = (red_learning == color[0]) \
#                       & (green_learning == color[1]) \
#                       & (blue_learning == color[2])
#     class_pixels.append(learning_pixels)
# # recupérer pour chaque classe quelques superpixels représentatifs
# X = []  # caractéristiques des superpixels
# Y = []  # identifiant de la classe
# for label in range(nb_superpixels):
#     # parcour l'ensemble des pixels du
#     # superpixel et regarder combien
#     # d'entre eux sont attribués à
#     # chaque classe
#     nb_for_each_class = []
#     idx_sp = superpixel_labels == int(label)
#     for learning_pixels in class_pixels:
#         # print("----------------------------------------------")
#         # print(idx_sp)
#         # print(learning_pixels)
#         # print("----------------------------------------------")
#
#         common_idx = np.logical_and(learning_pixels, idx_sp)
#         nb_for_each_class.append(np.sum(common_idx))
#     # tester si le superpixel contient des pixels
#     # appartenant à une et une seule classe
#     class_idx = -1
#     several_classes = 0
#     for idx in range(len(nb_for_each_class)):
#         if nb_for_each_class[idx] > 0:
#             if class_idx < 0:
#                 # le superpixel contient
#                 # des pixels appartenant
#                 # à l'une des classes
#                 class_idx = idx
#             else:
#                 # le superpixel contient
#                 # des pixels appartenant
#                 # à plusieurs classes :
#                 # ne pas le retenir comme
#                 # donnée d'apprentissage
#                 several_classes = True
#     # si le superpixel a été retenu comme donnée
#     # d'apprentissage, on stocker ses caractéristiques
#     # et l'identifiant de la classe
#     if (class_idx >= 0) and not several_classes:
#         Y.append(class_idx)
#         X.append(feature_superpixels[label])
#
# # ----------------------------------------------------------------------------------
# # creer le séparateur à vaste marge
# model_svm = svm.SVC(decision_function_shape='ovo')
# # paramètre du SVM permettant d'influencer la proportion
# # de données d'apprentissage pouvant être considérées comme
# # erronées
# model_svm.C = 4.
# # paramètre du noyau du SVM
# model_svm.gamma = 4.
# # indiquer que les probabilités d'appartenir à chaque classe
# # doivent être calculées
# model_svm.probability = True
# # entraîner le SVM
# model_svm.fit(X, Y)
#
# # ------------------------------------------------------------------------------------
# # predire la probabilité de chaque superpixel
# # d'appartenir à chacune des classes
# probas = model_svm.predict_proba(feature_superpixels)
# # predire la classe la plus probable pour chaque superpixel
# classification = model_svm.predict(feature_superpixels)
#
# # -------------------------------------------------------------------------------------
# # parcourir chacune des classes
# for class_id in range(len(class_colors)):
#     pixel_probas = np.zeros([height, width])
#     # transférer la probabilité du superpixel
#     # aux pixels qui le constituent
#     for label in range(nb_superpixels):
#         idx = superpixel_labels == label
#         pixel_probas[idx] = probas[label, class_id]
#     # afficher le résultat
#     plt.figure(figsize=(16, 8))
#     sns.heatmap(pixel_probas, xticklabels=False, yticklabels=False)
#     plt.show()





#
# # import the necessary packages
# from skimage.segmentation import slic
# from skimage.segmentation import mark_boundaries
# from skimage.util import img_as_float
# import matplotlib.pyplot as plt
# import numpy as np
# import argparse
# import cv2
#
# # construct the argument parser and parse the arguments
# ap = argparse.ArgumentParser()
# ap.add_argument("-i", "--image", required=True, help="Path to the image")
# args = vars(ap.parse_args())
#
# # load the image and apply SLIC and extract (approximately)
# # the supplied number of segments
#     # construct a mask for the segment
#     print
#     "[x] inspecting segment %d" % (i)
#     mask = np.zeros(image.shape[:2], dtype="uint8")
#     mask[segments == segVal] = 255
#
#     # show the masked region
#     cv2.imshow("Mask", mask)
#     cv2.imshow("Applied", cv2.bitwise_and(image, image, mask=mask))
#     cv2.waitKey(0)


import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib.image as mpimg
from PIL import Image
#
# cap = cv2.VideoCapture
#
# while(1):
#
#     # Take each frame
#     _, frame = cap.read()



# class_names = ['left', 'right', 'center']
# train_labels = []

import argparse
import dlib

list_images =[]

class Image:
    def __init__(self):
        pass

    def prepocessing(self, img_name):
        filename = img_name
        img = cv2.imread(filename)

        green = np.uint8([[[127, 145, 146]]])
        hsvGreen = cv2.cvtColor(green, cv2.COLOR_BGR2HSV)
        print(hsvGreen)
        lowerLimit = (hsvGreen[0][0][0] - 10, 100, 100)
        upperLimit = (hsvGreen[0][0][0] + 10, 255, 255)
        print(upperLimit)
        print(lowerLimit)

        j = 1
        while (j <= 12):
            filename = './training_img/' + str(j) + '.jpg'
            img = cv2.imread(filename)

            # Convert BGR to HSV
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

            # define range of blue color in HSV
            # lower_blue = np.array([69,8.2,67.1])
            # upper_blue = np.array([69,18.8,27.1])
            lower_blue = np.array([28, 19, 128])
            upper_blue = np.array([88, 255, 255])

            # Threshold the HSV image to get only blue colors
            mask = cv2.inRange(hsv, lower_blue, upper_blue)
            cv2.imshow('mask', mask)

            # Bitwise-AND mask and original image
            res = cv2.bitwise_and(img, img, mask=mask)

            cv2.imshow('frame', img)
            cv2.imshow('mask', mask)
            cv2.imshow('res', res)

            k = cv2.waitKey(5) & 0xFF
            if k == 27:
                break

            print(j)

            path = "./training_img/img_modif/masko" + str(j) + ".jpg";
            cv2.imwrite(path, mask);
            list_images.append(path)

            # i = 1
            # for element in os.listdir('./training_img'):
            #     imgpil = Image.open('./training_img/'+str(i)+'.jpg')
            #     # anciennement np.asarray
            #     img = np.array(imgpil)
            #     i += 1
            #     train_labels.append(img)
            #     imgpil.save("resultat.jpg")

            j += 1

        cv2.destroyAllWindows()



class BoxSelector(object):
    def __init__(self, image, window_name,color=(0,0,255)):
        #store image and an original copy
        self.image = image
        self.orig = image.copy()
        #capture start and end point co-ordinates
        self.start = None
        self.end = None
        #flag to indicate tracking
        self.track = False
        self.color = color
        self.window_name = window_name
        #hook callback to the named window
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name,self.mouseCallBack)


    def mouseCallBack(self, event, x, y, flags, params):
        #start tracking if left-button-clicked down
        if event==cv2.EVENT_LBUTTONDOWN:
            self.start = (x,y)
            self.track = True
        #capture/end tracking while mouse-move or left-button-click released
        elif self.track and (event==cv2.EVENT_MOUSEMOVE or event==cv2.EVENT_LBUTTONUP):
            self.end = (x,y)
            if not self.start==self.end:
                self.image = self.orig.copy()
                #draw rectangle on the image
                cv2.rectangle(self.image, self.start, self.end, self.color, 2)
                if event==cv2.EVENT_LBUTTONUP:
                    self.track=False
            #in case of clicked accidently, reset tracking
            else:
                self.image = self.orig.copy()
                self.start = None
                self.track = False
            cv2.imshow(self.window_name,self.image)


    @property
    def roiPts(self):
        if self.start and self.end:
            pts = np.array([self.start, self.end])
            s = np.sum(pts, axis=1)
            (x, y) = pts[np.argmin(s)]
            (xb, yb) = pts[np.argmax(s)]
            return [(x, y), (xb, yb)]
        else:
            return []

def main():
    img = Image()
    name = 'image2.jpg'
    img.prepocessing(name)

    ap = argparse.ArgumentParser()
    ap.add_argument("-d","--dataset",required=True,help="path to images dataset...")
    ap.add_argument("-a","--annotations",required=True,help="path to save annotations...")
    ap.add_argument("-i","--images",required=True,help="path to save images")
    args = vars(ap.parse_args())

    #annotations and image paths
    annotations = []
    imPaths = []
    #loop through each image and collect annotations
    for imagePath in list_images:
        #load image and create a BoxSelector instance
        image = cv2.imread(imagePath)
        bs = BoxSelector(image,"Image")
        cv2.imshow("Image",image)
        cv2.waitKey(0)
        #order the points suitable for the Object detector
        pt1,pt2 = bs.roiPts
        (x,y,xb,yb) = [pt1[0],pt1[1],pt2[0],pt2[1]]
        annotations.append([int(x),int(y),int(xb),int(yb)])
        imPaths.append(imagePath)

    # save annotations and image paths to disk
    annotations = np.array(annotations)
    imPaths = np.array(imPaths, dtype="unicode")
    np.save(args["annotations"], annotations)
    np.save(args["images"], imPaths)



class ObjectDetector(object):
    def __init__(self,options=None,loadPath=None):
        #create detector options
        self.options = options
        if self.options is None:
            self.options = dlib.simple_object_detector_training_options()
        #load the trained detector (for testing)
        if loadPath is not None:
            self._detector = dlib.simple_object_detector(loadPath)


if __name__ == "__main__":
    main()

