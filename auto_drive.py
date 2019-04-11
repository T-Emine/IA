from skimage.segmentation import mark_boundaries
from skimage.segmentation import slic
import matplotlib.pyplot as plt
import scipy.misc as im
from sklearn import svm
import seaborn as sns
import numpy as np

path_img = 'image2.jpg'

# charger une image
img = im.imread(path_img)

# utilisation de l'algorithme SLIC
superpixel_labels = slic(img, n_segments=1000, compactness=10)

# récupérer les canaux de couleur de l'image
red = img[:, :, 0]
green = img[:, :, 1]
blue = img[:, :, 2]

# recuperer le nombre de superpixels
nb_superpixels = np.max(superpixel_labels) + 1

# attribuer la couleur moyenne de chaque superpixel
# aux pixels lui appartenant
for label in range(nb_superpixels):
    idx = superpixel_labels == label
    red[idx] = np.mean(red[idx])
    green[idx] = np.mean(green[idx])
    blue[idx] = np.mean(blue[idx])

# dessiner les bordures des superpixels en noir et blanc
img = mark_boundaries(img, superpixel_labels, \
                      color=(1, 1, 1), outline_color=(0, 0, 0), \
                      mode='outer')


# afficher le résultat
plt.imshow(img)
plt.show()

#-----------------------------------------------------------------------------------------


# largeur et hauteur de l'image
width = np.shape(red)[1]
height = np.shape(red)[0]

# pour calculer la position du barycentre dans la largeur de l'image
x_idx = np.repeat(range(width), height)
x_idx = np.reshape(x_idx, [width, height])
x_idx = np.transpose(x_idx)

# pour calculer la position du barycentre dans la hauteur de l'image
y_idx = np.repeat(range(height), width)
y_idx = np.reshape(y_idx, [height, width])

# extraire les caractéristiques de chaque superpixel
feature_superpixels = []

for label in range(nb_superpixels):
    # pixels appartenant au superpixels
    idx = superpixel_labels == label

    # calcul et normalisation de la couleur moyenne
    c1_mean = np.mean(red[idx]) / 255
    c2_mean = np.mean(green[idx]) / 255
    c3_mean = np.mean(blue[idx]) / 255

    # calcul et normalisation de la position du barycentre
    x_mean = np.mean(x_idx[idx]) / (width - 1)
    y_mean = np.mean(y_idx[idx]) / (height - 1)

    # constitution du vecteur à 5 dimension
    sp = [c1_mean, c2_mean, c3_mean, x_mean, y_mean]
    feature_superpixels.append(sp)

# ----------------------------------------------------------------------------------------------------------------------

path_learning = 'img_learning2.png'
# lire l'image indiquant quels superpixels doivent être utilisés
# pour l'apprentissage
img_learning = im.imread(path_learning)

# récupérer les canaux de couleur de l'image
red_learning = img_learning[:, :, 0]
green_learning = img_learning[:, :, 1]
blue_learning = img_learning[:, :, 2]

# récupérer la couleur associée à chaque classe
class_colors = [(red_learning[y, x], green_learning[y, x], blue_learning[y, x]) for y in range(height) \
                for x in range(width)]
class_colors = set(class_colors)
class_colors.remove((0, 0, 0))
class_colors = list(class_colors)
# recupérer pour chaque classe tous les pixels qui lui sont attribués
class_pixels = []
for color in class_colors:
    learning_pixels = (red_learning == color[0]) \
                      & (green_learning == color[1]) \
                      & (blue_learning == color[2])
    class_pixels.append(learning_pixels)
# recupérer pour chaque classe quelques superpixels représentatifs
X = []  # caractéristiques des superpixels
Y = []  # identifiant de la classe
for label in range(nb_superpixels):
    # parcour l'ensemble des pixels du
    # superpixel et regarder combien
    # d'entre eux sont attribués à
    # chaque classe
    nb_for_each_class = []
    idx_sp = superpixel_labels == int(label)
    for learning_pixels in class_pixels:
        # print("----------------------------------------------")
        # print(idx_sp)
        # print(learning_pixels)
        # print("----------------------------------------------")

        common_idx = np.logical_and(learning_pixels, idx_sp)
        nb_for_each_class.append(np.sum(common_idx))
    # tester si le superpixel contient des pixels
    # appartenant à une et une seule classe
    class_idx = -1
    several_classes = 0
    for idx in range(len(nb_for_each_class)):
        if nb_for_each_class[idx] > 0:
            if class_idx < 0:
                # le superpixel contient
                # des pixels appartenant
                # à l'une des classes
                class_idx = idx
            else:
                # le superpixel contient
                # des pixels appartenant
                # à plusieurs classes :
                # ne pas le retenir comme
                # donnée d'apprentissage
                several_classes = True
    # si le superpixel a été retenu comme donnée
    # d'apprentissage, on stocker ses caractéristiques
    # et l'identifiant de la classe
    if (class_idx >= 0) and not several_classes:
        Y.append(class_idx)
        X.append(feature_superpixels[label])

# ----------------------------------------------------------------------------------
# creer le séparateur à vaste marge
model_svm = svm.SVC(decision_function_shape='ovo')
# paramètre du SVM permettant d'influencer la proportion
# de données d'apprentissage pouvant être considérées comme
# erronées
model_svm.C = 4.
# paramètre du noyau du SVM
model_svm.gamma = 4.
# indiquer que les probabilités d'appartenir à chaque classe
# doivent être calculées
model_svm.probability = True
# entraîner le SVM
model_svm.fit(X, Y)

# ------------------------------------------------------------------------------------
# predire la probabilité de chaque superpixel
# d'appartenir à chacune des classes
probas = model_svm.predict_proba(feature_superpixels)
# predire la classe la plus probable pour chaque superpixel
classification = model_svm.predict(feature_superpixels)

# -------------------------------------------------------------------------------------
# parcourir chacune des classes
for class_id in range(len(class_colors)):
    pixel_probas = np.zeros([height, width])
    # transférer la probabilité du superpixel
    # aux pixels qui le constituent
    for label in range(nb_superpixels):
        idx = superpixel_labels == label
        pixel_probas[idx] = probas[label, class_id]
    # afficher le résultat
    plt.figure(figsize=(16, 8))
    sns.heatmap(pixel_probas, xticklabels=False, yticklabels=False)
    plt.show()