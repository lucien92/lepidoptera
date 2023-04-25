# import cv2 as cv
# import numpy as np
# from matplotlib import pyplot as plt
import os
try:
    os.mkdir('/home/lucien/projet_lepinoc/data/test_annotated')
except:
    pass

# for img in os.listdir('/content/drive/MyDrive/all_img//'):
#     imge = cv.imread(f'/content/drive/MyDrive/all_img//{img}', cv.IMREAD_GRAYSCALE)

#     assert imge is not None, "file could not be read, check with os.path.exists()"
#     imge = cv.medianBlur(imge,5)
#     image = th2 = cv.adaptiveThreshold(imge,255,cv.ADAPTIVE_THRESH_MEAN_C,\
#                 cv.THRESH_BINARY,11,2)
#     #image = cv.adaptiveThreshold(img,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,\
#                 #cv.THRESH_BINARY,11,2)
#     #on veut retenir les tâches noires (pixel de 255) qui ne sont pas trop grands (aire < 1000) et ls encadrer avec un rectangle

#     #on commence par trouver les contours

#     contours, hierarchy = cv.findContours(image, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)


#     #on veut maintenant trouver les contours qui n'ont pas une largeur ou une longueur supéreure à 200

#     for cnt in contours:
#         area = cv.contourArea(cnt)
#         x,y,w,h = cv.boundingRect(cnt)
#         if area > 1500 and w<200 and h<200:
#             cv.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

#     cv.imwrite(f'/home/lucien/projet_lepinoc/data/test_annotated/{img}', image)

import cv2 as cv
import numpy as np
import os

for img in os.listdir('/content/drive/MyDrive/all_img//'):
    # Load image in grayscale
    imge = cv.imread(f'/content/drive/MyDrive/all_img//{img}', cv.IMREAD_GRAYSCALE)

    assert imge is not None, "file could not be read, check with os.path.exists()"
    imge = cv.medianBlur(imge,5)

    # Apply Canny filter to detect edges
    edges = cv.Canny(imge, 50, 150)

    # Apply threshold to separate butterfly features from the background
    ret,thresh = cv.threshold(edges,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)

    # Find contours of butterfly features
    contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    # Filter contours to remove those that are too small or too large
    for cnt in contours:
        area = cv.contourArea(cnt)
        x,y,w,h = cv.boundingRect(cnt)
        if area > 100 and area < 10000 and w < 300 and h < 300:
            # Draw bounding box around butterfly feature
            cv.rectangle(imge, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Save annotated image
    cv.imwrite(f'/home/lucien/projet_lepinoc/data/test_annotated/{img}', imge)

