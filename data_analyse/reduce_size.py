import os
import cv2
try:
    os.mkdir("/content/drive/MyDrive/all_img/_224/")
except:
    pass

for image in os.listdir("/content/drive/MyDrive/all_img/"):
    img = cv2.imread(f'/content/drive/MyDrive/all_img//{image}')
    #on veut transformer cette image en une image de traille 224x224
    img = cv2.resize(img, (224, 224))
    #on veut sauvegarder cette image dans /content/drive/MyDrive/all_img/_224
    cv2.imwrite("/content/drive/MyDrive/all_img/_224/" + image, img)