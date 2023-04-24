import os
import cv2
#we want to take all the img of the folders /home/lucien/projet_lepinoc/data/data/img and put them in a same folder
new_path_to_img = '/content/drive/MyDrive/all_img/'

path_to_img = '/home/lucien/projet_lepinoc/data/data'
try:


    os.mkdir(new_path_to_img)
    for folder in os.listdir(path_to_img):
        for folder2 in os.listdir(f'{path_to_img}/{folder}'):
            if folder2 == 'img':
                for file in os.listdir(f'{path_to_img}/{folder}/{folder2}'):
                    img = cv2.imread(f'{path_to_img}/{folder}/{folder2}/{file}')
                    cv2.imwrite(f'{new_path_to_img}/{file}', img)

except:
    pass

#on veut vérifier qu'il n'existe pas 2 photos avec le même nom dans ce fichier

import os    
import cv2
import numpy as np

path_to_img = '/home/lucien/projet_lepinoc/data/data/all_img'
img_names = []
for file in os.listdir(path_to_img):
    img_names.append(file)
img_names = np.array(img_names)

if len(img_names) == len(np.unique(img_names)):
    print('ok')
