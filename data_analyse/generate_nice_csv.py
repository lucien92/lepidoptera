#Dans ce script on génère un csv contenant les informations sur nos données organisées de la manière suivante: path to image, coordinate, label, image size

import os 

path_to_data_folder = '/home/lucien/projet_lepinoc/data/data'
folders = os.listdir(path_to_data_folder)

paths_to_annotations = [f'/home/lucien/projet_lepinoc/data/data/{folders[i]}/ann' for i in range(len(folders))]
print(paths_to_annotations)

#For the first folder of annotations and the corresponding labels

with open(f'/home/lucien/projet_lepinoc/data/data/{folders[0]}/classes.txt') as f:
    lines = f.readlines()
    labels = [line.replace('\n', '') for line in lines]

with open('/content/lepidoptera/data_analyse/annotations.csv', 'w') as f:
    path = paths_to_annotations[0]
    for file in os.listdir(path):
        with open(f'{path}/{file}') as f2:
            lines = f2.readlines()
            for line in lines:
                line = line.replace('\n', '')
                line = line.split(' ')
                line[0] = int(line[0])
                line = [line[1], line[2], line[3], line[4], labels[line[0]]]
                line = ','.join(line)
                line = f'{path}/{file[:-4]}.jpg,{line}'
                f.write(line + '\n')

#For the second folder of annotations and the corresponding labels

with open(f'/home/lucien/projet_lepinoc/data/data/{folders[1]}/classes.txt') as f:
    lines = f.readlines()
    labels = [line.replace('\n', '') for line in lines]

with open('/content/lepidoptera/data_analyse/annotations.csv', 'a') as f:
    path = paths_to_annotations[1]
    for file in os.listdir(path):
        with open(f'{path}/{file}') as f2:
            lines = f2.readlines()
            for line in lines:
                line = line.replace('\n', '')
                line = line.split(' ')
                line[0] = int(line[0])
                line = [line[1], line[2], line[3], line[4], labels[line[0]]]
                line = ','.join(line)
                line = f'{path}/{file[:-4]}.jpg,{line}'
                f.write(line + '\n')

#For the third folder of annotations and the corresponding labels

with open(f'/home/lucien/projet_lepinoc/data/data/{folders[2]}/classes.txt') as f:
    lines = f.readlines()
    labels = [line.replace('\n', '') for line in lines]

with open('/content/lepidoptera/data_analyse/annotations.csv', 'a') as f:
    path = paths_to_annotations[2]
    for file in os.listdir(path):
        with open(f'{path}/{file}') as f2:
            lines = f2.readlines()
            for line in lines:
                line = line.replace('\n', '')
                line = line.split(' ')
                line[0] = int(line[0])
                line = [line[1], line[2], line[3], line[4], labels[line[0]]]
                line = ','.join(line)
                line = f'{path}/{file[:-4]}.jpg,{line}'
                f.write(line + '\n')

#For the fourth folder of annotations and the corresponding labels

with open(f'/home/lucien/projet_lepinoc/data/data/{folders[3]}/classes.txt') as f:
    lines = f.readlines()
    labels = [line.replace('\n', '') for line in lines]

with open('/content/lepidoptera/data_analyse/annotations.csv', 'a') as f:
    path = paths_to_annotations[3]
    for file in os.listdir(path):
        with open(f'{path}/{file}') as f2:
            lines = f2.readlines()
            for line in lines:
                line = line.replace('\n', '')
                line = line.split(' ')
                line[0] = int(line[0])
                line = [line[1], line[2], line[3], line[4], labels[line[0]]]
                line = ','.join(line)
                line = f'{path}/{file[:-4]}.jpg,{line}'
                f.write(line + '\n')

#we want to write a final csv that merges all the annotations, and add the dimensions of the images at th eend of each line

#Vérifions si des images portent le même nom dans les différents dossiers

# import os

# path1, path2, path3, path4 = [f'/content/lepidoptera/data_analyse/annotations{i}.csv' for i in range(1, 5)]
# print(path1, path2, path3, path4)


path = '/content/lepidoptera/data_analyse/annotations.csv'
with open(path) as f:
    lines1 = f.readlines()
    lines1 = [line.split(',')[0] for line in lines1]
    #on veut regarder combien de photos différentes il y a 
    lines1 = set(lines1)
    print("Nombre d'annotations dans le csv", len(lines1))

#on veut compter le nombre de photos annotées
s = 0
for folder in paths_to_annotations:
    count = 0
    for file in os.listdir(folder):
        count += 1
    s += count
print("Nombre totale d'image annotée dans les data",s) #le nombre d'images annotées est 512, différent de 900 car toutes le simages de sont pas annotées



import cv2
def convert_bbox_format_reverse(bbox):
            # bbox est une liste ou un tuple de la forme [x_center, y_center, width, height]
            xmin = int(bbox[0] - 0.5 * bbox[2])
            ymin = int(bbox[1] - 0.5 * bbox[3])
            xmax = int(bbox[0] + 0.5 * bbox[2])
            ymax = int(bbox[1] + 0.5 * bbox[3])
            return xmin, ymin, xmax, ymax

with open('/content/lepidoptera/script/lepidoptere_detection/src/data/inputs/all_annotations.csv', 'w') as f:
        with open('/content/lepidoptera/data_analyse/annotations.csv', 'r') as f2:
            lines = f2.readlines()
            for line in lines:
                line = line.replace('\n', '')
                line = line.split(',')
                line = line[:5]
                specie = 'Anthophila'
                line[0] = line[0].replace('ann', 'img')
                try:
                    line[0] = line[0].replace('data/lepinoc2022-2/img','all_img')
                except:
                    pass
                try:
                    line[0] = line[0].replace('data/lepinoc2022/img','all_img')
                except:
                    pass
                try:
                    line[0] = line[0].replace('data/lepinoc2022-3/img','all_img')
                except:
                    pass
                try:
                    line[0] = line[0].replace('data/lepinoc2022-4/img','all_img')
                except:
                    pass
                img = cv2.imread(line[0])
                height, width, _ = img.shape
                line[1] = float(line[1])
                line[2] = float(line[2])
                line[3] = float(line[3])
                line[4] = float(line[4])
                line[1] = round(width*line[1],3)
                line[2] = round(height*line[2],3)
                line[3] = round(width*line[3],3) 
                line[4] = round(height*line[4],3) 
                line[1], line[2], line[3], line[4] = convert_bbox_format_reverse([line[1], line[2], line[3], line[4]])
                line[1] = str(line[1])
                line[2] = str(line[2])
                line[3] = str(line[3])
                line[4] = str(line[4])
                line = ','.join(line)
                line = f'{line},{specie},{width},{height}'
                f.write(line + '\n')
                