import pandas as pd 
import numpy as np
from argparse import ArgumentParser
import json
import os 
import sklearn

argparser=ArgumentParser("Splits the desired csv file to 3 different files for training/validation/test")

argparser.add_argument(
    '-c',
    '--conf',
    default='/content/lepidoptera/script/lepidoptere_detection/src/config/lepido_detection.json',
    help='path to configuration file')

def _main_(args):

    config_path=args.conf

    with open(config_path) as config_buffer:
        config=json.loads(config_buffer.read())

    path_to_dataset="/content/lepidoptera/script/lepidoptere_detection/src/data/inputs/all_annotations.csv"

    with open(path_to_dataset, 'r') as f:
        lines = f.readlines()
        img_path = []
        for line in lines:
            line = line.replace('\n', '')
            line = line.split(',')
            img_path.append(line[0])
        img_path = set(img_path)
        print("Il y a 433 images différents dans le csv final", len(img_path))

    # proportion of the splits of the whole dataset

    proportion_train=0.8
    proportion_val=0.1
    
    count = 0
   
    df=pd.read_csv(path_to_dataset)
    print(df.head())
    limit_train = int(proportion_train*len(img_path))
    print("limit_train", limit_train)
    limit_validation = int(proportion_val*len(img_path))
    limit_test = len(img_path) - limit_train - limit_validation
    #on veut afficher une liste contenant les éléments de la première colonne
    print("il y a 433 images différentes dans le csv qu'on split", len(df.iloc[:,0].unique()))

    with open(path_to_dataset, 'r') as f:
        lines = f.readlines()
        with open("/content/lepidoptera/script/lepidoptere_detection/src/data/inputs/train.csv", 'w') as f2:
            for i, line in enumerate(lines):
                #on veut ajouter limit_train image différent au csv train.csv
                if count < limit_train:
                    line1 = lines[i].split(',')
                    line2 = lines[i+1].split(',')
                    
                    if line1[0] == line2[0]:
                        f2.write(lines[i])
                        a = lines[i]
                    else:
                        count += 1
                        f2.write(lines[i])
                        a = lines[i]
                
                else:
                    break
        count = 0
        with open("/content/lepidoptera/script/lepidoptere_detection/src/data/inputs/valid.csv", 'w') as f3:
                #on veut enlever tous les éléments avant a dans la liste lines
                j = lines.index(a)
                for i in range(j, len(lines)):
                    if count < limit_validation:
                        line1 = lines[i].split(',')
                        line2 = lines[i+1].split(',')
                        if line1[0] == line2[0]:
                            f3.write(lines[i])
                            b = lines[i]
                        else:
                            count += 1
                            f3.write(lines[i])
                            b = lines[i]
                    else:
                        break
        count = 0
        with open("/content/lepidoptera/script/lepidoptere_detection/src/data/inputs/test.csv", 'w') as f4:
                j = lines.index(b)
                for i in range(j, len(lines)):
                    if count < limit_test:
                        line1 = lines[i].split(',')
                        line2 = lines[i+1].split(',')
                        if line1[0] == line2[0]:
                            f4.write(lines[i])
                        else:
                            count += 1
                            f4.write(lines[i])
                    else:
                        break

        #vérifions s'il n'y pas une image qui est présente dans deux csv différents

        with open("/content/lepidoptera/script/lepidoptere_detection/src/data/inputs/train.csv", 'r') as f5:
            lines = f5.readlines()
            img_path = []
            for line in lines:
                line = line.replace('\n', '')
                line = line.split(',')
                img_path.append(line[0])
            img_path = set(img_path)

        with open("/content/lepidoptera/script/lepidoptere_detection/src/data/inputs/valid.csv", 'r') as f6:
            lines = f6.readlines()
            img_path2 = []
            for line in lines:
                line = line.replace('\n', '')
                line = line.split(',')
                img_path2.append(line[0])
            img_path2 = set(img_path2)

        with open("/content/lepidoptera/script/lepidoptere_detection/src/data/inputs/test.csv", 'r') as f7:
            lines = f7.readlines()
            img_path3 = []
            for line in lines:
                line = line.replace('\n', '')
                line = line.split(',')
                img_path3.append(line[0])
            img_path3 = set(img_path3)

        #regardons si les listes ont un élément en commun
        print("il y a", len(img_path.intersection(img_path2)), "images en commun entre train et valid")
        print("il y a", len(img_path.intersection(img_path3)), "images en commun entre train et test")



if __name__=='__main__':
    _args = argparser.parse_args()
    _main_(args=_args)
