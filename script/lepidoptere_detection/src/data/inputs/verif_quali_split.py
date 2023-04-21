import csv
with open("/home/lucien/projet_lepinoc/script_lepinoc/script/lepidoptere_detection/src/data/inputs/all_annotations.csv", 'r') as f:
    #on veut lire toute sles lignes et compter le nombre d'images différentes
    lines = f.readlines()
    total_path = []
    for line in lines:
        line = line.split(',')
        path = line[0]
        total_path.append(path)
    total_path = set(total_path)
    print("nombre d'images différentes", len(total_path)) #le décalage avec le vrai nombre d'image dans le dossier est dû au fait que les images ne contenant pas de lepidoptere ne sont annotées
    
with open("/home/lucien/projet_lepinoc/script_lepinoc/script/lepidoptere_detection/src/data/inputs/all_annotations.csv", 'r') as f:
    #on veut lire toute sles lignes et compter le nombre d'images différentes
    lines = f.readlines()
    total_path = []
    for line in lines:
        line = line.split(',')
        path = line[0]
        total_path.append(path)
    total_path = set(total_path)
    print("nombre d'images différentes", len(total_path))