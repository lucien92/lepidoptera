#Dans ce fichier nous étudions la liste des labels qui se trouvent dans les dossiers data et nous les concaténant en enlevant les redondances

path1 = '/home/lucien/projet_lepinoc/data/data/lepinoc2022/classes.txt'
path2 = '/home/lucien/projet_lepinoc/data/data/lepinoc2022-2/classes.txt'
path3 = '/home/lucien/projet_lepinoc/data/data/lepinoc2022-3/classes.txt'
path4 = '/home/lucien/projet_lepinoc/data/data/lepinoc2022-4/classes.txt'

with open(path1) as f:
    lines1 = f.readlines()

with open(path2) as f:
    lines2 = f.readlines()

with open(path3) as f:
    lines3 = f.readlines()

with open(path4) as f:
    lines4 = f.readlines()

lines = lines1 + lines2 + lines3 + lines4
lines = list(set(lines))

print(lines)
print(len(lines))

all_labels = []
for label in lines:
    label = label.replace('\n', '')
    all_labels.append(label)

print(all_labels)