import os 
import argparse
import pandas as pd
from PIL import Image
from tqdm import tqdm


source_path = '/home/lucien/projet_lepinoc/bees_detection/data_bees_detection/Anthophila'
target_path ='/home/lucien/projet_lepinoc/bees_detection/data_bees_detection/Cropped_Anthophila'


def crop_image(img_path, x, y, w, h):
    '''
    Crop image from x, y, w, h coordinates
    '''
    img = Image.open(img_path)
    img = img.crop((x, y, x+w, y+h))
    
    return img 

def crop_images(source_path, target_path):
    '''
    After having run predict.py, we have a csv file with the detected bees in the source_path.
    This function crops the images and saves them in the target_path.
    Creates a csv file with the cropped images so that it can be used to train a classifier.
    Format of the csv file:
    file_path, xmin, ymin, xmax, ymax, class_name
    '''

    # find the csv file in source path 
    for file in os.listdir(source_path):
        if file.endswith('.csv') and file.__contains__('detected'):
            csv_path = os.path.join(source_path, file)
            break

    # read the csv file
    df_detected = pd.read_csv(csv_path,names=['file_path','xmin','ymin','xmax','ymax','class_name','width','height'],sep=',',index_col=False)

    # create the target path if it doesn't exist
    if not os.path.exists(target_path):
        os.makedirs(target_path)

    # DataFrame to save the cropped images
    df_cropped = pd.DataFrame(columns=['file_path','class_name','old_path'])

    # crop and save images
    for index, row in tqdm(df_detected.iterrows()):

        # want to keep the subfolder structure (e.g. Anthophila/Apis mellifera/(database_provenance/)apis_mellifera_1.jpg)
        # into Cropped_Anthophila/Apis mellifera/(database_provenance/)apis_mellifera_1.jpg

        img_path = row['file_path']  #Anthophila/Apis mellifera/(database_provenance/)apis_mellifera_1.jpg
        sep = os.path.commonpath([source_path, img_path]) #Anthophila

        img_sub_path = img_path.split(sep)[1].split(os.sep)[1:] #[Apis mellifera,(database_provenance),apis_mellifera_1.jpg]
        img_sub_path = os.path.join(*img_sub_path) #Apis mellifera/(database_provenance/)apis_mellifera_1.jpg
        new_img_path = os.path.join(target_path, img_sub_path) #/home/lucien/projet_lepinoc/bees_detection/data_bees_detection/Cropped_Anthophila/Apis mellifera/(database_provenance/)apis_mellifera_1.jpg



        # create the subfolder if it doesn't exist
        subfolder_1 = os.path.join(target_path, img_sub_path.split(os.path.sep)[0])    # /home/lucien/projet_lepinoc/bees_detection/data_bees_detection/Cropped_Anthophila/Apis mellifera
        if not os.path.exists(subfolder_1):
            os.makedirs(subfolder_1)



        # create the subfolder if it doesn't exist
        subfolder_2 = os.path.join(subfolder_1, img_sub_path.split(os.path.sep)[1]) # /home/lucien/projet_lepinoc/bees_detection/data_bees_detection/Cropped_Anthophila/Apis mellifera/(database_provenance)
        is_folder=len(img_sub_path.split(os.path.sep))>2
        if is_folder and not os.path.exists(subfolder_2):
            os.makedirs(subfolder_2)

        # create the subfolder if it doesn't exist
        subfolder_3 = os.path.join(subfolder_2, img_sub_path.split(os.path.sep)[2]) # /home/lucien/projet_lepinoc/bees_detection/data_bees_detection/Cropped_Anthophila/Apis mellifera/(database_provenance)/apis_mellifera_1.jpg
        is_folder=len(img_sub_path.split(os.path.sep))>3
        if is_folder and not os.path.exists(subfolder_3):
            os.makedirs(subfolder_3)


        # converts the coordinates of the bounding box to integers (some may be negative,or floats)
        for coord in ['xmin', 'ymin', 'xmax', 'ymax']:
            if row[coord] < 0:
                row[coord] = 0
            row[coord] = int(row[coord])


        # crop the image
        img = crop_image(img_path, row['xmin'], row['ymin'], row['xmax']-row['xmin'], row['ymax']-row['ymin'])

        # save the image in the target path
        img.save(new_img_path)

        # save the cropped image in a new df
        if df_cropped.empty:
            df_cropped = pd.DataFrame({'file_path': [new_img_path],
                                        'class_name': [img_sub_path.split(os.path.sep)[0]],    #Apis mellifera
                                        'old_path': [img_path]})
        else:
            new_df=pd.DataFrame({'file_path': [new_img_path],
                                'class_name': [img_sub_path.split(os.path.sep)[0]],    #Apis mellifera
                                'old_path': [img_path]})
            df_cropped=pd.concat([df_cropped,new_df],ignore_index=True)


    # save the csv file
    df_cropped.to_csv(os.path.join(target_path, 'cropped_images.csv'), index=False)



if __name__ == '__main__':

    crop_images(source_path, target_path)