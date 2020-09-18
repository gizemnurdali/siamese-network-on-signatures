import os
import cv2
import csv
import numpy as np
from PIL import Image
import pandas as pd


def read_data(file_path):
    if os.path.exists(file_path):
        im_path = []
        p_number = []
        sign_number = []
        sign_images = []
        for path in sorted(os.listdir(file_path)):
            if path.endswith(".jpg") or path.endswith(".png"):
                read_image = cv2.imread(os.path.join(file_path, path))
                sign_images.append(read_image)
            im_path.append(str(path))
            p_number.append(str(path[4:7]))
            sign_number.append(str(path[7:9]))            
    else:
        print('There is no such file.\n')
    return im_path,p_number,sign_number,sign_images

def preprocess_image(images): 
    denoises = []
    for image in images:
        image =  resize_image(image)
        denoise = cv2.fastNlMeansDenoising(image)
        denoises.append(denoise)
    return denoises

def resize_image(image):
    im_pil = Image.fromarray(image)
    pil_img = im_pil.resize((105,105),2)
    image = np.asarray(pil_img) 
    return image

def add_new_folder(file_path,image_path,images):
    if os.path.exists(file_path):
        print(file_path,'Folder already exists.\nPlease check the size of images in existing folder.')
    else:
        os.mkdir(file_path)
        counter = 0
        for path in image_path:
            filename =file_path + '/' + str(path) 
            file = cv2.imwrite(filename,images[counter])
            counter += 1
        print('New file was created.Resized images were saved to this file.\n')

if __name__ == '__main__':
    
    file_path = 'genuines'
    image_path,person_number,sign_number,images =read_data(file_path)
    #Noise in images were reduced.
    #Signature images are different sizes.So, images were resized to 105x105 according to the cnn model input and saved new file.
    images = preprocess_image(images)
    new_file_path = 'originals'
    add_new_folder(new_file_path,image_path,images)

    #Dataframe was created to include signature file path, person number and person's signature number in its columns. Then, it was saved as a csv file.
    im_file_path = pd.DataFrame(data = image_path ,index=range(940), columns=['File Path'])
    p_number = pd.DataFrame(data = person_number ,index=range(940), columns=['Person Number'])
    s_number = pd.DataFrame(data = sign_number ,index=range(940), columns=['Signature Number'])
    data = pd.concat([im_file_path,p_number,s_number],axis=1)
    data.to_csv('Genuines.csv', encoding='utf-8', index=False)