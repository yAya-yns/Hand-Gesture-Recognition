import os
import shutil
import random
import random

'''
HOW TO RUN:

1) Drag and drop acquisitions folder for dataset in the same file path
as this command

Should not need to run (I provide the images on our drive)
'''
people = os.listdir('./acquisitions')
if 'data' not in os.listdir():
    os.mkdir('data')

split_dirs = os.listdir('./data')
if 'train' not in split_dirs:
    os.mkdir('./data/train')
if 'test' not in split_dirs:
    os.mkdir('./data/test')
if 'validation' not in split_dirs:
    os.mkdir('./data/validation')

for person in people:

    gestures = os.listdir(f'./acquisitions/{person}')

    for gesture in gestures:
        images = os.listdir(f'./acquisitions/{person}/{gesture}')
        images = [image for image in images if 'png' in image]
        random.shuffle(images)

        num_images = len(images)
        training_idx = int(num_images*0.7)
        validation_idx = int(num_images*0.85)

        for image in images[:training_idx]:

            shutil.copy(f'./acquisitions/{person}/{gesture}/{image}', f'./data/train/{person}_{gesture}_{image}')

        for image in images[training_idx:validation_idx]:

            shutil.copy(f'./acquisitions/{person}/{gesture}/{image}', f'./data/validation/{person}_{gesture}_{image}')

        for image in images[validation_idx:]:

            shutil.copy(f'./acquisitions/{person}/{gesture}/{image}', f'./data/test/{person}_{gesture}_{image}')

