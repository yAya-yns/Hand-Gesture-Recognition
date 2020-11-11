import os
import shutil
import random
import random
"""
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
"""
proj_path = "./"
people = os.listdir(os.path.join(proj_path, 'acquisitions'))
if 'data' not in os.listdir(proj_path):
  os.mkdir(os.path.join(proj_path, 'data'))

split_dir = os.listdir(os.path.join(proj_path, 'data'))
if 'train' not in split_dir:
    os.mkdir(os.path.join(proj_path, 'data/train'))
if 'test' not in split_dir:
    os.mkdir(os.path.join(proj_path, 'data/test'))
if 'validation' not in split_dir:
    os.mkdir(os.path.join(proj_path, 'data/validation'))

for person in people:
  if person == 'desktop.ini':
    continue
  gestures = os.listdir(os.path.join(proj_path, f'acquisitions/{person}'))
  for gesture in gestures:
    if gesture == 'desktop.ini':
      continue
    if f'{gesture}' not in os.listdir(os.path.join(proj_path, 'data/validation')):
      os.mkdir(os.path.join(proj_path, f'data/validation/{gesture}'))
    if f'{gesture}' not in os.listdir(os.path.join(proj_path, 'data/train')):
      os.mkdir(os.path.join(proj_path, f'data/train/{gesture}'))
    if f'{gesture}' not in os.listdir(os.path.join(proj_path, 'data/test')):
      os.mkdir(os.path.join(proj_path, f'data/test/{gesture}'))

    images = os.listdir(os.path.join(proj_path, f'acquisitions/{person}/{gesture}'))
    images.remove('desktop.ini')
    random.shuffle(images)

    num_images = len(images)
    training_idx = int(num_images*0.7)
    validation_idx = int(num_images*0.85)

    for image in images[:training_idx]:
       source_path = os.path.join(proj_path, f'acquisitions/{person}/{gesture}/{image}')
       dest_path = os.path.join(proj_path, f'data/train/{gesture}/{person}_{image}')
       shutil.copy(source_path, dest_path)

    for image in images[training_idx:validation_idx]:
       source_path = os.path.join(proj_path, f'acquisitions/{person}/{gesture}/{image}')
       dest_path = os.path.join(proj_path, f'data/validation/{gesture}/{person}_{image}')
       shutil.copy(source_path, dest_path)

    for image in images[validation_idx:]:
       source_path = os.path.join(proj_path, f'acquisitions/{person}/{gesture}/{image}')
       dest_path = os.path.join(proj_path, f'data/test/{gesture}/{person}_{image}')
       shutil.copy(source_path, dest_path)
