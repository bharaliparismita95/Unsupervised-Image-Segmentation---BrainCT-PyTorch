import os
from datetime import datetime

import h5py
import numpy as np
from PIL import Image


def get_matching_file_names():
    images = os.listdir('training_images')
    filenames = []
    file_names = []

    for i in images:
        filenames.append(i)

    for file in filenames:
        image_path = os.path.join('training_images', file)
        file_names.append(image_path)

    print(file_names)

    return file_names


def get_raw_image(path, shape=None):
    img = Image.open(path)
    if shape:
        img = img.resize(shape, Image.BILINEAR)
    img = np.array(img)
    img = np.moveaxis(img, -1, 0)

    return img.astype(np.uint8)


def get_data_set(root: str, shape=None):
    file_names = get_matching_file_names()
    images = np.array([get_raw_image(image_path, shape) for image_path in file_names])

    return images


if __name__ == '__main__':
    train_dir = 'image_data'
    validation_split = 0.2
    image_shape = (192, 128)
    output_file = os.path.join('training_images', 'train-small.hdf5')

    print('Getting data...')
    train_images = get_data_set(train_dir, image_shape)
    print(train_images.shape)

    print('Generating training & validation sets...')
    N = int(len(train_images) * validation_split)
    val_images = train_images[:N]
    train_images = train_images[N:]

    print('Saving to file...')
    with h5py.File(output_file, 'w') as f:
        f.attrs['Date Created'] = str(datetime.now())
        f.attrs['Data Format'] = '(batch_size, channel, row, column)'
        f.attrs['Training Examples'] = len(train_images)
        f.attrs['Validation Examples'] = len(val_images)

        f.create_group('Training')
        f['Training'].create_dataset('Inputs', data=train_images, compression='gzip')

        f.create_group('Validation')
        f['Validation'].create_dataset('Inputs', data=val_images, compression='gzip')
