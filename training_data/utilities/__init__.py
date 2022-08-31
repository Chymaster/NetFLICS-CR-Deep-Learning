import matplotlib.pyplot as plt
def draw(images):

    combined_image = np.reshape(images, (-1, images.shape[-1]))
    plt.imshow(combined_image)
    plt.colorbar()
    plt.show()

import numpy as np
def padding(images):
    # Takes 28*28 images from EMNIST, returns 32*32 oadded image
    padded_images = np.zeros((images.shape[0],32,32))
    padded_images[:,2:-2,2:-2] = images
    return padded_images

def binarise(array):
    return np.where(array > 0.5*np.max(array), 1, 0)

def read_ground_truth(filename):
    import h5py
    with h5py.File('ground_truth.h5','r') as f:
        gt_intensity = f['intensity_ground_truth']
        gt_lifetime = f['lifetime_ground_truth']
    return gt_intensity, gt_lifetime