import matplotlib.pyplot as plt
def draw(images):

    fig = plt.figure()
    for i in range(1,len(images)+1):
        fig.add_subplot(1, len(images), i )
        plt.imshow(images[i-1])
    plt.show()

import numpy as np
def padding(images):
    # Takes 28*28 images from EMNIST, returns 32*32 oadded image
    padded_images = np.zeros((images.shape[0],32,32))
    padded_images[:,2:-2,2:-2] = images
    return padded_images

def binarise(array):
    return np.where(array > 0.5*np.max(array), 1, 0)