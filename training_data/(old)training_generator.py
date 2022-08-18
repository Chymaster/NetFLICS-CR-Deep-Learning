import numpy as np
from emnist import extract_training_samples
import utilities
from scipy.ndimage import gaussian_filter

from utilities import draw


images_digits, _ = extract_training_samples('digits')
images_letters, _ = extract_training_samples('letters')
images =np.concatenate((images_digits, images_letters))
np.random.shuffle(images)

## Genearate a lot of random ordered lifetime/intensity images

# take 4 images:
select_images = images[:4]

# Padding
padded_images = np.zeros((select_images.shape[0],32,32))
padded_images[:,2:-2,2:-2] = select_images


# Turn into binary
binary_selected_images = np.where(padded_images > 0.5*np.max(padded_images), 1, 0)

##########################################################################################
# REPLACE, ROTATE AND FLIP IMAGE BEFORE SPLITTING THEM INTO INTENSITY AND LIFETIME IMAGE #
##########################################################################################
# Generate intensity image
intensity_images = binary_selected_images * np.random.uniform(low = 200, high = 800, size = (binary_selected_images.shape[0],1,1))		# Assign intensity values
gaussian_intensity_images = np.zeros(intensity_images.shape)							# Gaussian Blur intensity images
for i in range(len(intensity_images)):
	gaussian_intensity_images[i,:,:] = gaussian_filter(intensity_images[i,:,:], 0.1)

# Generate lifetime image
lifetime = np.random.uniform(low = 0.3, high = 1.2, size = (gaussian_intensity_images.shape[0],1,1))									# unit in ns
lifetime_images = np.where(gaussian_intensity_images != 0, 1, 0) * lifetime
#####################################################################################################################################################################

## Combine into new image
# Combine multiple
column = np.random.randint(1,high = 5)
row = np.random.randint(1,high = 5)
number_of_characters = column*row
pixel_per_character = 32

# Order to combine characters
order = np.array(range(number_of_characters-1))
np.random.shuffle(order)

# pick images to be combined
pass
# randomly flip/rotate every image
pass
# combine them into a single image

combined_image = np.zeros((column*size_per_character, row*size_per_character))

