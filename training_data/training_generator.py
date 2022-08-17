import numpy as np
from emnist import extract_training_samples
import utilities
from scipy.ndimage import gaussian_filter

from utilities import draw


images_digits, _ = extract_training_samples('digits')
images_letters, _ = extract_training_samples('letters')
images =np.concatenate((images_digits, images_letters))
np.random.shuffle(images)

# take 4 images:
select_images = images[:4]

# Padding
select_images = utilities.padding(select_images)


# Turn into binary
binary_selected_images = np.where(select_images > 0.5*np.max(select_images), 1, 0)

# Generate intensity image
intensity_images = binary_selected_images * np.random.uniform(low = 200, high = 800, size = (binary_selected_images.shape[0],1,1))		# Assign intensity values
gaussian_intensity_images = np.zeros(intensity_images.shape)							# Gaussian Blur intensity images
for i in range(len(intensity_images)):
	gaussian_intensity_images[i,:,:] = gaussian_filter(intensity_images[i,:,:], 0.1)

draw(gaussian_intensity_images)

# Generate lifetime image
lifetime = np.random.uniform(low = 0.3, high = 1.2)										# unit in ns
lifetime_images = np.where(gaussian_intensity_images != 0, lifetime, 0) * np.random.uniform(low = 0.3, high = 1.2, size = (gaussian_intensity_images.shape[0],1,1))
draw(lifetime_images)
