import numpy as np
from emnist import extract_training_samples
import utilities
from scipy.ndimage import gaussian_filter


images_digits, _ = extract_training_samples('digits')
images_letters, _ = extract_training_samples('letters')
images =np.concatenate((images_digits, images_letters))
np.random.shuffle(images)

# take 4 images:
select_images = images[:4]

# Padding
select_images = utilities.padding(select_images)

utilities.draw(select_images)

# Turn into binary
binary_selected_images = utilities.binarise(select_images)

# Generate intensity image
intensity_images = binary_selected_images * np.random.uniform(low=200, high=800)		# Assign intensity values
gaussian_intensity_images = np.zeros(intensity_images.shape)							# Gaussian Blur intensity images
for i in range(len(intensity_images)):
	gaussian_intensity_images[i,:,:] = gaussian_filter(intensity_images[i,:,:], 0.4)

