# This is a functino that generate the ground truth of lifetime and intensity image, 
# Output in the format of intensity_image_array, lifetime_image_array
from emnist import extract_training_samples
import numpy as np
import tensorflow as tf
from utilities import draw

## Create shuffled, resized to 128*128, binarised EMNIST letters/digits

# Combine letters and digits and shuffle the order
digits, _ = extract_training_samples('digits')
letters, _ = extract_training_samples('letters')
characters =np.concatenate((digits, letters))
np.random.shuffle(characters)

# Resize to 128*128
characters = np.expand_dims(characters,axis=3)      # Add colour dimension because tf resize accepts dim (batch, x, y, colour_channel)
characters = tf.image.resize(characters,(128,128))  # resize all to 128,128
characters = np.squeeze(characters, axis=3)        # Squeeze dimension back to (batch, x, y)

# Binarise all
characters = np.where(characters > 0.5*np.max(characters), 1, 0)

## Generate combined and resized images
batch = 10
for i in range(batch):


# draw(characters[:4])