import tensorflow as tf
import emnist
import numpy as np
import plotly.express as px

# load EMNIST dataset
train_images, train_labels = emnist.extract_training_samples('characters')
test_images, test_labels = emnist.extract_test_samples('characters')

# work in progress, see handwritten_digits_recognition.py for results

