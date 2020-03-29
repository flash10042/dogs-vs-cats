import numpy as np
import os
import cv2
from tqdm import tqdm
import random

PATH = 'train'
IMG_SIZE = 90

training_data = []

for img in tqdm(os.listdir(PATH)):
    try:
        img_array = cv2.resize(cv2.imread(os.path.join(PATH, img), cv2.IMREAD_GRAYSCALE)/255, (IMG_SIZE, IMG_SIZE))
        if img.split('.')[0] == 'cat':
            training_data.append([img_array, 0])
        else:
            training_data.append([img_array, 1])
    except:
        pass

print(len(training_data))

random.shuffle(training_data)

X = []
y = []

for feature, label in training_data:
    X.append(feature)
    y.append(label)

X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)

np.save('X.npy', X)
np.save('y.npy', y)
