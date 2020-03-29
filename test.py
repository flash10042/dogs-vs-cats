import numpy as np

from keras.models import Sequential, load_model
from keras.layers import Dense, Conv2D, Flatten, Activation, MaxPooling2D
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
import time
from tqdm import tqdm
import os
import cv2
import pandas as pd


model = load_model('my_model.h5')

PATH = 'test'
IMG_SIZE = 90

X_test = []

for img in tqdm(os.listdir(PATH)):
    img_array = cv2.resize(cv2.imread(os.path.join(PATH, img), cv2.IMREAD_GRAYSCALE)/255, (IMG_SIZE, IMG_SIZE))
    X_test.append(img_array)


X_test = np.array(X_test).reshape(-1, IMG_SIZE, IMG_SIZE, 1)

predictions = model.predict(X_test)

output = pd.DataFrame([[i+1, pred[0]] for i, pred in enumerate(predictions)], columns=['id', 'label'])
output.to_csv('submission.csv', index=False)
