import numpy as np

from keras.models import Sequential 
from keras.layers import Dense, Conv2D, Flatten, Activation, MaxPooling2D
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
import time

X = np.load('X.npy')
y = np.load('y.npy')

# ----------------- TRY TO USE FORMATED STRING AT FILEPATH, THIS SHOUD BE EFFECTIVE AF -----------
#checkpoint = ModelCheckpoint('logs/best_model.h5', monitor='val_loss', verbose=0, save_best_only=True, 
#   save_weights_only=False, mode='min')

model = Sequential()

model.add(Conv2D(64, (3, 3), input_shape=X.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())

model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
    optimizer=Adam(),
    metrics=['accuracy'])

model.fit(X, y,
    batch_size=32,
    epochs=8,
    verbose=1,
    #callbacks=[checkpoint],
    validation_split=0.2)

model.save('my_model.h5')
