import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib as plt
from keras import Sequential
from keras.callbacks import ReduceLROnPlateau, CSVLogger
from keras.layers import Conv2D, BatchNormalization, MaxPool2D, Dropout, Flatten, Dense
from keras.utils import to_categorical
from keras_preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

y = train["label"]
x = train.drop(labels=["label"], axis=1)

x = x.values.reshape(-1, 28, 28, 1)
test = test.values.reshape(-1, 28, 28, 1)

y = to_categorical(y, num_classes=10)
x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.1, random_state=1)

data_generator = ImageDataGenerator(rescale=1. / 255, rotation_range=10, zoom_range=0.15, width_shift_range=0.1,
                                    height_shift_range=0.1, featurewise_center=False, horizontal_flip=False,
                                    vertical_flip=False)
data_generator.fit(x_train)

model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(5, 5), padding='Same', activation='relu', input_shape=(28, 28, 1)))
model.add(BatchNormalization())
model.add(Conv2D(filters=32, kernel_size=(5, 5), padding='Same', activation='relu'))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.5))
model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='Same', activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='Same', activation='relu'))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(256, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(10, activation="softmax"))
# model.summary()

# Using the adam optimizer and cross entropy loss
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# Using the sgd optimizer
# model.compile(optimizer="sgd", loss="categorical_crossentropy", metrics=["accuracy"])

# Using the RMSprop optimizer
# optimizer = RMSprop(lr=0.0001, rho=0.9, epsilon=1e-08, decay=0.00001)

learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', factor=0.5, verbose=1, patience=5, min_lr=0.0001)
epochs = 30
batch_size = 128

history = model.fit_generator(data_generator.flow(x_train, y_train, batch_size=batch_size),
                              epochs=epochs, validation_data=(x_val, y_val),
                              verbose=1, steps_per_epoch=x_train.shape[0] // batch_size,
                              callbacks=[learning_rate_reduction])

# predictions = model.predict_classes(test, verbose=1)
# pd.DataFrame({"ImageId": list(range(1, len(predictions) + 1)),
#               "Label": predictions}).to_csv("Kaggle_Submission.csv",
#                                             index=False,
#                                             header=True)

#
# , CSVLogger("Training Logs.csv",
#                                                                             append=False,
#                                                                             separator=";")