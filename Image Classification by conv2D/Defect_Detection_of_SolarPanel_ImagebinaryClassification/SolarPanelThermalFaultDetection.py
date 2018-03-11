# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 13:50:21 2018

@author: A664092
"""

from keras.preprocessing.image import ImageDataGenerator


def readData():
    train_datagen = ImageDataGenerator(rescale=1. / 255)
    test_datagen = ImageDataGenerator(rescale=1. / 255)

    train_dataset = train_datagen.flow_from_directory(
        'data/train',
        target_size=(64, 64),
        batch_size=16,
        class_mode='binary')

    validation_dataset = test_datagen.flow_from_directory(
        'data/test',
        target_size=(64, 64),
        batch_size=16,
        class_mode='binary')
    return train_dataset, validation_dataset


from keras.models import Sequential, load_model
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten


def createModel(train_dataset, validation_dataset):

    model = Sequential()

    # Step 1 - Convolution
    model.add(Conv2D(32, (3, 3), input_shape=(64, 64, 3), activation='relu'))
    # Step 2 - Pooling
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # Adding a second convolutional layer
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # Adding a third convolutional layer
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # Step 3 - Flattening
    model.add(Flatten())
    # Step 4 - Full connection
    model.add(Dense(units=128, activation='relu'))
    model.add(Dense(units=1, activation='sigmoid'))

    # Compiling the CNN
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit_generator(train_dataset,
                        steps_per_epoch=120,
                        epochs=10,
                        validation_data=validation_dataset,
                        validation_steps=17)

    return model


def saveModel(model):
    model.save("model.hf")


def loadModel():
    return load_model("model.hf")


def prediction(image_path):
    model = loadModel()
    import numpy as np
    from keras.preprocessing import image
    test_image = image.load_img(image_path, target_size=(64, 64))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)
    result = model.predict(test_image)
    # training_set.class_indices
    if result[0][0] == 1:
        print('No defect')
        return 'No Defect'
    else:
        print('Defect')
        return 'Defect'


#read data
train_dataset,validation_dataset = readData()
#creat model
model = createModel(train_dataset,validation_dataset)
#save model
saveModel(model)

prediction('image1.jpeg')


