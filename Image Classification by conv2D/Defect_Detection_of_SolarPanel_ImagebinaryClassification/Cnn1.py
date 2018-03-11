# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.models import load_model

# # Initialising the CNN
# classifier = Sequential()
#
# # Step 1 - Convolution
# classifier.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))
# # Step 2 - Pooling
# classifier.add(MaxPooling2D(pool_size = (2, 2)))
#
# # Adding a second convolutional layer
# classifier.add(Conv2D(64, (3, 3), activation = 'relu'))
# classifier.add(MaxPooling2D(pool_size = (2, 2)))
#
# # Adding a third convolutional layer
# classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
# classifier.add(MaxPooling2D(pool_size = (2, 2)))
#
# # Step 3 - Flattening
# classifier.add(Flatten())
#
# # Step 4 - Full connection
# classifier.add(Dense(units = 128, activation = 'relu'))
# classifier.add(Dense(units = 1, activation = 'sigmoid'))
#
# # Compiling the CNN
# classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
#
#
# # Part 2 - Fitting the CNN to the images
#
# from keras.preprocessing.image import ImageDataGenerator
#
# train_datagen = ImageDataGenerator(rescale = 1./255,
#                                    shear_range = 0.2,
#                                    zoom_range = 0.2,
#                                    horizontal_flip = True)
#
# test_datagen = ImageDataGenerator(rescale = 1./255)
#
# training_set = train_datagen.flow_from_directory('C:/Users/A638054/PycharmProjects/SurfaceFaultDetectionCnn/dataset1/training_set',target_size = (64, 64),batch_size = 32,class_mode = 'binary')
# test_set = test_datagen.flow_from_directory('C:/Users/A638054/PycharmProjects/SurfaceFaultDetectionCnn/dataset1/test_set',
#                                             target_size = (64, 64),
#                                             batch_size = 32,
#                                             class_mode = 'binary')
# classifier.fit_generator(training_set,
#                          steps_per_epoch = 863,
#                          epochs = 100,
#                          validation_data = test_set,
#                          validation_steps = 30)
# classifier.save('SolarPanelThermalFaultDetection.h5')

def load_saved_model():
    return load_model('SolarPanelThermalFaultDetection.h5')



# Part 3 - Making new predictions

model = load_saved_model()
import numpy as np
from keras.preprocessing import image

image_path = 'image5.jpg'

test_image = image.load_img(image_path, target_size = (64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = model.predict(test_image)
print (result)

import cv2

#training_set.class_indices
if result[0][0] == 1:
    prediction = 'NoDefect'
    print('Nodefect')
    image = cv2.imread(image_path)
    cv2.putText(image, "No Defect predicted", (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    #cv2.waitKey(300)
    cv2.imshow("Predictions", image)
    cv2.waitKey(30000)
else:
    prediction = 'Defect'
    print('Defect')

    image = cv2.imread(image_path)
    cv2.putText(image, "Defect predicted", (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    #cv2.waitKey(300)
    cv2.imshow("Predictions", image)
    cv2.waitKey(30000)