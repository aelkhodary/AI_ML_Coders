#ImageDataGenerator
import urllib.request
import zipfile
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def downloadTraningImages():
    url = "https://storage.googleapis.com/laurencemoroney-blog.appspot.com/horse-or-human.zip"
    file_name = 'horse-or-human.zip'
    training_dir = 'E:\delete\horse-or-human/training'
    # Download File with file name
    # urllib.request.urlretrieve(url, file_name)
    '''
    # unzip file
    zip_ref = zipfile.ZipFile(file_name, 'r')
    zip_ref.extractall(training_dir)
    zip_ref.close()
    '''
    return training_dir

def getBinaryImages(training_dir):
    # All images will be rescaled by 1./255
    # Image Augmentation(zoom in)
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    train_generator = train_datagen.flow_from_directory(
        training_dir,
        target_size=(300, 300),
        class_mode='binary'
    )
    return train_generator


def downloadTestingImages():
    validation_url = "https://storage.googleapis.com/laurencemoroney-blog.appspot.com/validation-horse-or-human.zip"
    validation_file_name = "validation-horse-or-human.zip"
    validation_dir = "E:\delete\horse-or-human/validation"
    urllib.request.urlretrieve(validation_url, validation_file_name)
    zip_ref = zipfile.ZipFile(validation_file_name, 'r')
    zip_ref.extractall(validation_dir)
    zip_ref.close()
    return validation_dir

def predictImages(model):
    #path = "E:\delete\horse-or-human/predict/horse-3.jpg"
    path = "E:\delete\horse-or-human/predict/human.webp"
    img = image.load_img(path, target_size=(300, 300))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    image_tensor = np.vstack([x])
    classes = model.predict(image_tensor)
    print("classfication is :{} ".format(classes))
    if classes[0] > 0.5:
        print(" Is a human")
    else:
        print(" Is a horse")





if __name__ == '__main__':
    print("-------------------Start Process-------- ")
    training_dir = downloadTraningImages()
    train_generator = getBinaryImages(training_dir)
    '''
     1-First Layer : We're defining 16 filters, each 3 × 3
     
     2-The input shape of the image is (300, 300, 3)
        because our input image is 300 × 300  and it’s in color, so there
        are three channels
        
     3-One neuron in the output layer as for we have binary classifier .
       the sigmoid function is to drive one set of values toward 0 and the other
       toward 1,
       which is perfect for binary classification.
       
     4- Several more convolutional layers. We do this because our
        image source is quite large, and we want, over time,
        to have many smaller images,each with features highlighted     
    '''
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(16, (3, 3), activation='relu',
                               input_shape=(300, 300, 3)),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.summary()

    model.compile(loss='binary_crossentropy',
                  optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.001),
                  metrics=['accuracy'])

    validation_dir = downloadTestingImages()
    validation_generator = getBinaryImages(validation_dir)

    # Train and validate your model
    history = model.fit(train_generator,
                        epochs=15,
                        validation_data=validation_generator)

    ## Predict the image
    predictImages(model)



    print("-------------------End Process-------- ")

