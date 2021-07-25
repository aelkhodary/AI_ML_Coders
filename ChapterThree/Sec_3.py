""" Multiclass Classification """
import urllib.request
import zipfile
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def downloadTraningImages():
    url = "https://storage.googleapis.com/laurencemoroney-blog.appspot.com/rps.zip"
    file_name = 'rps.zip'
    training_dir = 'E:\delete\hand-gestures/training'
    """
    # Download File with file name
    urllib.request.urlretrieve(url, file_name)
    # unzip file
    zip_ref = zipfile.ZipFile(file_name, 'r')
    zip_ref.extractall(training_dir)
    zip_ref.close()
    """
    return training_dir


# Label images automatically
def getBinaryImages(training_dir, validation_dir):
    if validation_dir is None:
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
            target_size=(150, 150),
            class_mode='categorical'
        )
        return train_generator
    else:
        validation_datagen = ImageDataGenerator(rescale=1. / 255)

        train_generator = validation_datagen.flow_from_directory(
            validation_dir,
            target_size=(150, 150),
            class_mode='categorical'
        )
        return train_generator







def downloadTestingImages():
    url = "https://storage.googleapis.com/laurencemoroney-blog.appspot.com/rps-test-set.zip"
    file_name = 'rps-test-set.zip'
    validation_dir = 'E:\delete\hand-gestures/testing'
    """
    # Download File with file name
    urllib.request.urlretrieve(url, file_name)
    # unzip file
    zip_ref = zipfile.ZipFile(file_name, 'r')
    zip_ref.extractall(validation_dir)
    zip_ref.close()
    """
    return validation_dir


def predictImages(model):
    path = "E:\delete\hand-gestures/predict/paper.png"
    img = image.load_img(path, target_size=(150, 150))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    image_tensor = np.vstack([x])
    classes = model.predict(image_tensor, batch_size=10)
    print("classfication is :{} ".format(classes))



if __name__ == '__main__':
    print("-------------------Start Process-------- ")
    # Download dataset
    training_dir = downloadTraningImages()
    # Label images automatically
    train_generator = getBinaryImages(training_dir, None)
    model = tf.keras.models.Sequential([
        # Note the input shape is the desired size of the image:
        # 150x150 with 3 bytes color
        # This is the first convolution
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu',
                               input_shape=(150, 150, 3)),
        tf.keras.layers.MaxPooling2D(2, 2),
        # The second convolution
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        # The third convolution
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        # The fourth convolution
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        # Flatten the results to feed into a DNN
        tf.keras.layers.Flatten(),
        # 512 neuron hidden layer
        tf.keras.layers.Dense(512, activation='relu'),
        # 3 number of classes
        tf.keras.layers.Dense(3, activation='softmax')
    ])
    model.summary()
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop',
                  metrics=['accuracy'])

    validation_dir = downloadTestingImages()
    validation_generator = getBinaryImages(None, validation_dir)

    # Train and validate your model
    history = model.fit(train_generator,
                        epochs=1,
                        validation_data=validation_generator, verbose=1)
    model.save("rps.h5")
    # Predict the image
    predictImages(model)
    print("-------------------End Process-------- ")
