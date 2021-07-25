# Transfer Learning
import urllib.request
import zipfile
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.inception_v3 import InceptionV3


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


def getBinaryImages(training_dir, validation_dir):
    # All images will be rescaled by 1./255
    # Image Augmentation(zoom in)
    if validation_dir is None:
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
            batch_size=20,
            target_size=(150, 150),
            class_mode='binary')
    else:
        # Note that the validation data should not be augmented!
        train_datagen = ImageDataGenerator(rescale=1.0 / 255.)
        train_generator = train_datagen.flow_from_directory(
            validation_dir,
            batch_size=20,
            target_size=(150, 150),
            class_mode='binary')

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
    # path = "E:\delete\horse-or-human/predict/horse-3.jpg"
    # path = "E:\delete\horse-or-human/predict/human.webp"
    path = "E:\delete\horse-or-human/predict/human_2.jpg"
    img = image.load_img(path, target_size=(150, 150))
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
    train_generator = getBinaryImages(training_dir, None)

    weights_url = "https://storage.googleapis.com/mledu-datasets" \
                  "/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5 "

    weights_file = "inception_v3.h5"
    urllib.request.urlretrieve(weights_url, weights_file)
    pre_trained_model = InceptionV3(input_shape=(150, 150, 3),
                                    include_top=False,
                                    weights=None)
    pre_trained_model.load_weights(weights_file)

    # pre_trained_model.summary()

    # Freeze the entire network from retraining
    for layer in pre_trained_model.layers:
        layer.trainable = False

    last_layer = pre_trained_model.get_layer('mixed7')
    print('last layer output shape: ', last_layer.output_shape)
    last_output = last_layer.output

    # Flatten the output layer to 1 dimension
    x = tf.keras.layers.Flatten()(last_output)
    # Add a fully connected layer with 1,024 hidden units and ReLU activation
    x = tf.keras.layers.Dense(1024, activation='relu')(x)
    # Add a final sigmoid layer for classification
    x = tf.keras.layers.Dense(1, activation='sigmoid')(x)

    model = tf.keras.Model(pre_trained_model.input, x)

    model.compile(loss='binary_crossentropy',
                  optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.0001),
                  metrics=['accuracy'])

    validation_dir = downloadTestingImages()
    validation_generator = getBinaryImages(None, validation_dir)

    # Train and validate your model
    history = model.fit_generator(train_generator,
                                  epochs=20,
                                  validation_data=validation_generator,
                                  verbose=1)

    # Predict the image
    predictImages(model)

    print("-------------------End Process-------- ")
