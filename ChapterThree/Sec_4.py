""" Dropout Regularization """
import tensorflow as tf

if __name__ == '__main__':
    print('Start ML')
    data = tf.keras.datasets.fashion_mnist
    (training_images, training_labels), (test_images, test_labels) = data.load_data()
    training_images = training_images / 255.0
    test_images = test_images / 255.0
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(256, activation=tf.nn.relu),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(128, activation=tf.nn.relu),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(64, activation=tf.nn.relu),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10, activation=tf.nn.softmax)
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    model.fit(training_images, training_labels, epochs=50)
    model.evaluate(test_images, test_labels)
    classifications = model.predict(test_images)
    print(classifications)
    print("Test Labels : {} ".format(test_labels))

    print(classifications[0])
    print("Test Labels : {} ".format(test_labels[0]))