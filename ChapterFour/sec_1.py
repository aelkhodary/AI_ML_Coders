""" Getting Started with TFDS """
import tensorflow as tf
import tensorflow_datasets as tfds

if __name__ == '__main__':
    """
    mnist_data = tfds.load("fashion_mnist")
    for item in mnist_data:
        print(item)
    """
    mnist_train = tfds.load(name="fashion_mnist", split="train")
    assert isinstance(mnist_train, tf.data.Dataset)
    print(type(mnist_train))
    for item in mnist_train.take(1):
        print(type(item))
        print(item.keys())
        print(item['image'])
        print(item['label'])
    """
    mnist_test = tfds.load(name="fashion_mnist", split="test")
    assert isinstance(mnist_test, tf.data.Dataset)
    print(type(mnist_test))
    
    """
    mnist_test, info = tfds.load(name="fashion_mnist", with_info="true")
    print(info)

