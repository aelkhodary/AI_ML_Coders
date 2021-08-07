""" Working with Real Data Sources """


import tensorflow as tf
import tensorflow_datasets as tfds

if __name__ == '__main__':
    imdb_sentences = []
    train_ds = tfds.load('imdb_reviews', split="train", with_info=True, as_supervised=True)
    train_data = tfds.as_numpy(train_ds)
    print(type(train_data))
    for item in train_data:
        print(type(item))
        print(item)
        #imdb_sentences.append(str(item['text']))