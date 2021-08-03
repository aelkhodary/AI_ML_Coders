""" Introduction to Natural Language Processing """
# Understanding padding
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Getting Started with Tokenization
if __name__ == '__main__':
    sentences = [
        'Today is a sunny day',
        'Today is a rainy day',
        'Is it sunny today?',
        'I really enjoyed walking in the snow today'
    ]
    test_data = [
        'Today is a snowy day',
        'Will it be rainy tomorrow?'
    ]

    tokenizer = Tokenizer(num_words=100, oov_token="<OOV>")
    tokenizer.fit_on_texts(sentences)
    word_index = tokenizer.word_index
    # print(type(tokenizer))
    # print(type(word_index))
    print(word_index)
    sequences = tokenizer.texts_to_sequences(sentences)
    padded = pad_sequences(sequences, padding='post', maxlen=6, truncating='post')
    print(padded)
    # print(type(sequences))
    print(sequences)
    test_sequences = tokenizer.texts_to_sequences(test_data)
    print(test_sequences)



