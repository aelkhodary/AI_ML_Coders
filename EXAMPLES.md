# AI_ML_Coders - Comprehensive Examples

This document provides complete, runnable examples for all major components in the AI_ML_Coders project.

## Table of Contents

1. [Basic Neural Network Examples](#basic-neural-network-examples)
2. [Image Classification Examples](#image-classification-examples)
3. [Computer Vision Examples](#computer-vision-examples)
4. [Natural Language Processing Examples](#natural-language-processing-examples)
5. [Custom Dataset Examples](#custom-dataset-examples)
6. [Callback Examples](#callback-examples)
7. [Configuration Examples](#configuration-examples)

## Basic Neural Network Examples

### Example 1: Simple Linear Regression

```python
"""
Basic neural network for learning linear relationships
Based on ChapterOne/main.py
"""
import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

def create_linear_model():
    """Create a simple linear regression model"""
    model = Sequential([
        Dense(units=1, input_shape=[1])
    ])
    model.compile(optimizer='sgd', loss='mean_squared_error')
    return model

def train_linear_model():
    """Train the model to learn y = 2x + 1"""
    # Create model
    model = create_linear_model()
    
    # Training data representing y = 2x + 1
    xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
    ys = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype=float)
    
    # Train the model
    print("Training linear regression model...")
    model.fit(xs, ys, epochs=1000, verbose=0)
    
    # Make predictions
    test_values = [10.0, 5.0, -2.0]
    for val in test_values:
        prediction = model.predict([val], verbose=0)
        print(f"Input: {val}, Predicted: {prediction[0][0]:.2f}, Expected: {2*val + 1}")
    
    # Show learned weights
    weights = model.layers[0].get_weights()
    print(f"Learned weights: {weights}")
    
    return model

if __name__ == "__main__":
    model = train_linear_model()
```

### Example 2: Multi-Input Neural Network

```python
"""
Extended neural network with multiple inputs
"""
import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

def create_multi_input_model():
    """Create a model with multiple inputs"""
    model = Sequential([
        Dense(64, activation='relu', input_shape=[3]),  # 3 input features
        Dense(32, activation='relu'),
        Dense(1)  # Single output
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

def generate_synthetic_data(n_samples=1000):
    """Generate synthetic data for multi-input regression"""
    np.random.seed(42)
    X = np.random.randn(n_samples, 3)
    # y = 2*x1 + 3*x2 - x3 + noise
    y = 2*X[:, 0] + 3*X[:, 1] - X[:, 2] + np.random.randn(n_samples) * 0.1
    return X, y

def train_multi_input_example():
    """Train multi-input neural network"""
    # Generate data
    X_train, y_train = generate_synthetic_data(800)
    X_test, y_test = generate_synthetic_data(200)
    
    # Create and train model
    model = create_multi_input_model()
    
    print("Training multi-input model...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=100,
        batch_size=32,
        verbose=0
    )
    
    # Evaluate
    loss, mae = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Loss: {loss:.4f}, Test MAE: {mae:.4f}")
    
    # Make sample predictions
    sample_inputs = np.array([[1.0, 2.0, 0.5], [-1.0, 1.0, 2.0]])
    predictions = model.predict(sample_inputs, verbose=0)
    
    for i, (inp, pred) in enumerate(zip(sample_inputs, predictions)):
        expected = 2*inp[0] + 3*inp[1] - inp[2]
        print(f"Input: {inp}, Predicted: {pred[0]:.2f}, Expected: {expected:.2f}")
    
    return model, history

if __name__ == "__main__":
    model, history = train_multi_input_example()
```

## Image Classification Examples

### Example 3: Fashion MNIST with Custom Architecture

```python
"""
Fashion MNIST classification with customizable architecture
Based on ChapterTwo/main.py
"""
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.models import Sequential

def load_fashion_mnist():
    """Load and preprocess Fashion MNIST data"""
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
    
    # Normalize pixel values
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    
    return (x_train, y_train), (x_test, y_test)

def create_fashion_model(hidden_units=128, dropout_rate=0.2):
    """Create Fashion MNIST classification model"""
    model = Sequential([
        Flatten(input_shape=(28, 28)),
        Dense(hidden_units, activation='relu'),
        Dropout(dropout_rate),
        Dense(64, activation='relu'),
        Dropout(dropout_rate),
        Dense(10, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def train_fashion_model():
    """Train Fashion MNIST classifier"""
    # Load data
    (x_train, y_train), (x_test, y_test) = load_fashion_mnist()
    
    # Create model
    model = create_fashion_model()
    
    print("Training Fashion MNIST classifier...")
    history = model.fit(
        x_train, y_train,
        validation_data=(x_test, y_test),
        epochs=10,
        batch_size=128,
        verbose=1
    )
    
    # Evaluate
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"Test accuracy: {test_acc:.4f}")
    
    # Class names for Fashion MNIST
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    
    # Make predictions on first 5 test samples
    predictions = model.predict(x_test[:5], verbose=0)
    
    for i in range(5):
        predicted_class = tf.argmax(predictions[i]).numpy()
        actual_class = y_test[i]
        confidence = predictions[i][predicted_class]
        
        print(f"Sample {i+1}:")
        print(f"  Predicted: {class_names[predicted_class]} (confidence: {confidence:.2f})")
        print(f"  Actual: {class_names[actual_class]}")
        print(f"  Correct: {'✓' if predicted_class == actual_class else '✗'}")
    
    return model, history

if __name__ == "__main__":
    model, history = train_fashion_model()
```

## Computer Vision Examples

### Example 4: CNN with Data Augmentation

```python
"""
Convolutional Neural Network with data augmentation
Based on ChapterThree examples
"""
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np

def create_cnn_model(input_shape=(28, 28, 1), num_classes=10):
    """Create a CNN model for image classification"""
    model = tf.keras.Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def create_data_generators():
    """Create data generators with augmentation"""
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    return train_datagen, test_datagen

def train_cnn_with_augmentation():
    """Train CNN with data augmentation on Fashion MNIST"""
    # Load data
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
    
    # Reshape for CNN (add channel dimension)
    x_train = x_train.reshape(-1, 28, 28, 1)
    x_test = x_test.reshape(-1, 28, 28, 1)
    
    # Create model
    model = create_cnn_model()
    
    # Create data generators
    train_datagen, test_datagen = create_data_generators()
    
    # Fit generators on data
    train_generator = train_datagen.flow(x_train, y_train, batch_size=32)
    test_generator = test_datagen.flow(x_test, y_test, batch_size=32)
    
    print("Training CNN with data augmentation...")
    history = model.fit(
        train_generator,
        steps_per_epoch=len(x_train) // 32,
        epochs=10,
        validation_data=test_generator,
        validation_steps=len(x_test) // 32,
        verbose=1
    )
    
    # Evaluate without augmentation
    test_loss, test_acc = model.evaluate(x_test/255.0, y_test, verbose=0)
    print(f"Test accuracy: {test_acc:.4f}")
    
    return model, history

class AccuracyCallback(tf.keras.callbacks.Callback):
    """Custom callback to stop training at target accuracy"""
    def __init__(self, target_accuracy=0.95):
        self.target_accuracy = target_accuracy
    
    def on_epoch_end(self, epoch, logs=None):
        if logs.get('accuracy', 0) > self.target_accuracy:
            print(f"\nReached {self.target_accuracy*100}% accuracy, stopping training!")
            self.model.stop_training = True

def train_with_callback():
    """Train CNN with custom callback"""
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
    
    x_train = x_train.reshape(-1, 28, 28, 1) / 255.0
    x_test = x_test.reshape(-1, 28, 28, 1) / 255.0
    
    model = create_cnn_model()
    callback = AccuracyCallback(target_accuracy=0.95)
    
    print("Training with accuracy callback...")
    history = model.fit(
        x_train, y_train,
        validation_data=(x_test, y_test),
        epochs=50,
        batch_size=128,
        callbacks=[callback],
        verbose=1
    )
    
    return model, history

if __name__ == "__main__":
    # Run CNN with augmentation
    model, history = train_cnn_with_augmentation()
    
    # Run with callback
    # model_cb, history_cb = train_with_callback()
```

### Example 5: Binary Image Classification

```python
"""
Binary image classification example
Based on ChapterThree/Sec_1.py concepts
"""
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np

def create_binary_cnn(input_shape=(150, 150, 3)):
    """Create CNN for binary classification"""
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=input_shape),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(1, activation='sigmoid')  # Binary output
    ])
    
    model.compile(
        loss='binary_crossentropy',
        optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.001),
        metrics=['accuracy']
    )
    
    return model

def create_binary_data_generator():
    """Create data generator for binary classification"""
    datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    return datagen

def simulate_binary_classification():
    """Simulate binary classification training"""
    # This example simulates the structure without actual image files
    print("Binary Classification CNN Example")
    print("="*40)
    
    model = create_binary_cnn()
    model.summary()
    
    # Simulate training data
    batch_size = 20
    img_height, img_width = 150, 150
    
    # Create dummy data for demonstration
    x_dummy = np.random.random((100, img_height, img_width, 3))
    y_dummy = np.random.randint(0, 2, (100, 1))
    
    print("\nTraining on dummy data...")
    history = model.fit(
        x_dummy, y_dummy,
        epochs=3,
        batch_size=batch_size,
        validation_split=0.2,
        verbose=1
    )
    
    # Simulate prediction
    test_image = np.random.random((1, img_height, img_width, 3))
    prediction = model.predict(test_image, verbose=0)
    
    print(f"\nPrediction: {prediction[0][0]:.4f}")
    print(f"Classification: {'Class 1' if prediction[0][0] > 0.5 else 'Class 0'}")
    
    return model, history

if __name__ == "__main__":
    model, history = simulate_binary_classification()
```

## Natural Language Processing Examples

### Example 6: Text Tokenization and Preprocessing

```python
"""
Comprehensive NLP preprocessing example
Based on ChapterFive examples
"""
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from bs4 import BeautifulSoup
import string
import re

def basic_tokenization_example():
    """Basic tokenization example from ChapterFive/sec_1.py"""
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
    
    # Create tokenizer
    tokenizer = Tokenizer(num_words=100, oov_token="<OOV>")
    tokenizer.fit_on_texts(sentences)
    
    # Get word index
    word_index = tokenizer.word_index
    print("Word Index:")
    for word, index in word_index.items():
        print(f"  {word}: {index}")
    
    # Convert to sequences
    sequences = tokenizer.texts_to_sequences(sentences)
    print(f"\nOriginal sequences: {sequences}")
    
    # Pad sequences
    padded = pad_sequences(sequences, padding='post')
    print(f"Padded sequences:\n{padded}")
    
    # Test on new data
    test_sequences = tokenizer.texts_to_sequences(test_data)
    print(f"\nTest sequences: {test_sequences}")
    
    return tokenizer, sequences, padded

def advanced_text_preprocessing(texts, max_features=10000, max_length=100):
    """Advanced text preprocessing pipeline"""
    
    def clean_text(text):
        """Clean individual text"""
        # Convert to lowercase
        text = text.lower()
        
        # Remove HTML tags
        soup = BeautifulSoup(text, 'html.parser')
        text = soup.get_text()
        
        # Add spaces around punctuation
        text = re.sub(r'([.!?])', r' \1 ', text)
        text = re.sub(r'[^a-zA-Z.!?]+', ' ', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    # Clean all texts
    cleaned_texts = [clean_text(text) for text in texts]
    
    # Create tokenizer
    tokenizer = Tokenizer(
        num_words=max_features,
        oov_token="<OOV>",
        filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'
    )
    
    # Fit tokenizer
    tokenizer.fit_on_texts(cleaned_texts)
    
    # Convert to sequences
    sequences = tokenizer.texts_to_sequences(cleaned_texts)
    
    # Pad sequences
    padded_sequences = pad_sequences(
        sequences,
        maxlen=max_length,
        padding='post',
        truncating='post'
    )
    
    return tokenizer, padded_sequences, cleaned_texts

def sentiment_preprocessing_example():
    """Example preprocessing for sentiment analysis"""
    sample_texts = [
        "I love this movie! It's absolutely amazing.",
        "This film is terrible. I hate it so much.",
        "The movie was okay, nothing special.",
        "<p>Great acting and <b>wonderful</b> story!</p>",
        "Boring and predictable. Would not recommend.",
        "Best movie ever!!! 😍😍😍"
    ]
    
    print("Sentiment Analysis Preprocessing Example")
    print("="*50)
    
    # Preprocess texts
    tokenizer, padded_sequences, cleaned_texts = advanced_text_preprocessing(
        sample_texts,
        max_features=1000,
        max_length=20
    )
    
    print("Original texts:")
    for i, text in enumerate(sample_texts):
        print(f"  {i+1}: {text}")
    
    print("\nCleaned texts:")
    for i, text in enumerate(cleaned_texts):
        print(f"  {i+1}: {text}")
    
    print(f"\nVocabulary size: {len(tokenizer.word_index)}")
    print(f"Padded sequences shape: {padded_sequences.shape}")
    
    print("\nPadded sequences:")
    for i, seq in enumerate(padded_sequences):
        print(f"  {i+1}: {seq}")
    
    return tokenizer, padded_sequences

if __name__ == "__main__":
    # Run basic example
    print("Basic Tokenization Example")
    print("="*30)
    basic_tokenization_example()
    
    print("\n" + "="*50 + "\n")
    
    # Run advanced example
    sentiment_preprocessing_example()
```

### Example 7: Complete NLP Pipeline

```python
"""
Complete NLP pipeline for text classification
"""
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

def create_text_classifier(vocab_size, embedding_dim, max_length, num_classes):
    """Create a text classification model"""
    model = Sequential([
        Embedding(vocab_size, embedding_dim, input_length=max_length),
        LSTM(64, dropout=0.5, recurrent_dropout=0.5),
        Dense(32, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax' if num_classes > 2 else 'sigmoid')
    ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy' if num_classes > 2 else 'binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def generate_sample_text_data():
    """Generate sample text data for classification"""
    positive_texts = [
        "I love this product, it's amazing!",
        "Great quality and fast shipping.",
        "Excellent customer service, highly recommended.",
        "This is the best purchase I've made.",
        "Outstanding value for money.",
    ]
    
    negative_texts = [
        "Terrible product, complete waste of money.",
        "Poor quality, broke after one day.",
        "Worst customer service ever experienced.",
        "Do not buy this, you'll regret it.",
        "Overpriced and underdelivered.",
    ]
    
    neutral_texts = [
        "The product is okay, nothing special.",
        "Average quality for the price.",
        "It works as expected.",
        "Neither good nor bad.",
        "Standard product, meets basic needs.",
    ]
    
    # Combine texts and labels
    texts = positive_texts + negative_texts + neutral_texts
    labels = [2] * len(positive_texts) + [0] * len(negative_texts) + [1] * len(neutral_texts)
    
    # Multiply to create more data
    texts = texts * 20  # 300 samples total
    labels = labels * 20
    
    return texts, labels

def train_text_classifier():
    """Train a complete text classification model"""
    print("Text Classification Pipeline Example")
    print("="*40)
    
    # Generate sample data
    texts, labels = generate_sample_text_data()
    
    # Configuration
    vocab_size = 1000
    embedding_dim = 50
    max_length = 20
    num_classes = 3  # positive, neutral, negative
    
    # Preprocess texts
    tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
    tokenizer.fit_on_texts(texts)
    
    sequences = tokenizer.texts_to_sequences(texts)
    padded_sequences = pad_sequences(sequences, maxlen=max_length, padding='post')
    
    # Convert to numpy arrays
    X = np.array(padded_sequences)
    y = np.array(labels)
    
    # Split data
    split_idx = int(0.8 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    print(f"Vocabulary size: {len(tokenizer.word_index)}")
    
    # Create and train model
    model = create_text_classifier(vocab_size, embedding_dim, max_length, num_classes)
    
    print("\nModel architecture:")
    model.summary()
    
    print("\nTraining model...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=10,
        batch_size=16,
        verbose=1
    )
    
    # Evaluate
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nTest accuracy: {test_acc:.4f}")
    
    # Test predictions
    test_texts = [
        "This product is absolutely fantastic!",
        "Terrible quality, very disappointed.",
        "It's an average product, nothing special."
    ]
    
    test_sequences = tokenizer.texts_to_sequences(test_texts)
    test_padded = pad_sequences(test_sequences, maxlen=max_length, padding='post')
    
    predictions = model.predict(test_padded, verbose=0)
    class_names = ['Negative', 'Neutral', 'Positive']
    
    print("\nSample predictions:")
    for text, pred in zip(test_texts, predictions):
        predicted_class = np.argmax(pred)
        confidence = pred[predicted_class]
        print(f"Text: '{text}'")
        print(f"Predicted: {class_names[predicted_class]} (confidence: {confidence:.3f})")
        print()
    
    return model, tokenizer, history

if __name__ == "__main__":
    model, tokenizer, history = train_text_classifier()
```

## Custom Dataset Examples

### Example 8: ProcessedDataset Usage

```python
"""
Example usage of the ProcessedDataset class
"""
from processed_dataset import ProcessedDataset, ChatMessage
import torch

def create_mock_tokenizer():
    """Create a mock tokenizer for demonstration"""
    class MockTokenizer:
        def __call__(self, sample):
            # Simulate tokenization
            messages = sample['messages']
            tokens = []
            
            for msg in messages:
                role_token = 1 if msg.role == "user" else 2
                content_tokens = [3, 4, 5, 6]  # Mock content tokens
                tokens.extend([role_token] + content_tokens)
            
            return {"tokens": tokens}
    
    return MockTokenizer()

def demonstrate_processed_dataset():
    """Demonstrate ProcessedDataset usage"""
    print("ProcessedDataset Example")
    print("="*25)
    
    # Note: This example assumes the dataset exists
    # In practice, you would need to create the dataset first
    try:
        # Create tokenizer
        tokenizer = create_mock_tokenizer()
        
        # Create dataset
        dataset = ProcessedDataset(tokenizer=tokenizer, packed=False)
        
        print(f"Dataset length: {len(dataset)}")
        
        # Get a sample
        sample = dataset[0]
        print(f"Sample keys: {sample.keys()}")
        print(f"Tokens: {sample['tokens']}")
        print(f"Labels: {sample['labels']}")
        
    except Exception as e:
        print(f"Note: Actual dataset not available - {e}")
        print("This example shows the expected usage pattern.")

def create_chat_message_examples():
    """Demonstrate ChatMessage usage"""
    print("\nChatMessage Examples")
    print("="*20)
    
    # User message example
    user_msg = ChatMessage(
        role="user",
        content=[
            {"type": "text", "content": "What is machine learning?"}
        ],
        ipython=False,
        eot=False,
        masked=False
    )
    
    print("User message:")
    print(f"  Role: {user_msg.role}")
    print(f"  Content: {user_msg.content}")
    print(f"  IPython: {user_msg.ipython}")
    print(f"  EOT: {user_msg.eot}")
    print(f"  Masked: {user_msg.masked}")
    
    # Assistant message example
    assistant_msg = ChatMessage(
        role="assistant",
        content=[
            {"type": "text", "content": "Machine learning is a subset of AI..."}
        ],
        ipython=False,
        eot=True,
        masked=False
    )
    
    print("\nAssistant message:")
    print(f"  Role: {assistant_msg.role}")
    print(f"  Content: {assistant_msg.content}")
    print(f"  EOT: {assistant_msg.eot}")
    
    # Multi-content message
    complex_msg = ChatMessage(
        role="user",
        content=[
            {"type": "text", "content": "Here's some code:"},
            {"type": "code", "content": "print('Hello, World!')"}
        ],
        ipython=True,
        eot=False,
        masked=False
    )
    
    print("\nComplex message:")
    print(f"  Role: {complex_msg.role}")
    print(f"  Content types: {[item['type'] for item in complex_msg.content]}")
    print(f"  IPython: {complex_msg.ipython}")
    
    return [user_msg, assistant_msg, complex_msg]

if __name__ == "__main__":
    demonstrate_processed_dataset()
    messages = create_chat_message_examples()
```

## Callback Examples

### Example 9: Custom Callback Collection

```python
"""
Collection of custom callback examples
"""
import tensorflow as tf
import numpy as np
import time

class EarlyStoppingCallback(tf.keras.callbacks.Callback):
    """Stop training when accuracy reaches target"""
    def __init__(self, target_accuracy=0.95, patience=3):
        super().__init__()
        self.target_accuracy = target_accuracy
        self.patience = patience
        self.wait = 0
        self.best_accuracy = 0
    
    def on_epoch_end(self, epoch, logs=None):
        current_accuracy = logs.get('accuracy', 0)
        
        if current_accuracy >= self.target_accuracy:
            print(f"\nReached target accuracy of {self.target_accuracy:.2%}!")
            self.model.stop_training = True
        
        if current_accuracy > self.best_accuracy:
            self.best_accuracy = current_accuracy
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                print(f"\nNo improvement for {self.patience} epochs, stopping...")
                self.model.stop_training = True

class LearningRateScheduler(tf.keras.callbacks.Callback):
    """Custom learning rate scheduler"""
    def __init__(self, schedule_func):
        super().__init__()
        self.schedule_func = schedule_func
    
    def on_epoch_begin(self, epoch, logs=None):
        new_lr = self.schedule_func(epoch)
        tf.keras.backend.set_value(self.model.optimizer.learning_rate, new_lr)
        print(f"Epoch {epoch+1}: Learning rate = {new_lr:.6f}")

class MetricsLogger(tf.keras.callbacks.Callback):
    """Log detailed metrics during training"""
    def __init__(self):
        super().__init__()
        self.epoch_times = []
        self.batch_times = []
    
    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start_time = time.time()
    
    def on_epoch_end(self, epoch, logs=None):
        epoch_time = time.time() - self.epoch_start_time
        self.epoch_times.append(epoch_time)
        
        print(f"\nEpoch {epoch+1} completed in {epoch_time:.2f}s")
        if logs:
            for metric, value in logs.items():
                print(f"  {metric}: {value:.4f}")
    
    def on_batch_begin(self, batch, logs=None):
        self.batch_start_time = time.time()
    
    def on_batch_end(self, batch, logs=None):
        batch_time = time.time() - self.batch_start_time
        self.batch_times.append(batch_time)

def lr_schedule(epoch):
    """Learning rate schedule function"""
    initial_lr = 0.001
    if epoch < 10:
        return initial_lr
    elif epoch < 20:
        return initial_lr * 0.5
    else:
        return initial_lr * 0.1

def demonstrate_callbacks():
    """Demonstrate custom callbacks"""
    print("Custom Callbacks Demonstration")
    print("="*30)
    
    # Load sample data
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    
    # Create simple model
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Create callbacks
    early_stopping = EarlyStoppingCallback(target_accuracy=0.90, patience=3)
    lr_scheduler = LearningRateScheduler(lr_schedule)
    metrics_logger = MetricsLogger()
    
    # Train with callbacks
    print("\nTraining with custom callbacks...")
    history = model.fit(
        x_train, y_train,
        validation_data=(x_test, y_test),
        epochs=25,
        batch_size=128,
        callbacks=[early_stopping, lr_scheduler, metrics_logger],
        verbose=0  # Suppress default output since we're using custom logger
    )
    
    # Print summary statistics
    print(f"\nTraining completed!")
    print(f"Total epochs: {len(history.history['loss'])}")
    print(f"Average epoch time: {np.mean(metrics_logger.epoch_times):.2f}s")
    print(f"Best validation accuracy: {max(history.history['val_accuracy']):.4f}")
    
    return model, history, metrics_logger

if __name__ == "__main__":
    model, history, logger = demonstrate_callbacks()
```

## Configuration Examples

### Example 10: YAML Configuration Usage

```python
"""
Example of working with YAML configuration
Based on custom_recipe.yaml
"""
import yaml
from dataclasses import dataclass
from typing import Optional, Dict, Any

@dataclass
class ModelConfig:
    """Model configuration dataclass"""
    component: str
    vocab_size: Optional[int] = None
    hidden_size: Optional[int] = None

@dataclass
class TokenizerConfig:
    """Tokenizer configuration dataclass"""
    component: str
    path: str
    max_seq_len: Optional[int] = None

@dataclass
class TrainingConfig:
    """Training configuration dataclass"""
    batch_size: int
    epochs: int
    learning_rate: float
    device: str
    dtype: str
    gradient_accumulation_steps: int = 1
    compile: bool = False

def load_config_from_yaml(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file"""
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        return config
    except FileNotFoundError:
        print(f"Config file {config_path} not found. Using default configuration.")
        return get_default_config()

def get_default_config() -> Dict[str, Any]:
    """Get default configuration"""
    return {
        'model': {
            '_component_': 'torchtune.models.llama3_2.llama3_2_1b'
        },
        'tokenizer': {
            '_component_': 'torchtune.models.llama3.llama3_tokenizer',
            'path': '/tmp/Llama-3.2-1B-Instruct/original/tokenizer.model',
            'max_seq_len': None
        },
        'dataset': {
            '_component_': 'processed_dataset.ProcessedDataset',
            'packed': False
        },
        'batch_size': 4,
        'epochs': 1,
        'optimizer': {
            '_component_': 'bitsandbytes.optim.PagedAdamW8bit',
            'lr': 2e-5
        },
        'device': 'cuda',
        'dtype': 'bf16',
        'gradient_accumulation_steps': 1,
        'compile': False
    }

def parse_training_config(config_dict: Dict[str, Any]) -> TrainingConfig:
    """Parse training configuration from dictionary"""
    return TrainingConfig(
        batch_size=config_dict.get('batch_size', 4),
        epochs=config_dict.get('epochs', 1),
        learning_rate=config_dict.get('optimizer', {}).get('lr', 2e-5),
        device=config_dict.get('device', 'cuda'),
        dtype=config_dict.get('dtype', 'bf16'),
        gradient_accumulation_steps=config_dict.get('gradient_accumulation_steps', 1),
        compile=config_dict.get('compile', False)
    )

def create_custom_config():
    """Create a custom configuration"""
    custom_config = {
        'output_dir': '/tmp/my_custom_training',
        'model': {
            '_component_': 'torchtune.models.llama3_2.llama3_2_1b'
        },
        'tokenizer': {
            '_component_': 'torchtune.models.llama3.llama3_tokenizer',
            'path': '/path/to/tokenizer.model',
            'max_seq_len': 2048
        },
        'dataset': {
            '_component_': 'processed_dataset.ProcessedDataset',
            'packed': True
        },
        'batch_size': 8,
        'epochs': 3,
        'optimizer': {
            '_component_': 'bitsandbytes.optim.PagedAdamW8bit',
            'lr': 1e-5
        },
        'loss': {
            '_component_': 'torchtune.modules.loss.CEWithChunkedOutputLoss'
        },
        'device': 'cuda',
        'dtype': 'bf16',
        'gradient_accumulation_steps': 2,
        'compile': True,
        'enable_activation_checkpointing': True,
        'log_every_n_steps': 10
    }
    
    return custom_config

def save_config_to_yaml(config: Dict[str, Any], output_path: str):
    """Save configuration to YAML file"""
    with open(output_path, 'w') as file:
        yaml.dump(config, file, default_flow_style=False, indent=2)
    print(f"Configuration saved to {output_path}")

def demonstrate_config_usage():
    """Demonstrate configuration usage"""
    print("Configuration Management Example")
    print("="*35)
    
    # Try to load existing config
    config = load_config_from_yaml('custom_recipe.yaml')
    
    print("Loaded configuration:")
    print(f"  Batch size: {config.get('batch_size')}")
    print(f"  Epochs: {config.get('epochs')}")
    print(f"  Device: {config.get('device')}")
    print(f"  Data type: {config.get('dtype')}")
    
    # Parse training config
    training_config = parse_training_config(config)
    print(f"\nParsed training configuration:")
    print(f"  {training_config}")
    
    # Create custom config
    custom_config = create_custom_config()
    print(f"\nCustom configuration created with batch size: {custom_config['batch_size']}")
    
    # Save custom config
    save_config_to_yaml(custom_config, 'my_custom_recipe.yaml')
    
    return config, training_config, custom_config

if __name__ == "__main__":
    config, training_config, custom_config = demonstrate_config_usage()
```

## Running the Examples

To run any of these examples:

1. **Save the example code** to a Python file (e.g., `example_1.py`)
2. **Install required dependencies**:
   ```bash
   pip install tensorflow torch datasets transformers beautifulsoup4 pyyaml
   ```
3. **Run the example**:
   ```bash
   python example_1.py
   ```

Each example is self-contained and demonstrates specific aspects of the AI_ML_Coders project. You can modify the parameters, architectures, and configurations to experiment with different approaches and learn how each component works.

These examples provide practical, runnable code that you can use as templates for your own machine learning projects.