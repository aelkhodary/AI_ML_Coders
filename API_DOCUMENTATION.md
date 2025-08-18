# AI_ML_Coders - Comprehensive API Documentation

## Table of Contents

1. [Overview](#overview)
2. [Core Dataset Module](#core-dataset-module)
3. [Chapter One - Basic Neural Networks](#chapter-one---basic-neural-networks)
4. [Chapter Two - Fashion MNIST Classification](#chapter-two---fashion-mnist-classification)
5. [Chapter Three - Computer Vision](#chapter-three---computer-vision)
6. [Chapter Four - TensorFlow Datasets](#chapter-four---tensorflow-datasets)
7. [Chapter Five - Natural Language Processing](#chapter-five---natural-language-processing)
8. [Configuration Files](#configuration-files)
9. [Usage Examples](#usage-examples)
10. [Installation and Setup](#installation-and-setup)

## Overview

This project contains a comprehensive collection of machine learning and AI examples, organized into chapters that progress from basic neural networks to advanced NLP processing. The codebase includes:

- Custom dataset processing for fine-tuning language models
- Computer vision applications with CNN architectures
- Natural language processing with tokenization and sentiment analysis
- TensorFlow/Keras implementations with callbacks and data generators
- Configuration files for model training and fine-tuning

## Core Dataset Module

### `processed_dataset.py`

This module provides core functionality for processing datasets for language model fine-tuning.

#### Classes

##### `ProcessedDataset(Dataset)`

A PyTorch Dataset class for loading and processing conversational data for language model training.

**Parameters:**
- `tokenizer` (optional): Tokenizer instance for encoding text
- `packed` (bool, default=False): Whether to pack sequences for efficiency

**Methods:**

###### `__init__(self, tokenizer=None, packed=False)`

Initialize the dataset with optional tokenizer and packing configuration.

**Example:**
```python
from processed_dataset import ProcessedDataset

# Initialize dataset
dataset = ProcessedDataset(tokenizer=my_tokenizer, packed=True)
```

###### `__getitem__(self, idx)`

Retrieve a single sample from the dataset.

**Parameters:**
- `idx` (int): Index of the sample to retrieve

**Returns:**
- `dict`: Dictionary containing 'tokens' and 'labels' keys

**Example:**
```python
sample = dataset[0]
print(sample['tokens'])  # Tokenized input
print(sample['labels'])  # Target labels
```

###### `__len__(self)`

Get the total number of samples in the dataset.

**Returns:**
- `int`: Number of samples

#### Named Tuples

##### `ChatMessage`

A named tuple representing a chat message with structured content.

**Fields:**
- `role` (str): Role of the message sender ("user" or "assistant")
- `content` (list): List of content dictionaries with "type" and "content" keys
- `ipython` (bool): Whether the message contains IPython code
- `eot` (bool): End of turn marker
- `masked` (bool): Whether the message should be masked during training

**Example:**
```python
from processed_dataset import ChatMessage

user_msg = ChatMessage(
    role="user",
    content=[{"type": "text", "content": "Hello, how are you?"}],
    ipython=False,
    eot=False,
    masked=False
)
```

## Chapter One - Basic Neural Networks

### `ChapterOne/main.py`

Demonstrates basic neural network training with TensorFlow/Keras for linear regression.

#### Functions

##### `main execution block`

Creates and trains a simple single-layer neural network to learn a linear relationship.

**Key Components:**
- Single Dense layer with 1 unit
- SGD optimizer
- Mean squared error loss
- Linear relationship learning (y = 2x + 1)

**Example Usage:**
```python
# Run the script directly
python ChapterOne/main.py
```

**Expected Output:**
- Prediction for input [10.0]
- Learned weights showing the linear relationship

## Chapter Two - Fashion MNIST Classification

### `ChapterTwo/main.py`

Implements a basic neural network for Fashion MNIST classification.

#### Key Features

- Fashion MNIST dataset loading and preprocessing
- 3-layer neural network architecture
- Adam optimizer with sparse categorical crossentropy loss
- Image normalization (pixel values / 255.0)

**Architecture:**
- Flatten layer (28x28 → 784)
- Dense layer (128 units, ReLU activation)
- Output layer (10 units, Softmax activation)

**Example Usage:**
```python
python ChapterTwo/main.py
```

### `ChapterTwo/callback.py`

Demonstrates custom callbacks for training control.

#### Classes

##### `myCallback(tf.keras.callbacks.Callback)`

Custom callback class that stops training when accuracy reaches 95%.

**Methods:**

###### `on_epoch_end(self, epoch, logs={})`

Called at the end of each epoch to check accuracy and potentially stop training.

**Parameters:**
- `epoch` (int): Current epoch number
- `logs` (dict): Dictionary containing training metrics

**Example:**
```python
from ChapterTwo.callback import myCallback

callbacks = myCallback()
model.fit(x_train, y_train, callbacks=[callbacks])
```

## Chapter Three - Computer Vision

### `ChapterThree/Sec_1.py`

Advanced image classification with data augmentation and binary classification.

#### Functions

##### `downloadTraningImages()`

Downloads and prepares training images for horse vs human classification.

**Returns:**
- `str`: Path to the training directory

**Example:**
```python
training_dir = downloadTraningImages()
```

##### `getBinaryImages(training_dir)`

Creates an ImageDataGenerator with augmentation for binary classification.

**Parameters:**
- `training_dir` (str): Path to training images directory

**Returns:**
- `DirectoryIterator`: Configured data generator

**Augmentation Features:**
- Rescaling (1/255)
- Rotation (40 degrees)
- Width/height shift (20%)
- Shear and zoom (20%)
- Horizontal flip

**Example:**
```python
train_generator = getBinaryImages(training_dir)
```

##### `downloadTestingImages()`

Downloads and prepares validation images.

**Returns:**
- `str`: Path to validation directory

##### `predictImages(model)`

Makes predictions on individual images using the trained model.

**Parameters:**
- `model`: Trained Keras model

**Example:**
```python
predictImages(trained_model)
```

### `ChapterThree/convolution.py`

Convolutional neural network implementation with Fashion MNIST.

#### Classes

##### `myCallback(tf.keras.callbacks.Callback)`

Custom callback that stops training at 99% accuracy.

**Methods:**

###### `on_epoch_end(self, epoch, logs={})`

Monitors accuracy and stops training when threshold is reached.

**CNN Architecture:**
- Conv2D layers with increasing filters (64, 64)
- MaxPooling2D layers for dimensionality reduction
- Flatten and Dense layers for classification

## Chapter Four - TensorFlow Datasets

### `ChapterFour/sec_1.py`

Demonstrates TensorFlow Datasets (TFDS) usage for data loading.

#### Key Features

- Loading Fashion MNIST using TFDS
- Dataset splitting (train/test)
- Data type inspection and structure analysis

**Example Usage:**
```python
import tensorflow_datasets as tfds

# Load training data
mnist_train = tfds.load(name="fashion_mnist", split="train")

# Inspect data structure
for item in mnist_train.take(1):
    print(item.keys())  # ['image', 'label']
```

## Chapter Five - Natural Language Processing

### `ChapterFive/sec_1.py`

Basic text tokenization and sequence processing.

#### Key Features

- Text tokenization with Keras Tokenizer
- Out-of-vocabulary token handling
- Sequence padding for uniform input length

**Example Data:**
```python
sentences = [
    'Today is a sunny day',
    'Today is a rainy day',
    'Is it sunny today?',
    'I really enjoyed walking in the snow today'
]
```

**Usage:**
```python
from tensorflow.keras.preprocessing.text import Tokenizer

tokenizer = Tokenizer(num_words=100, oov_token="<OOV>")
tokenizer.fit_on_texts(sentences)
sequences = tokenizer.texts_to_sequences(sentences)
```

### `ChapterFive/sec_5.py`

Advanced NLP preprocessing for emotion classification.

#### Key Features

- CSV data loading and preprocessing
- HTML tag removal with BeautifulSoup
- Stopword filtering
- Punctuation handling
- Train/test dataset splitting

#### Functions

##### Text Preprocessing Pipeline

1. **Data Loading**: Reads binary emotion CSV file
2. **Text Cleaning**: 
   - Lowercase conversion
   - Punctuation spacing
   - HTML tag removal
   - Stopword filtering
3. **Tokenization**: Creates sequences with vocabulary limits
4. **Padding**: Ensures uniform sequence length

**Configuration Parameters:**
- `vocab_size`: 20,000 words
- `embedding_dim`: 32 dimensions
- `max_length`: 10 tokens
- `training_size`: 28,000 samples

**Example:**
```python
# Preprocessing configuration
vocab_size = 20000
max_length = 10
oov_tok = "<OOV>"

tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(training_sentences)
```

## Configuration Files

### `custom_recipe.yaml`

Configuration file for Llama 3.2 model fine-tuning using TorchTune.

#### Key Sections

##### Model Configuration
```yaml
model:
  _component_: torchtune.models.llama3_2.llama3_2_1b
```

##### Tokenizer Configuration
```yaml
tokenizer:
  _component_: torchtune.models.llama3.llama3_tokenizer
  path: /tmp/Llama-3.2-1B-Instruct/original/tokenizer.model
  max_seq_len: null
```

##### Dataset Configuration
```yaml
dataset:
  _component_: processed_dataset.ProcessedDataset
  packed: False
```

##### Training Parameters
- **Batch Size**: 4
- **Epochs**: 1
- **Learning Rate**: 2e-5
- **Optimizer**: PagedAdamW8bit
- **Loss**: CEWithChunkedOutputLoss
- **Device**: CUDA
- **Precision**: bf16

##### Memory Management
- **Activation Checkpointing**: Disabled
- **Activation Offloading**: Disabled
- **Gradient Accumulation Steps**: 1

## Usage Examples

### Example 1: Basic Neural Network Training

```python
# ChapterOne example
import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

# Create simple model
model = Sequential([Dense(units=1, input_shape=[1])])
model.compile(optimizer='sgd', loss='mean_squared_error')

# Training data
xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
ys = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype=float)

# Train model
model.fit(xs, ys, epochs=1000)

# Make prediction
prediction = model.predict([10.0])
print(f"Prediction for 10.0: {prediction}")
```

### Example 2: Image Classification with Callbacks

```python
# ChapterTwo example with custom callback
import tensorflow as tf
from ChapterTwo.callback import myCallback

# Load and preprocess data
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# Create model
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Compile and train with callback
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
callbacks = myCallback()
model.fit(x_train, y_train, epochs=50, callbacks=[callbacks])
```

### Example 3: Computer Vision with Data Augmentation

```python
# ChapterThree example
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Create data generator with augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Load images from directory
train_generator = train_datagen.flow_from_directory(
    'path/to/training/data',
    target_size=(300, 300),
    class_mode='binary'
)
```

### Example 4: NLP Text Processing

```python
# ChapterFive example
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Sample texts
sentences = ['Today is sunny', 'Tomorrow will be rainy']

# Create and fit tokenizer
tokenizer = Tokenizer(num_words=1000, oov_token="<OOV>")
tokenizer.fit_on_texts(sentences)

# Convert to sequences and pad
sequences = tokenizer.texts_to_sequences(sentences)
padded = pad_sequences(sequences, maxlen=10, padding='post')

print(f"Word index: {tokenizer.word_index}")
print(f"Sequences: {sequences}")
print(f"Padded: {padded}")
```

### Example 5: Custom Dataset Usage

```python
# Using ProcessedDataset
from processed_dataset import ProcessedDataset, ChatMessage

# Create dataset instance
dataset = ProcessedDataset(tokenizer=my_tokenizer, packed=False)

# Access samples
sample = dataset[0]
print(f"Tokens: {sample['tokens']}")
print(f"Labels: {sample['labels']}")

# Create custom chat message
message = ChatMessage(
    role="user",
    content=[{"type": "text", "content": "Hello!"}],
    ipython=False,
    eot=False,
    masked=False
)
```

## Installation and Setup

### Requirements

```bash
# Core dependencies
pip install tensorflow
pip install torch
pip install datasets
pip install transformers
pip install beautifulsoup4
pip install tensorflow-datasets
```

### Optional Dependencies

```bash
# For advanced training
pip install bitsandbytes
pip install torchtune
```

### Directory Structure

```
AI_ML_Coders/
├── processed_dataset.py           # Core dataset processing
├── custom_recipe.yaml            # Training configuration
├── ChapterOne/
│   └── main.py                   # Basic neural networks
├── ChapterTwo/
│   ├── main.py                   # Fashion MNIST classification
│   └── callback.py               # Custom training callbacks
├── ChapterThree/
│   ├── Sec_1.py                  # Image classification with augmentation
│   ├── Sec_2.py                  # Additional CV examples
│   ├── Sec_3.py                  # Advanced CV techniques
│   ├── Sec_4.py                  # CV utilities
│   └── convolution.py            # CNN implementations
├── ChapterFour/
│   ├── sec_1.py                  # TensorFlow Datasets
│   ├── Sec_2.py                  # Advanced TFDS usage
│   ├── Sec_3.py                  # Dataset transformations
│   └── Sec_4.py                  # Custom dataset creation
└── ChapterFive/
    ├── sec_1.py                  # Basic NLP tokenization
    ├── sec_2.py                  # Text preprocessing
    ├── sec_3.py                  # Sequence modeling
    ├── sec_4.py                  # Advanced NLP techniques
    └── sec_5.py                  # Emotion classification
```

### Getting Started

1. **Clone the repository**
2. **Install dependencies** using the requirements above
3. **Run individual chapters** to explore different ML concepts
4. **Modify configurations** in `custom_recipe.yaml` for fine-tuning experiments
5. **Use the ProcessedDataset** class for custom data processing needs

### Common Use Cases

- **Learning ML/AI fundamentals**: Start with ChapterOne and progress through chapters
- **Computer Vision projects**: Use ChapterThree examples as templates
- **NLP applications**: Leverage ChapterFive preprocessing and tokenization
- **Model fine-tuning**: Utilize the ProcessedDataset and custom_recipe.yaml
- **Custom callbacks**: Extend the callback examples in ChapterTwo and ChapterThree

This documentation provides comprehensive coverage of all public APIs, functions, and components in the AI_ML_Coders project. Each section includes detailed descriptions, parameters, return values, and practical examples to help users understand and implement the functionality effectively.