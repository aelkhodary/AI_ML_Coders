# AI_ML_Coders - Quick Reference Guide

## 🔍 Function & Class Quick Lookup

### Core Dataset Module (`processed_dataset.py`)

| Component | Type | Description |
|-----------|------|-------------|
| `ProcessedDataset` | Class | PyTorch Dataset for conversational data processing |
| `ChatMessage` | NamedTuple | Structured chat message representation |

```python
# Quick usage
from processed_dataset import ProcessedDataset, ChatMessage
dataset = ProcessedDataset(tokenizer=tokenizer, packed=False)
sample = dataset[0]  # Returns {'tokens': [...], 'labels': [...]}
```

### Chapter One - Basic Neural Networks

| File | Key Components | Purpose |
|------|----------------|---------|
| `main.py` | Linear regression model | Learn y = 2x + 1 relationship |

```python
# Quick run
python ChapterOne/main.py
```

### Chapter Two - Fashion MNIST

| File | Key Components | Purpose |
|------|----------------|---------|
| `main.py` | 3-layer neural network | Fashion MNIST classification |
| `callback.py` | `myCallback` class | Stop training at 95% accuracy |

```python
# Quick usage
from ChapterTwo.callback import myCallback
callback = myCallback()  # Stops at 95% accuracy
```

### Chapter Three - Computer Vision

| File | Key Functions | Purpose |
|------|---------------|---------|
| `Sec_1.py` | `downloadTraningImages()`, `getBinaryImages()`, `predictImages()` | Horse vs Human classification |
| `convolution.py` | `myCallback` class | CNN with Fashion MNIST (99% accuracy stop) |

```python
# Quick usage
from ChapterThree.Sec_1 import downloadTraningImages, getBinaryImages
training_dir = downloadTraningImages()
train_gen = getBinaryImages(training_dir)
```

### Chapter Four - TensorFlow Datasets

| File | Key Features | Purpose |
|------|--------------|---------|
| `sec_1.py` | TFDS loading examples | Fashion MNIST with TFDS |

```python
# Quick usage
import tensorflow_datasets as tfds
mnist_train = tfds.load(name="fashion_mnist", split="train")
```

### Chapter Five - Natural Language Processing

| File | Key Features | Purpose |
|------|--------------|---------|
| `sec_1.py` | Basic tokenization | Text to sequences conversion |
| `sec_5.py` | Advanced preprocessing | Emotion classification pipeline |

```python
# Quick tokenization
from tensorflow.keras.preprocessing.text import Tokenizer
tokenizer = Tokenizer(num_words=100, oov_token="<OOV>")
```

## 🛠️ Common Patterns

### Creating a Custom Callback
```python
import tensorflow as tf

class CustomCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if logs.get('accuracy') > 0.95:
            self.model.stop_training = True
```

### Image Data Augmentation
```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    horizontal_flip=True
)
```

### Text Preprocessing Pipeline
```python
# 1. Load and clean text
# 2. Remove HTML tags with BeautifulSoup
# 3. Filter stopwords
# 4. Tokenize with Keras Tokenizer
# 5. Pad sequences for uniform length
```

### CNN Architecture Pattern
```python
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=(28,28,1)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])
```

## 📊 Model Architectures Summary

### Basic Neural Network (Chapter 1)
- **Input**: Single value
- **Architecture**: 1 Dense layer (1 unit)
- **Output**: Single prediction
- **Use Case**: Linear regression

### Fashion MNIST Classifier (Chapter 2)
- **Input**: 28x28 grayscale images
- **Architecture**: Flatten → Dense(128) → Dense(10)
- **Output**: 10 class probabilities
- **Use Case**: Multi-class classification

### CNN for Images (Chapter 3)
- **Input**: 300x300 RGB or 28x28 grayscale
- **Architecture**: Multiple Conv2D + MaxPool → Dense layers
- **Output**: Binary or multi-class classification
- **Use Case**: Computer vision tasks

### NLP Pipeline (Chapter 5)
- **Input**: Raw text strings
- **Processing**: Tokenization → Sequence conversion → Padding
- **Output**: Numerical sequences for model input
- **Use Case**: Text classification, sentiment analysis

## ⚙️ Configuration Quick Reference

### Training Configuration (`custom_recipe.yaml`)
```yaml
# Key parameters
batch_size: 4
epochs: 1
lr: 2e-5
vocab_size: 20000
max_length: 10
device: cuda
dtype: bf16
```

### Common Hyperparameters
- **Learning Rate**: 1e-3 to 1e-5
- **Batch Size**: 16, 32, 64 (adjust based on memory)
- **Epochs**: 10-100 (use callbacks for early stopping)
- **Optimizer**: Adam (general), SGD (simple), RMSprop (RNNs)

## 🚀 Quick Commands

```bash
# Run individual chapters
python ChapterOne/main.py
python ChapterTwo/callback.py
python ChapterThree/Sec_1.py
python ChapterFour/sec_1.py
python ChapterFive/sec_1.py

# Install dependencies
pip install tensorflow torch datasets transformers beautifulsoup4

# For advanced features
pip install bitsandbytes torchtune
```

## 📚 Related Files

- **[Complete API Documentation](./API_DOCUMENTATION.md)** - Detailed function descriptions and examples
- **[README.md](./README.md)** - Project overview and getting started guide
- **[custom_recipe.yaml](./custom_recipe.yaml)** - Training configuration template

This quick reference provides immediate access to the most commonly used functions, classes, and patterns in the AI_ML_Coders project.