# AI_ML_Coders

A comprehensive collection of machine learning and artificial intelligence examples, tutorials, and implementations using TensorFlow, Keras, and PyTorch. This repository provides hands-on examples progressing from basic neural networks to advanced natural language processing and computer vision applications.

## 🚀 Quick Start

This project is organized into chapters that build upon each other, covering fundamental to advanced AI/ML concepts:

- **Chapter 1**: Basic Neural Networks - Linear regression with simple dense layers
- **Chapter 2**: Fashion MNIST Classification - Multi-class image classification with callbacks  
- **Chapter 3**: Computer Vision - CNNs, data augmentation, and binary image classification
- **Chapter 4**: TensorFlow Datasets - Working with TFDS for efficient data loading
- **Chapter 5**: Natural Language Processing - Text tokenization, preprocessing, and sentiment analysis

## 📚 Documentation

For comprehensive API documentation, usage examples, and detailed explanations of all functions and classes, see:

**[📖 Complete API Documentation](./API_DOCUMENTATION.md)**

## 🛠️ Key Features

- **Custom Dataset Processing**: `ProcessedDataset` class for language model fine-tuning
- **Training Callbacks**: Custom callbacks for training control and early stopping
- **Data Augmentation**: Advanced image preprocessing and augmentation techniques
- **NLP Pipeline**: Complete text preprocessing pipeline with tokenization and padding
- **Configuration Management**: YAML-based configuration for model training
- **Multiple ML Frameworks**: Examples using TensorFlow, Keras, and PyTorch

## 🏗️ Project Structure

```
AI_ML_Coders/
├── 📁 ChapterOne/          # Basic neural networks
├── 📁 ChapterTwo/          # Fashion MNIST + callbacks  
├── 📁 ChapterThree/        # Computer vision & CNNs
├── 📁 ChapterFour/         # TensorFlow Datasets
├── 📁 ChapterFive/         # Natural language processing
├── 📄 processed_dataset.py  # Core dataset processing class
├── 📄 custom_recipe.yaml   # Training configuration
└── 📄 API_DOCUMENTATION.md # Complete API documentation
```

## 🚀 Quick Examples

### Basic Neural Network
```python
# Simple linear regression
python ChapterOne/main.py
```

### Image Classification
```python  
# Fashion MNIST with custom callbacks
python ChapterTwo/callback.py
```

### Computer Vision
```python
# Binary image classification with data augmentation
python ChapterThree/Sec_1.py
```

### Natural Language Processing
```python
# Text tokenization and preprocessing
python ChapterFive/sec_1.py
```

## 📦 Installation

```bash
# Core dependencies
pip install tensorflow torch datasets transformers beautifulsoup4 tensorflow-datasets

# Optional for advanced training
pip install bitsandbytes torchtune
```

## 🎯 Use Cases

- **Learning AI/ML fundamentals** - Progressive examples from basic to advanced
- **Computer vision projects** - CNN architectures and image processing
- **NLP applications** - Text preprocessing and sentiment analysis  
- **Model fine-tuning** - Custom dataset classes and training configurations
- **Research and experimentation** - Modular components for rapid prototyping

## 📖 Learning Path

1. Start with **Chapter 1** for neural network basics
2. Progress through **Chapter 2** for classification concepts
3. Explore **Chapter 3** for computer vision techniques
4. Learn data handling in **Chapter 4** with TensorFlow Datasets
5. Master text processing in **Chapter 5** for NLP applications

For detailed explanations, parameter descriptions, and comprehensive examples, refer to the [API Documentation](./API_DOCUMENTATION.md).

## 🤝 Contributing

This is an educational repository designed to help developers learn AI/ML concepts through practical examples. Feel free to explore, modify, and extend the examples for your own learning journey.
