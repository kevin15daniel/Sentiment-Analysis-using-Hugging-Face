# DistilBERT Emotion Classification Model

## Overview

This project presents a fine-tuned **DistilBERT model** for emotion classification. The model has been trained to predict emotional states from text input and has achieved high performance on the evaluation set.

## Model Details

- **Model Type**: DistilBERT (fine-tuned version of `distilbert-base-uncased`)
- **Evaluation Results**:
  - **Loss**: 0.1448
  - **Accuracy**: 94.2%

This model was fine-tuned on an unknown dataset to predict emotions from text, making it a reliable choice for text classification tasks involving emotional analysis.

## Intended Use

This model is designed to classify emotions from text input. You can use it for applications like:
- Sentiment analysis
- Customer feedback analysis
- Social media monitoring
- Any other application where detecting emotion in text is useful

## Limitations

- The exact dataset used for fine-tuning is unknown, so the model's generalization capabilities may vary depending on the input.
- This model does not support multilingual text or specific domain-based adaptations.

## Training Procedure

The model was trained using the following setup:

- **Learning Rate**: 5e-05
- **Train Batch Size**: 64
- **Evaluation Batch Size**: 64
- **Epochs**: 2
- **Optimizer**: AdamW (with betas=(0.9, 0.999), epsilon=1e-08)
- **Scheduler**: Linear learning rate scheduler
- **Seed**: 42

### Training Results:
- **Epoch 1**: Validation Loss = 0.1857, Accuracy = 92.55%
- **Epoch 2**: Validation Loss = 0.1448, Accuracy = 94.2%

## Framework and Libraries

- **Transformers**: 4.48.3
- **Pytorch**: 2.5.1+cu124
- **Datasets**: 3.3.2
- **Tokenizers**: 0.21.0

## Usage

To use the model, you can load it with the following code:

```python
from transformers import pipeline

model = pipeline("text-classification", model="distilbert-emotion")
text = "I am feeling great today!"
prediction = model(text)

print(prediction)
```

This will return the predicted emotion for the given text.

## Conclusion

This **DistilBERT Emotion Classification Model** is a powerful tool for analyzing emotions in text. It’s fast, accurate, and can be easily integrated into various applications where emotion detection is required.
