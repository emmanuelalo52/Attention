
# Neural Network Components for Self-Attention Model

## Overview
This repository provides a custom implementation of several essential neural network components that are useful in building transformer-based models. The focus is on implementing Layer Normalization, Dropout, Linear, ReLU, Softmax, Self-Attention, Multi-Head Attention, and Position-Wise Feed Forward layers, using PyTorch. These modules are building blocks for attention mechanisms, widely used in transformer architectures for Natural Language Processing (NLP) tasks.

## Dependencies
- Python 3.7+
- PyTorch 1.8+
- NumPy

### Install Required Libraries:
```
pip install torch numpy
```

---

## Components

### 1. **Layer Normalization (`LayerNorm`)**
This class implements Layer Normalization, which normalizes inputs across the features dimension.

- **Attributes:**
  - `gamma`: Trainable scale parameter.
  - `beta`: Trainable shift parameter.
  - `eps`: Small epsilon value to avoid division by zero.

- **Usage:**
```python
layer_norm = LayerNorm(dim=128)
normalized_output = layer_norm(input_tensor)
```

### 2. **Dropout (`Dropout`)**
This class implements dropout regularization. During training, it randomly sets input units to zero with a probability `p`.

- **Attributes:**
  - `training`: Boolean indicating if the model is in training mode.
  - `p`: Dropout probability.

- **Usage:**
```python
dropout_layer = Dropout(training=True, p=0.1)
dropped_output = dropout_layer(input_tensor)
```

### 3. **Linear Layer (`Linear`)**
This implements a fully connected (linear) layer with optional bias.

- **Attributes:**
  - `weight`: Weight matrix initialized with a scaled random normal distribution.
  - `bias`: Optional bias vector.

- **Usage:**
```python
linear_layer = Linear(fan_in=128, fan_out=64, bias=True)
output = linear_layer(input_tensor)
```

### 4. **ReLU Activation (`ReLU`)**
A simple ReLU activation function that sets negative values to zero.

- **Usage:**
```python
relu = ReLU()
activated_output = relu(input_tensor)
```

### 5. **Softmax Activation (`Softmax`)**
This class implements the Softmax function for multi-class classification.

- **Usage:**
```python
softmax = Softmax()
softmax_output = softmax(input_tensor)
```

### 6. **Self-Attention (`SelfAttention`)**
Implements a scaled dot-product self-attention mechanism, which is a key component in transformer models.

- **Attributes:**
  - `query`, `key`, `value`: Linear layers to compute query, key, and value matrices.
  - `tril`: Lower triangular matrix to apply masking for causal attention.
  - `dropout`: Dropout applied after softmax.

- **Usage:**
```python
self_attention = SelfAttention(n_embed=128, head_size=64, block_size=128)
attention_output = self_attention(input_tensor)
```

### 7. **Multi-Head Attention (`MultiHeadAttention`)**
Combines multiple `SelfAttention` heads and projects their concatenated output using a linear layer.

- **Attributes:**
  - `n_heads`: List of `SelfAttention` layers.
  - `proj`: Linear layer to project concatenated heads.
  - `dropout`: Dropout applied after projection.

- **Usage:**
```python
multi_head_attn = MultiHeadAttention(num_heads=8, head_size=64, n_embed=128)
mha_output = multi_head_attn(input_tensor)
```

### 8. **Position-Wise Feed Forward Network (`PositionWiseForward`)**
Implements the Feed Forward Network (FFN) layer as used in transformers, which applies two linear transformations with a ReLU activation in between.

- **Usage:**
```python
ffn = PositionWiseForward(n_embed=128)
ffn_output = ffn(input_tensor)
```

---

## Example Usage
Here is an example of how to use the components together in a transformer-like architecture:
```python
import torch

# Sample Input
input_tensor = torch.randn(32, 128, 128)  # Batch size, Sequence length, Embedding size

# Initialize Layers
layer_norm = LayerNorm(dim=128)
dropout_layer = Dropout(training=True, p=0.1)
self_attention = SelfAttention(n_embed=128, head_size=64, block_size=128)
multi_head_attn = MultiHeadAttention(num_heads=8, head_size=64, n_embed=128)
ffn = PositionWiseForward(n_embed=128)

# Forward Pass
x = layer_norm(input_tensor)
x = dropout_layer(x)
x = self_attention(x)
x = multi_head_attn(x)
x = ffn(x)
```

---

## Notes
- This implementation assumes an understanding of transformer models and is tailored for experimentation and research purposes.
- It is recommended to run these modules on a GPU for performance reasons when working with large data or complex models.
- Error handling and additional features like weight initialization are kept minimal for clarity.

---

## Future Improvements
- Adding more initialization options for weights and biases.
- Enhancing numerical stability in the Softmax and attention mechanisms.
- Optimizing dropout implementation to leverage built-in PyTorch features more effectively.

## License
This project is open-source and available under the MIT License.
