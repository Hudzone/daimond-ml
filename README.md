# dAImond 💎

Deep Learning framework for Ruby, inspired by PyTorch. Written from scratch with love for the Ruby community. Features automatic differentiation, neural networks, and optional Rust backend for 50-100x speedup.

[![Gem Version](https://badge.fury.io/rb/daimond.svg)](https://rubygems.org/gems/daimond)
[![Ruby](https://img.shields.io/badge/ruby-%23CC342D.svg?style=for-the-badge&logo=ruby&logoColor=white)](https://www.ruby-lang.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **Why Ruby?** IDK, i just love this lang. dAImond brings back the happiness to ML.

## Features

- 🔥 **Automatic Differentiation** - Full autograd engine with computational graphs
- 🧠 **Neural Networks** - Linear layers, activations (ReLU, Sigmoid, Softmax, Tanh)
- 📊 **Optimizers** - SGD with momentum, learning rate scheduling
- 🎯 **Loss Functions** - MSE, CrossEntropy
- 💾 **Model Serialization** - Save/load trained models with Marshal
- 📈 **Data Loaders** - Batch processing, shuffling, MNIST support
- ⚡ **Fast Backend** - Numo::NArray for vectorized operations (C-speed)
- 🎨 **Beautiful API** - Idiomatic Ruby DSL, chainable methods

## Installation

Add this line to your Gemfile:

```ruby
gem 'daimond'
```

Or install manually:
```ruby
gem install daimond
```

**Dependencies:** Ruby 2.7+, numo-narray

## Quick Start
```ruby
require 'daimond'

# Define your model
class NeuralNet < Daimond::NN::Module
  attr_reader :fc1, :fc2
  
  def initialize
    super()
    @fc1 = Daimond::NN::Linear.new(784, 128)
    @fc2 = Daimond::NN::Linear.new(128, 10)
    @parameters = @fc1.parameters + @fc2.parameters
  end
  
  def forward(x)
    x = @fc1.forward(x).relu
    @fc2.forward(x).softmax
  end
end

# Training loop
model = NeuralNet.new
optimizer = Daimond::Optim::SGD.new(model.parameters, lr: 0.1, momentum: 0.9)
criterion = Daimond::Loss::CrossEntropyLoss.new

# Forward → Backward → Update
loss = criterion.call(model.forward(input), target)
optimizer.zero_grad
loss.backward!
optimizer.step
```

## MNIST Example (97% Accuracy!)
**Train a classifier on 60,000 handwritten digits:**
```ruby
ruby examples/mnist.rb
```
**Results:**
```text
Epoch 1/5: Loss = 0.2898, Accuracy = 91.35%
Epoch 2/5: Loss = 0.1638, Accuracy = 95.31%
Epoch 3/5: Loss = 0.1389, Accuracy = 96.2%
Epoch 4/5: Loss = 0.1195, Accuracy = 96.68%
Epoch 5/5: Loss = 0.1083, Accuracy = 97.12%
```

**Save your model:**
```ruby
model.save('models/mnist_model.bin')
```

**Load and predict:**
```ruby
model = NeuralNet.new
model.load('models/mnist_model.bin')
prediction = model.forward(test_image)
```

## Performance
| Backend   | MNIST (60k) Speed | Accuracy   |
| --------- | ----------------- | ---------- |
| Pure Ruby | ~30 min/epoch     | 97%        |
| Numo (C)  | ~3 min/epoch      | 97%        |
| **Rust**  | **~12 sec/epoch** | **89-98%** |

## With Rust Backend
For 50-100x speedup, compile Rust extensions:
```bash
cd ext/daimond_rust
cargo build --release
cd ../..
ruby examples/mnist_conv_rust.rb
```

## Roadmap
- [x] Core autograd engine
- [x] Linear layers & activations
- [x] MNIST 97% accuracy (Adam)
- [x] Conv2D + MaxPool layers
- [x] Rust backend
- [x] Training bosualization
- [ ] Batch Normalization & Dropout
- [ ] GPU support (OpenCL/CUDA via FFI)
- [ ] ONNX export/import

## Contributing
I'll be happy to see any contributors! Please read CONTRIBUTING.md for details.

## License
MIT License - see LICENSE file.