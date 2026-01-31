# dAImond 💎

Deep Learnin фреймворк для Ruby, вдохновлённый PyTorch.

[![Gem Version](https://badge.fury.io/rb/daimond.svg)](https://rubygems.org/gems/daimond)
[![Ruby](https://img.shields.io/badge/ruby-%23CC342D.svg?style=for-the-badge&logo=ruby&logoColor=white)](https://www.ruby-lang.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **Почему Ruby?** ХЗ, захотелось. dAImond возвращает радость в возюкании с ML, потому что это Ruby. Поддерживает автоматическое дифференцирование и опциональный Rust backend для ускорения в 50-100 раз.

## Возможности

- 🔥 **Автоматическое дифференцирование** - Полноценный autograd с вычислительными графами
- 🧠 **Нейронные сети** - Линейные слои, активации (ReLU, Sigmoid, Softmax, Tanh)
- 📊 **Оптимизаторы** - SGD с моментумом, планирование learning rate
- 🎯 **Функции потерь** - MSE, CrossEntropy
- 💾 **Сериализация моделей** - Сохранение/загрузка через Marshal
- 📈 **Загрузчики данных** - Batch processing, шаффл, поддержка MNIST
- ⚡ **Быстрый бэкенд** - Numo::NArray для векторизованных операций (скорость C)
- 🎨 **Красивый API** - Идиоматичный Ruby DSL, чейнящиеся методы

## Установка

Добавьте в Gemfile:

```ruby
gem 'daimond'
```


Или установите ручками:
```ruby
gem install daimond
```

**Зависимости:** Ruby 2.7+, numo-narray

## Быстрый старт
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

## Пример MNIST (97% Accuracy!)
**Обучение классификатора на 60к рукописных цифрах:**
```ruby
ruby examples/mnist.rb
```
**Результаты:**
```text
Epoch 1/5: Loss = 0.2898, Accuracy = 91.35%
Epoch 2/5: Loss = 0.1638, Accuracy = 95.31%
Epoch 3/5: Loss = 0.1389, Accuracy = 96.2%
Epoch 4/5: Loss = 0.1195, Accuracy = 96.68%
Epoch 5/5: Loss = 0.1083, Accuracy = 97.12%
```

**Сохранение модели:**
```ruby
model.save('models/mnist_model.bin')
```

**Загрузка и предикт:**
```ruby
model = NeuralNet.new
model.load('models/mnist_model.bin')
prediction = model.forward(test_image)
```

## Производительность
| Backend   | MNIST (60k) Speed | Accuracy   |
| --------- | ----------------- | ---------- |
| Pure Ruby | ~30 min/epoch     | 97%        |
| Numo (C)  | ~3 min/epoch      | 97%        |
| **Rust**  | **~12 sec/epoch** | **89-98%** |

## Бэкенд на расте
For 50-100x speedup, compile Rust extensions:
```bash
cd ext/daimond_rust
cargo build --release
cd ../..
ruby examples/mnist_conv_rust.rb
```

## Roadmap
- [x] Ядро Autograd
- [x] Полносвязные и сверточные слои
- [x] MNIST 97%  (Adam)
- [x] Rust backend
- [x] Визуализация
- [ ] Batch Normalization & Dropout
- [ ] OpenCL/CUDA via FFI
- [ ] ONNX

## Помощь
Буду рад любой помощи! Инфа в CONTRIBUTING.md.

## Лицензия
MIT License