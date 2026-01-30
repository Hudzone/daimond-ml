require_relative '../lib/daimond'



puts "Loading MNIST dataset..."
# Для начала возьмем подмножество (1000 примеров), чтобы быстро проверить
# Когда заработает, убери `.first(1000)`
train_images = Daimond::Data::MNIST.load_images('train-images-idx3-ubyte.gz').first(1000)
train_labels = Daimond::Data::MNIST.load_labels('train-labels-idx1-ubyte.gz').first(1000)

test_images = Daimond::Data::MNIST.load_images('t10k-images-idx3-ubyte.gz').first(200)
test_labels = Daimond::Data::MNIST.load_labels('t10k-labels-idx1-ubyte.gz').first(200)

puts "Train samples: #{train_images.length}, Test samples: #{test_images.length}"

# Создаем DataLoader
train_loader = Daimond::Data::DataLoader.new(train_images, train_labels, batch_size: 32, shuffle: true)

# Модель: 784 -> 128 -> 10
class MNISTNet < Daimond::NN::Module

  attr_reader :fc1, :fc2

  def initialize
    super()
    @fc1 = Daimond::NN::Linear.new(784, 128)
    @fc2 = Daimond::NN::Linear.new(128, 10)
    @parameters = @fc1.parameters + @fc2.parameters
  end

  def forward(x)
    # x: [batch, 784]
    h = @fc1.forward(x).relu
    logits = @fc2.forward(h)  # без softmax здесь, он будет в loss
    logits.softmax
  end
end

model = MNISTNet.new
optimizer = Daimond::Optim::SGD.new(model.parameters, lr: 0.1, momentum: 0.9)
criterion = Daimond::Loss::CrossEntropyLoss.new

puts "Training..."
epochs = 5

epochs.times do |epoch|
  total_loss = 0
  correct = 0
  total = 0
  batch_num = 0  # <--- Добавь счетчик батчей

  train_loader.each_batch do |x, y|
    # Forward
    pred = model.forward(x)
    loss = criterion.call(pred, y)

    total_loss += loss.data[0]

    # Accuracy (твой текущий код)
    batch_size = x.shape[0]
    batch_size.times do |i|
      predicted_class = pred.data[i, true].argmax
      correct += 1 if predicted_class == y.data[i]
      total += 1
    end

    # Backward
    optimizer.zero_grad
    loss.backward!

    # === ОТЛАДКА ЗДЕСЬ ===
    if batch_num == 0 && epoch == 0  # Только для первого батча первой эпохи
      grad_norm_fc2 = model.fc2.weight.grad.abs.mean
      grad_norm_fc1 = model.fc1.weight.grad.abs.mean
      puts "  [Debug] FC2 grad norm: #{grad_norm_fc2.round(6)}"
      puts "  [Debug] FC1 grad norm: #{grad_norm_fc1.round(6)}"
    end
    batch_num += 1
    # =====================

    optimizer.step
  end

  avg_loss = total_loss / train_loader.batches_count
  accuracy = 100.0 * correct / total
  puts "Epoch #{epoch + 1}/#{epochs}: Loss = #{avg_loss.round(4)}, Accuracy = #{accuracy.round(2)}%"
end

# Проверка весов
puts "Weight sample (should change from initial): #{model.fc1.weight.data[0, 0..4].to_a}"

# Тест на одном примере
puts "\nTesting on first test image..."
test_x = Daimond::Tensor.new([test_images[0]])
pred = model.forward(test_x)
predicted_digit = pred.data[0, true].argmax
puts "Predicted: #{predicted_digit}, Actual: #{test_labels[0]}"