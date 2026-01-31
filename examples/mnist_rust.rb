require_relative '../lib/daimond'

puts "=== Daimond MNIST with Rust Backend ==="
puts "Rust available: #{defined?(Daimond::Rust) ? '✅ YES' : '❌ NO'}"

# Проверка что Rust реально работает
if defined?(Daimond::Rust)
  test_tensor = Daimond::Rust::Tensor.zeros(5, 5)
  puts "Rust tensor shape: #{test_tensor.shape}"
  puts "✅ Rust backend loaded successfully!\n\n"
else
  puts "⚠️  Falling back to pure Ruby\n\n"
end

puts "Loading MNIST dataset..."
# Сначала пробуем на 10000, потом можно убрать .first(10000)
train_images = Daimond::Data::MNIST.load_images('train-images-idx3-ubyte.gz')
train_labels = Daimond::Data::MNIST.load_labels('train-labels-idx1-ubyte.gz')

sample_size = 10000  # Можно увеличить до 60000 если 10000 работает быстро
train_images = train_images.first(sample_size)
train_labels = train_labels.first(sample_size)

puts "Using #{sample_size} samples"

test_images = Daimond::Data::MNIST.load_images('t10k-images-idx3-ubyte.gz')
test_labels = Daimond::Data::MNIST.load_labels('t10k-labels-idx1-ubyte.gz')

puts "Train: #{train_images.length}, Test: #{test_images.length}"

train_loader = Daimond::Data::DataLoader.new(train_images, train_labels, batch_size: 32, shuffle: true)

class MNISTNet < Daimond::NN::Module
  attr_reader :fc1, :fc2

  def initialize
    super()
    @fc1 = Daimond::NN::Linear.new(784, 128)
    @fc2 = Daimond::NN::Linear.new(128, 10)
    @parameters = @fc1.parameters + @fc2.parameters
  end

  def forward(x)
    h = @fc1.forward(x).relu
    @fc2.forward(h).softmax
  end
end

model = MNISTNet.new
optimizer = Daimond::Optim::SGD.new(model.parameters, lr: 0.1, momentum: 0.9)
criterion = Daimond::Loss::CrossEntropyLoss.new

puts "\nTraining..."
epochs = 5
start_time = Time.now

epochs.times do |epoch|
  total_loss = 0
  correct = 0
  total = 0

  # Счетчики для отладки Rust usage
  rust_operations = 0
  ruby_operations = 0

  train_loader.each_batch do |x, y|
    # Проверяем размер батча для отладки
    if x.shape[0] > 100 && defined?(Daimond::Rust)
      rust_operations += 1
    else
      ruby_operations += 1
    end

    pred = model.forward(x)
    loss = criterion.call(pred, y)

    total_loss += loss.data[0]

    batch_size = x.shape[0]
    batch_size.times do |i|
      predicted_class = pred.data[i, true].argmax
      correct += 1 if predicted_class == y.data[i]
      total += 1
    end

    optimizer.zero_grad
    loss.backward!
    optimizer.step
  end

  avg_loss = total_loss / train_loader.batches_count
  accuracy = 100.0 * correct / total
  puts "Epoch #{epoch + 1}/#{epochs}: Loss = #{avg_loss.round(4)}, Accuracy = #{accuracy.round(2)}%"
  puts "  Backend: #{rust_operations}x Rust, #{ruby_operations}x Ruby" if epoch == 0

  # Показываем среднее время на эпоху
  elapsed = Time.now - start_time
  avg_epoch_time = elapsed / (epoch + 1)
  remaining = avg_epoch_time * (epochs - epoch - 1)
  puts "  Time: #{elapsed.round(1)}s elapsed, ~#{remaining.round(1)}s remaining"
end

total_time = Time.now - start_time
puts "\n🎉 Training complete in #{total_time.round(2)}s!"

# Результаты
puts "\nTesting on first test image..."
test_x = Daimond::Tensor.new([test_images[0]])
pred = model.forward(test_x)
predicted_digit = pred.data[0, true].argmax
puts "Predicted: #{predicted_digit}, Actual: #{test_labels[0]}"

puts "\n=== BACKEND STATS ==="
puts "Rust operations: #{$rust_count || 0}"
puts "Ruby operations: #{$ruby_count || 0}"

# Сравнение с pure Ruby (если Rust был использован)
if defined?(Daimond::Rust)
  puts "\n💡 Tip: Run with Ruby-only by removing 'require_relative' for Rust to compare speeds"
end