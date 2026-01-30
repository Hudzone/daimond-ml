require_relative '../lib/daimond'

puts "Loading MNIST dataset..."
train_images = Daimond::Data::MNIST.load_images('train-images-idx3-ubyte.gz')
train_labels = Daimond::Data::MNIST.load_labels('train-labels-idx1-ubyte.gz')

test_images = Daimond::Data::MNIST.load_images('t10k-images-idx3-ubyte.gz').first(200)
test_labels = Daimond::Data::MNIST.load_labels('t10k-labels-idx1-ubyte.gz').first(200)

puts "Train samples: #{train_images.length}, Test samples: #{test_images.length}"

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
    logits = @fc2.forward(h)
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

  train_loader.each_batch do |x, y|
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
end

# Сохраняем модель
model.save('models/mnist_model.bin')

# Тестируем
puts "\nTesting..."
test_x = Daimond::Tensor.new([test_images[0]])
pred = model.forward(test_x)
puts "First test: Predicted #{pred.data[0, true].argmax}, Actual #{test_labels[0]}"

puts "\nPredictions (first 10):"
10.times do |i|
  test_x = Daimond::Tensor.new([test_images[i]])
  pred = model.forward(test_x)
  predicted = pred.data[0, true].argmax
  status = predicted == test_labels[i] ? "✓" : "✗"
  puts "  #{status} Image #{i}: Predicted #{predicted}, Actual #{test_labels[i]}"
end