require_relative '../lib/daimond'
require_relative '../lib/daimond/utils/training_logger'

puts "=== dAImond ConvNet with Visualization ==="
puts "Rust available: #{Daimond::RustBackend.available? ? '✅ YES' : '❌ NO'}"

unless Daimond::RustBackend.available?
  puts "Rust backend required! Compile: cd ext/daimond_rust && cargo build --release"
  exit 1
end

# Загрузка MNIST
puts "\nLoading MNIST..."
train_images = Daimond::Data::MNIST.load_images('train-images-idx3-ubyte.gz').first(10000)
train_labels = Daimond::Data::MNIST.load_labels('train-labels-idx1-ubyte.gz').first(10000)

puts "Preparing data..."
train_images = train_images.map { |img| Numo::DFloat[*img].reshape(1, 28, 28) }

# ConvNet архитектура
class ConvNetRust < Daimond::NN::Module
  attr_reader :conv1, :pool1, :conv2, :fc1

  def initialize
    super()
    @conv1 = Daimond::NN::Conv2dRust.new(1, 6, 5)
    @pool1 = Daimond::NN::MaxPool2dRust.new(2)
    @conv2 = Daimond::NN::Conv2dRust.new(6, 16, 5)
    @pool2 = Daimond::NN::MaxPool2dRust.new(2)

    @flatten = Daimond::NN::Flatten.new
    @fc1 = Daimond::NN::Linear.new(256, 120)
    @fc2 = Daimond::NN::Linear.new(120, 84)
    @fc3 = Daimond::NN::Linear.new(84, 10)

    @parameters = @conv1.parameters + @conv2.parameters +
                  @fc1.parameters + @fc2.parameters + @fc3.parameters
  end

  def forward(x)
    x = @conv1.forward(x).relu
    x = @pool1.forward(x)
    x = @conv2.forward(x).relu
    x = @pool2.forward(x)
    x = @flatten.forward(x)
    x = @fc1.forward(x).relu
    x = @fc2.forward(x).relu
    @fc3.forward(x).softmax
  end
end

# DataLoader
class ConvDataLoader
  def initialize(images, labels, batch_size: 32)
    @images = images
    @labels = labels
    @batch_size = batch_size
  end

  def each_batch
    @images.each_slice(@batch_size).with_index do |imgs, i|
      labels_slice = @labels[i * @batch_size, @batch_size]
      next if imgs.size != @batch_size

      x = Daimond::Tensor.new(Numo::DFloat[*imgs])
      y = Daimond::Tensor.new(Numo::Int32[*labels_slice])
      yield x, y
    end
  end

  def batches_count
    (@images.length.to_f / @batch_size).ceil
  end
end

# Обучение с визуализацией
puts "\nTraining ConvNet (Rust accelerated)..."
loader = ConvDataLoader.new(train_images, train_labels, batch_size: 32)
model = ConvNetRust.new
optimizer = Daimond::Optim::Adam.new(model.parameters, lr: 0.001)
criterion = Daimond::Loss::CrossEntropyLoss.new
logger = Daimond::Utils::TrainingLogger.new('mnist_conv_log.csv')

5.times do |epoch|
  total_loss = 0
  correct = 0
  total = 0

  loader.each_batch do |x, y|
    pred = model.forward(x)
    loss = criterion.call(pred, y)

    total_loss += loss.data[0]

    batch_size = x.shape[0]
    batch_size.times do |i|
      correct += 1 if pred.data[i, true].argmax == y.data[i]
      total += 1
    end

    optimizer.zero_grad
    loss.backward!
    optimizer.step
  end

  acc = 100.0 * correct / total
  avg_loss = total_loss / loader.batches_count

  logger.log(epoch + 1, avg_loss, acc)

  elapsed = logger.instance_variable_get(:@history).last[:time]
  puts "Epoch #{epoch+1}: Loss=#{avg_loss.round(4)}, Acc=#{acc.round(2)}% (#{elapsed.round(1)}s)"
end

# Красивые графики!
logger.plot_loss
logger.plot_accuracy
logger.summary

puts "\n🎉 Training complete! Check mnist_conv_log.csv"