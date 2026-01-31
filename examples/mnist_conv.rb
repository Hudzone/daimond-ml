require_relative '../lib/daimond'
require 'numo/narray'

puts "Loading MNIST..."
train_images = Daimond::Data::MNIST.load_images('train-images-idx3-ubyte.gz')
train_labels = Daimond::Data::MNIST.load_labels('train-labels-idx1-ubyte.gz')

# Reshape для Conv: [batch, 1, 28, 28] вместо [batch, 784]
train_images = train_images.map { |img| Numo::DFloat[*img].reshape(1, 28, 28) }

puts "Preparing data..."
# Берём subset для скорости (весь датасет будет очень медленно на чистом Ruby)
subset = 5000
train_images = train_images.first(subset)
train_labels = train_labels.first(subset)

class ConvNet < Daimond::NN::Module
  attr_reader :conv1, :pool1, :conv2, :fc1, :fc2

  def initialize
    super()
    # LeNet-like архитектура
    @conv1 = Daimond::NN::Conv2d.new(1, 6, 5)    # 1→6 каналов, ядро 5x5
    @pool1 = Daimond::NN::MaxPool2d.new(2)       # 14x14
    @conv2 = Daimond::NN::Conv2d.new(6, 16, 5)   # 6→16, 10x10 -> 5x5 после pool
    @pool2 = Daimond::NN::MaxPool2d.new(2)       # 5x5 -> 2x2 (почти)

    @flatten = Daimond::NN::Flatten.new

    # После двух пулингов: 16 каналов * 4x4 = 256 (примерно, зависит от размеров)
    # Для точности: 28->24->12->8->4 (если stride=1) или 28->24->12->8->4
    # На самом деле: 28-5+1=24 -> pool=12 -> 12-5+1=8 -> pool=4
    # Итого: 16 * 4 * 4 = 256

    @fc1 = Daimond::NN::Linear.new(256, 120)
    @fc2 = Daimond::NN::Linear.new(120, 84)
    @fc3 = Daimond::NN::Linear.new(84, 10)

    @parameters = @conv1.parameters + @conv2.parameters +
                  @fc1.parameters + @fc2.parameters + @fc3.parameters
  end

  def forward(x)
    # x: [batch, 1, 28, 28]
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

# Создаем DataLoader для 4D данных
class ConvDataLoader
  def initialize(images, labels, batch_size: 32)
    @images = images  # уже [N, 1, 28, 28]
    @labels = labels
    @batch_size = batch_size
  end

  def each_batch
    @images.each_slice(@batch_size).with_index do |imgs, i|
      labels_slice = @labels[i*@batch_size, @batch_size]
      next if imgs.size != @batch_size  # skip last incomplete batch

      x = Daimond::Tensor.new(Numo::DFloat[*imgs])
      y = Daimond::Tensor.new(Numo::Int32[*labels_slice])
      yield x, y
    end
  end

  def batches_count
    (@images.length.to_f / @batch_size).ceil
  end
end

loader = ConvDataLoader.new(train_images, train_labels, batch_size: 16)  # Маленький batch для скорости
model = ConvNet.new
optimizer = Daimond::Optim::SGD.new(model.parameters, lr: 0.01, momentum: 0.9)  # Меньше LR для Conv
criterion = Daimond::Loss::CrossEntropyLoss.new

puts "Training ConvNet (this will be SLOW on pure Ruby)..."
puts "Batch size 16, samples: #{subset}"

epochs = 3  # Меньше эпох, потому что медленно

epochs.times do |epoch|
  total_loss = 0
  correct = 0
  total = 0
  batch_num = 0

  loader.each_batch do |x, y|
    pred = model.forward(x)
    loss = criterion.call(pred, y)

    total_loss += loss.data[0]

    # Accuracy
    @batch_size = x.shape[0]
    @batch_size.times do |i|
      if pred.data[i, true].argmax == y.data[i]
        correct += 1
      end
      total += 1
    end

    optimizer.zero_grad
    loss.backward!
    optimizer.step

    batch_num += 1
    print "\rBatch #{batch_num}/#{loader.batches_count}" if batch_num % 10 == 0
  end

  avg_loss = total_loss / loader.batches_count
  acc = 100.0 * correct / total
  puts "\nEpoch #{epoch+1}: Loss=#{avg_loss.round(4)}, Acc=#{acc.round(2)}%"
end

puts "\nDone! ConvNet trained."
puts "Note: Pure Ruby Conv2D is very slow. This is proof-of-concept."
puts "For production, use PyTorch/TorchVision or wait for Rust backend."