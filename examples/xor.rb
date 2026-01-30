require_relative '../lib/daimond'

# XOR данные
X = [
  [0, 0],
  [0, 1],
  [1, 0],
  [1, 1]
]
Y = [[0], [1], [1], [0]]

# Создаем модель
class XORNet < Daimond::NN::Module
  def initialize
    super()
    @fc1 = Daimond::NN::Linear.new(2, 2)
    @fc2 = Daimond::NN::Linear.new(2, 1)
    @parameters = @fc1.parameters + @fc2.parameters
  end

  def forward(x)
    h = @fc1.forward(x).sigmoid
    @fc2.forward(h).sigmoid
  end
end

model = XORNet.new
optimizer = Daimond::Optim::SGD.new(model.parameters, lr: 1.0)

puts "Training XOR with Autograd..."

1000.times do |epoch|
  total_loss = 0

  X.each_with_index do |x_row, i|
    y_true = Y[i][0]

    # Forward
    x_tensor = Daimond::Tensor.new([x_row])
    y_true_tensor = Daimond::Tensor.new([[y_true]])

    pred = model.forward(x_tensor)

    # MSE Loss: (pred - y)^2
    diff = pred - y_true_tensor
    loss = (diff * diff).sum

    total_loss += loss.data[0]

    # Backward - вот где магия!
    optimizer.zero_grad
    loss.backward!

    # Update
    optimizer.step
  end

  if epoch % 100 == 0
    avg_loss = total_loss / 4.0
    puts "Epoch #{epoch}: Loss = #{avg_loss.round(6)}"
  end
end

# Тестируем результат
puts "\nResults:"
X.each_with_index do |x, i|
  pred = model.forward(Daimond::Tensor.new([x]))
  puts "Input: #{x} -> Predicted: #{pred.data[0].round(4)}, Expected: #{Y[i][0]}"
end