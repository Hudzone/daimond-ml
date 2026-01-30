require_relative '../lib/daimond'

# XOR датасет
X = [
  [0, 0],
  [0, 1],
  [1, 0],
  [1, 1]
]
Y = [[0], [1], [1], [0]]

# Конвертируем в тензоры
x_tensor = X.map { |row| Daimond::Tensor.new([row]) }
y_tensor = Y.map { |row| Daimond::Tensor.new([row]) }

# Модель: 2 -> 2 -> 1
class XORModel < Daimond::NN::Module
  def initialize
    super()
    @fc1 = Daimond::NN::Linear.new(2, 2)
    @fc2 = Daimond::NN::Linear.new(2, 1)
    @parameters = @fc1.parameters + @fc2.parameters
  end

  def forward(x)
    h = x.dot(@fc1.weight.data.transpose) + @fc1.bias.data
    h = h.map { |v| v > 0 ? v : 0 }  # ReLU руками для простоты

    out = h.dot(@fc2.weight.data.transpose) + @fc2.bias.data
    # Сигмоида: 1 / (1 + exp(-x))
    out.map { |v| 1.0 / (1.0 + Math.exp(-v)) }
  end

  attr_reader :fc1, :fc2
end

model = XORModel.new
optimizer = Daimond::Optim::SGD.new(model.parameters, lr: 0.5)

puts "Training XOR..."

1000.times do |epoch|
  total_loss = 0

  x_tensor.each_with_index do |x, i|
    y = y_tensor[i]

    # Forward
    pred = model.forward(x.data)
    loss = ((pred - y.data)**2).sum  # MSE

    total_loss += loss

    # Backward (ручной для начала)
    # TODO: Здесь нужен полноценный autograd, пока упрощенно
  end

  puts "Epoch #{epoch}: Loss = #{total_loss / 4.0}" if epoch % 100 == 0
end