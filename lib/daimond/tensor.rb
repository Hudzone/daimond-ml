require 'numo/narray'

module Daimond
  class Tensor
    attr_accessor :data, :grad, :prev, :op, :label

    def initialize(data, prev: [], op: nil, label: nil)
      @data = data.is_a?(Numo::DFloat) ? data : Numo::DFloat[*data]
      @grad = Numo::DFloat.zeros(*@data.shape)
      @prev = prev
      @op = op
      @label = label
    end

    def shape
      @data.shape
    end

    # Операции с поддержкой градиентов
    def +(other)
      other = other.is_a?(Tensor) ? other : Tensor.new(other)
      out = Tensor.new(@data + other.data, prev: [self, other], op: '+')

      # Backward function для сложения
      define_singleton_method(:backward) do
        self.grad += out.grad
        other.grad += out.grad
      end

      out
    end

    def *(other)  # Поэлементное умножение
      other = other.is_a?(Tensor) ? other : Tensor.new(other)
      out = Tensor.new(@data * other.data, prev: [self, other], op: '*')

      define_singleton_method(:backward) do
        self.grad += other.data * out.grad
        other.grad += self.data * out.grad
      end

      out
    end

    def dot(other)
      other = other.is_a?(Tensor) ? other : Tensor.new(other)
      out = Tensor.new(@data.dot(other.data), prev: [self, other], op: 'dot')

      define_singleton_method(:backward) do
        self.grad += out.grad.dot(other.data.transpose)
        other.grad += self.data.transpose.dot(out.grad)
      end

      out
    end

    def relu
      out = Tensor.new(@data.map { |x| x > 0 ? x : 0.0 }, prev: [self], op: 'relu')

      define_singleton_method(:backward) do
        self.grad += out.data.map { |x| x > 0 ? 1.0 : 0.0 } * out.grad
      end

      out
    end

    def sigmoid
      # 1 / (1 + exp(-x))
      out_data = @data.map { |x| 1.0 / (1.0 + Math.exp(-x)) }
      out = Tensor.new(out_data, prev: [self], op: 'sigmoid')

      # Производная сигмоиды: sigmoid(x) * (1 - sigmoid(x))
      define_singleton_method(:backward) do
        self.grad += out.data * (1.0 - out.data) * out.grad
      end

      out
    end

    def sum
      out = Tensor.new(Numo::DFloat[@data.sum], prev: [self], op: 'sum')

      define_singleton_method(:backward) do
        self.grad += Numo::DFloat.ones(*self.shape) * out.grad[0]
      end

      out
    end

    def mean
      out = Tensor.new(Numo::DFloat[@data.mean], prev: [self], op: 'mean')
      n = @data.size

      define_singleton_method(:backward) do
        self.grad += Numo::DFloat.ones(*self.shape) * (out.grad[0] / n)
      end

      out
    end

    # Обратное распространение
    def backward!
      # Топологическая сортировка графа
      topo = []
      visited = []

      build_topo = lambda do |v|
        return if visited.include?(v)
        visited << v
        v.prev.each { |child| build_topo.call(child) }
        topo << v
      end

      build_topo.call(self)

      self.grad = Numo::DFloat[1.0]  # seed

      topo.reverse.each do |node|
        node.backward if node.respond_to?(:backward)
      end
    end

    def to_s
      "Tensor(#{@data.to_a}, shape=#{shape})"
    end

    alias inspect to_s

    # Для инициализации весов
    def self.randn(*shape)
      data = Numo::DFloat.new(*shape).rand_norm
      Tensor.new(data)
    end

    def self.zeros(*shape)
      Tensor.new(Numo::DFloat.zeros(*shape))
    end

    def -(other)
      other = other.is_a?(Tensor) ? other : Tensor.new(other)
      out = Tensor.new(@data - other.data, prev: [self, other], op: '-')

      define_singleton_method(:backward) do
        self.grad += out.grad
        other.grad -= out.grad
      end

      out
    end

  end
end