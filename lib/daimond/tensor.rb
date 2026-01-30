require 'numo/narray'

module Daimond
  class Tensor
    attr_accessor :data, :grad, :prev, :op, :_backward, :label

    def initialize(data, prev: [], op: nil, label: nil)
      @data = data.is_a?(Numo::DFloat) ? data : Numo::DFloat[*data]
      @grad = Numo::DFloat.zeros(*@data.shape)
      @prev = prev
      @op = op
      @label = label
      @_backward = lambda {}  # По умолчанию пустая функция
    end

    def shape
      @data.shape
    end

    def +(other)
      other = other.is_a?(Tensor) ? other : Tensor.new(other)
      left = self
      right = other
      out = Tensor.new(@data + other.data, prev: [self, other], op: '+')

      out._backward = lambda do
        grad = out.grad

        # Для left (может быть broadcasted, но здесь обычно нет)
        if grad.shape.length > left.shape.length
          left.grad += grad.sum(axis: 0)
        else
          left.grad += grad
        end

        # Для right (bias) — суммируем по batch
        if grad.shape.length > right.shape.length
          right.grad += grad.sum(axis: 0)
        else
          right.grad += grad
        end
      end

      out
    end

    def -(other)
      other = other.is_a?(Tensor) ? other : Tensor.new(other)
      left = self
      right = other
      out = Tensor.new(@data - other.data, prev: [self, other], op: '-')

      out._backward = lambda do
        grad = out.grad
        left.grad += grad

        if grad.shape.length > right.shape.length
          right.grad -= grad.sum(axis: 0)
        else
          right.grad -= grad
        end
      end

      out
    end

    def *(other)  # Поэлементное
      other = other.is_a?(Tensor) ? other : Tensor.new(other)
      left = self
      right = other
      out = Tensor.new(@data * other.data, prev: [self, other], op: '*')

      out._backward = lambda do
        grad = out.grad
        left.grad += right.data * grad
        right.grad += left.data * grad
      end

      out
    end

    def dot(other)
      other = other.is_a?(Tensor) ? other : Tensor.new(other)
      left = self   # input [batch, in]
      right = other # weight [in, out]
      out = Tensor.new(@data.dot(other.data), prev: [self, other], op: 'dot')

      out._backward = lambda do
        grad = out.grad  # [batch, out]

        # dL/dW = X^T @ grad  -> [in, batch] @ [batch, out] = [in, out]
        right.grad += left.data.transpose.dot(grad)

        # dL/dX = grad @ W^T  -> [batch, out] @ [out, in] = [batch, in]
        left.grad += grad.dot(right.data.transpose)
      end

      out
    end

    def relu
      input = self
      out = Tensor.new(@data.map { |x| x > 0 ? x : 0.0 }, prev: [self], op: 'relu')

      out._backward = lambda do
        grad = out.grad
        mask = out.data.map { |x| x > 0 ? 1.0 : 0.0 }
        input.grad += mask * grad
      end

      out
    end

    def sigmoid
      input = self
      s = @data.map { |x| 1.0 / (1.0 + Math.exp(-x)) }
      out = Tensor.new(s, prev: [self], op: 'sigmoid')

      out._backward = lambda do
        grad = out.grad
        input.grad += (out.data * (1.0 - out.data)) * grad
      end

      out
    end

    def sum
      input = self
      out = Tensor.new(Numo::DFloat[@data.sum], prev: [self], op: 'sum')

      out._backward = lambda do
        input.grad += Numo::DFloat.ones(*input.shape) * out.grad[0]
      end

      out
    end

    def mean
      input = self
      out = Tensor.new(Numo::DFloat[@data.mean], prev: [self], op: 'mean')
      n = @data.size

      out._backward = lambda do
        input.grad += Numo::DFloat.ones(*input.shape) * (out.grad[0] / n)
      end

      out
    end

    def backward!
      # Топологическая сортировка
      topo = []
      visited = []

      build_topo = lambda do |v|
        return if visited.include?(v)
        visited << v
        v.prev.each { |child| build_topo.call(child) }
        topo << v
      end

      build_topo.call(self)

      self.grad = Numo::DFloat[1.0]  # seed gradient

      # Идём в обратном порядке (от loss к входам)
      topo.reverse.each do |node|
        node._backward.call
      end
    end

    def to_s
      "Tensor(shape=#{shape}, mean=#{@data.mean.round(4)})"
    end

    alias inspect to_s

    def self.randn(*shape)
      data = Numo::DFloat.new(*shape).rand_norm
      Tensor.new(data)
    end

    def self.zeros(*shape)
      Tensor.new(Numo::DFloat.zeros(*shape))
    end

    def softmax
      input = self
      # Численная стабильность: вычитаем max по каждой строке
      max_val = @data.max(axis: 1).reshape(@data.shape[0], 1)
      exp_data = Numo::NMath.exp(@data - max_val)
      sum_exp = exp_data.sum(axis: 1).reshape(@data.shape[0], 1)
      out_data = exp_data / sum_exp

      out = Tensor.new(out_data, prev: [self], op: 'softmax')

      # Backward упрощенный (для связки с CrossEntropy)
      out._backward = lambda do
        input.grad += out.grad
      end

      out
    end
  end
end