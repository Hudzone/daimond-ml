require_relative 'module'

module Daimond
  module NN
    class Flatten < Module
      def initialize(start_dim: 1, end_dim: -1)
        super()
        @start_dim = start_dim
        @end_dim = end_dim
        @input_shape = nil
      end

      def forward(input)
        @input_shape = input.shape.dup
        batch = input.shape[0]
        rest = input.shape[1..-1].inject(:*)

        out_data = input.data.reshape(batch, rest)
        out = Tensor.new(out_data, prev: [input], op: 'flatten')

        out._backward = lambda do
          input.grad += out.grad.reshape(*@input_shape)
        end

        out
      end
    end
  end
end