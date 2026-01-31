require_relative 'module'

module Daimond
  module NN
    class MaxPool2dRust < Module
      def initialize(kernel_size)
        super()
        @kernel_size = kernel_size
      end

      def forward(input)

        batch = input.shape[0]
        channels = input.shape[1]
        h = input.shape[2]
        w = input.shape[3]
        k = @kernel_size

        if Daimond::RustBackend.available?
          output_data = Daimond::RustBackend.maxpool2d(
            input.data, batch, channels, h, w, k
          )

          out = Tensor.new(output_data, prev: [input], op: 'maxpool2d_rust')
          out._backward = lambda {}
          return out
        else
          raise "Rust backend required for MaxPool2dRust"
        end
      end
    end
  end
end