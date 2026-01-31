require_relative 'module'

module Daimond
  module NN
    class Conv2dRust < Module
      attr_reader :weight, :bias

      def initialize(in_channels, out_channels, kernel_size)
        super()
        @in_channels = in_channels
        @out_channels = out_channels
        @kernel_size = kernel_size

        # Xavier инициализация
        k = kernel_size
        limit = Math.sqrt(2.0 / (in_channels * k * k))

        @weight = Tensor.new(
          Numo::DFloat.new(out_channels, in_channels, k, k).rand * 2 * limit - limit
        )
        @bias = Tensor.zeros(out_channels)
        @parameters = [@weight, @bias]
      end

      def forward(input)
        # input: [batch, in_c, h, w]
        batch = input.shape[0]
        in_c = @in_channels
        out_c = @out_channels
        h = input.shape[2]
        w = input.shape[3]
        k = @kernel_size

        # Используем Rust backend
        if Daimond::RustBackend.available?
          output_data = Daimond::RustBackend.conv2d(
            input.data, @weight.data, @bias.data,
            batch, in_c, out_c, h, w, k
          )
          out = Tensor.new(output_data, prev: [input, @weight, @bias], op: 'conv2d_rust')

          # Backward будет позже, пока заглушка
          out._backward = lambda {}

          return out
        else
          raise "Rust backend required for Conv2dRust"
        end
      end
    end
  end
end