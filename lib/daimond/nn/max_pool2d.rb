require_relative 'module'

module Daimond
  module NN
    class MaxPool2d < Module
      def initialize(kernel_size, stride: nil)
        super()
        @kernel_size = kernel_size.is_a?(Array) ? kernel_size : [kernel_size, kernel_size]
        @stride = stride || kernel_size
        @mask = nil  # для backward
      end

      def forward(input)
        # input: [batch, channels, h, w]
        batch_size = input.shape[0]
        channels = input.shape[1]
        h_in = input.shape[2]
        w_in = input.shape[3]

        k_h, k_w = @kernel_size
        s = @stride

        h_out = (h_in - k_h) / s + 1
        w_out = (w_in - k_w) / s + 1

        output = Numo::DFloat.zeros(batch_size, channels, h_out, w_out)
        @mask = {}  # запоминаем индексы максимумов

        batch_size.times do |b|
          channels.times do |c|
            h_out.times do |i|
              w_out.times do |j|
                # Окно пулинга
                i0 = i * s
                j0 = j * s
                window = input.data[b, c, i0...i0+k_h, j0...j0+k_w]

                max_val = window.max
                output[b, c, i, j] = max_val

                # Сохраняем позицию максимума для backward
                max_idx = window.to_a.flatten.index(max_val)
                @mask[[b, c, i, j]] = [i0 + max_idx / k_w, j0 + max_idx % k_w]
              end
            end
          end
        end

        out = Tensor.new(output, prev: [input], op: 'maxpool2d')

        out._backward = lambda do
          grad = out.grad
          batch_size.times do |b|
            channels.times do |c|
              h_out.times do |i|
                w_out.times do |j|
                  idx_i, idx_j = @mask[[b, c, i, j]]
                  input.grad[b, c, idx_i, idx_j] += grad[b, c, i, j]
                end
              end
            end
          end
        end

        out
      end
    end
  end
end