require_relative 'module'

module Daimond
  module NN
    class Conv2d < Module
      def initialize(in_channels, out_channels, kernel_size, stride: 1, padding: 0)
        super()
        @in_channels = in_channels
        @out_channels = out_channels
        @kernel_size = kernel_size.is_a?(Array) ? kernel_size : [kernel_size, kernel_size]
        @stride = stride
        @padding = padding

        # Xavier инициализация для Conv: sqrt(2 / (in * k * k))
        k_h, k_w = @kernel_size
        limit = Math.sqrt(2.0 / (in_channels * k_h * k_w))

        # Веса: [out_channels, in_channels, k_h, k_w]
        @weight = Tensor.new(
          Numo::DFloat.new(out_channels, in_channels, k_h, k_w).rand * 2 * limit - limit
        )
        @bias = Tensor.zeros(out_channels)

        @parameters = [@weight, @bias]
      end

      def forward(input)
        # input: [batch, in_channels, height, width]
        batch_size = input.shape[0]
        in_c = @in_channels
        out_c = @out_channels
        k_h, k_w = @kernel_size

        # Размеры входа
        h_in = input.shape[2]
        w_in = input.shape[3]

        # Размеры выхода (без padding пока)
        h_out = ((h_in + 2 * @padding - k_h) / @stride).floor + 1
        w_out = ((w_in + 2 * @padding - k_w) / @stride).floor + 1

        # Выходной тензор
        output = Numo::DFloat.zeros(batch_size, out_c, h_out, w_out)

        # Добавляем padding если нужно
        if @padding > 0
          padded = Numo::DFloat.zeros(batch_size, in_c, h_in + 2*@padding, w_in + 2*@padding)
          padded[true, true, @padding...h_in+@padding, @padding...w_in+@padding] = input.data
          x_data = padded
        else
          x_data = input.data
        end

        # Свертка (4 вложенных цикла — медленно, но понятно)
        batch_size.times do |b|
          out_c.times do |oc|
            h_out.times do |i|
              w_out.times do |j|
                # Координаты окна
                i0 = i * @stride
                j0 = j * @stride

                # Извлекаем окно и считаем свёртку
                window = x_data[b, true, i0...i0+k_h, j0...j0+k_w]
                kernel = @weight.data[oc, true, true, true]

                output[b, oc, i, j] = (window * kernel).sum + @bias.data[oc]
              end
            end
          end
        end

        out_tensor = Tensor.new(output, prev: [input, @weight, @bias], op: 'conv2d')

        # Backward (упрощённо — только для stride=1, padding=0)
        out_tensor._backward = lambda do
          grad_output = out_tensor.grad  # [batch, out_c, h_out, w_out]

          # Градиент по весам
          @out_channels.times do |oc|
            @in_channels.times do |ic|
              k_h.times do |kh|
                k_w.times do |kw|
                  # Сумма по всем позициям где этот вес участвовал
                  grad_sum = 0.0
                  batch_size.times do |b|
                    h_out.times do |i|
                      w_out.times do |j|
                        # Координаты входа
                        i_in = i * @stride + kh
                        j_in = j * @stride + kw

                        grad_sum += x_data[b, ic, i_in, j_in] * grad_output[b, oc, i, j]
                      end
                    end
                  end
                  @weight.grad[oc, ic, kh, kw] += grad_sum
                end
              end
            end

            # Градиент по bias
            @bias.grad[oc] += grad_output[true, oc, true, true].sum
          end

          # Градиент по входу (если нужен)
          if input.grad
            # full convolution с rotated kernel
            # Упрощено для stride=1
          end
        end

        out_tensor
      end
    end
  end
end