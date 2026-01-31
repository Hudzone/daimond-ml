require_relative 'rust/daimond_rust' rescue nil

module Daimond
  # Проверяем загрузилась ли Rust библиотека
  def self.rust_available?
    defined?(Daimond::Rust) && Daimond::Rust.respond_to?(:fast_matmul_flat)
  end

  # Модуль-обертка для вызовов
  module RustBackend
    class << self
      def available?
        Daimond.rust_available?
      end

      def conv2d(input_data, weight_data, bias_data, batch, in_c, out_c, h, w, k)
        return nil unless available?

        flat_input = input_data.flatten.to_a
        flat_weight = weight_data.flatten.to_a
        flat_bias = bias_data.to_a

        result_flat = Daimond::Rust.conv2d_native(
          flat_input, flat_weight, flat_bias,
          batch, in_c, out_c, h, w, k
        )

        h_out = h - k + 1
        w_out = w - k + 1
        Numo::DFloat[*result_flat].reshape(batch, out_c, h_out, w_out)
      end

      def maxpool2d(input_data, batch, channels, h, w, k)
        return nil unless available?

        flat_input = input_data.flatten.to_a
        result_flat = Daimond::Rust.maxpool2d_native(
          flat_input, batch, channels, h, w, k
        )

        h_out = h / k
        w_out = w / k
        Numo::DFloat[*result_flat].reshape(batch, channels, h_out, w_out)
      end

      def matmul_data(narray_a, narray_b)
        return nil unless available?

        shape_a = narray_a.shape
        shape_b = narray_b.shape

        flat_a = narray_a.flatten.to_a
        flat_b = narray_b.flatten.to_a

        result_flat = Daimond::Rust.fast_matmul_flat(
          flat_a, flat_b, shape_a[0], shape_a[1], shape_b[1]
        )

        Numo::DFloat[*result_flat].reshape(shape_a[0], shape_b[1])
      end
    end
  end
end