require_relative 'module'

module Daimond
  module NN
    class Linear < Module
      def initialize(in_features, out_features)
        super()
        # Простая инициализация: small random values
        @weight = Tensor.new(Numo::DFloat.new(in_features, out_features).rand_norm * 0.01)
        @bias = Tensor.zeros(out_features)
        @parameters = [@weight, @bias]
      end

      def forward(input)
        # Теперь возвращаем Tensor с поддержкой autograd!
        input.dot(@weight) + @bias
      end

      attr_reader :weight, :bias
    end
  end
end