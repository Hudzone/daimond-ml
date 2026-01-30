require_relative 'module'

module Daimond
  module NN
    class Linear < Module
      def initialize(in_features, out_features)
        super()
        limit = Math.sqrt(6.0 / (in_features + out_features))
        @weight = Tensor.new(Numo::DFloat.new(in_features, out_features).rand * 2 * limit - limit)
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