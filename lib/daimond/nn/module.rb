module Daimond
  module NN
    class Module
      def initialize
        @parameters = []
      end

      def parameters
        @parameters
      end

      def zero_grad
        @parameters.each do |p|
          p.grad = Numo::DFloat.zeros(*p.shape)
        end
      end

      def forward(*args)
        raise NotImplementedError
      end

      def call(*args)
        forward(*args)
      end
    end
  end
end
