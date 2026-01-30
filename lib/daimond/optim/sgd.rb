module Daimond
  module Optim
    class SGD
      def initialize(parameters, lr: 0.01)
        @parameters = parameters
        @lr = lr
      end

      def step
        @parameters.each do |param|
          param.data -= @lr * param.grad
        end
      end

      def zero_grad
        @parameters.each do |p|
          p.grad = Numo::DFloat.zeros(*p.shape)
        end
      end
    end
  end
end