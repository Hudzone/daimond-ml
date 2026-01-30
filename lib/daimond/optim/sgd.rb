module Daimond
  module Optim
    class SGD
      def initialize(parameters, lr: 0.01, momentum: 0.9)
        @parameters = parameters
        @lr = lr
        @momentum = momentum
        @velocities = parameters.map { |p| Numo::DFloat.zeros(*p.shape) }
      end

      def step
        @parameters.each_with_index do |param, i|
          @velocities[i] = @momentum * @velocities[i] + param.grad
          param.data -= @lr * @velocities[i]
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