module Daimond
  module Optim
    class Adam < SGD
      def initialize(parameters, lr: 0.001, betas: [0.9, 0.999], eps: 1e-8)
        super(parameters, lr: lr)
        @betas = betas
        @eps = eps

        # Первые и вторые моменты
        @m = @parameters.map { |p| Numo::DFloat.zeros(*p.shape) }  # первый момент (среднее)
        @v = @parameters.map { |p| Numo::DFloat.zeros(*p.shape) }  # второй момент (квадраты)
        @t = 0  # шаг обновления
      end

      def step
        @t += 1
        beta1, beta2 = @betas

        @parameters.each_with_index do |param, i|
          # Градиент
          g = param.grad

          # Обновляем моменты
          @m[i] = beta1 * @m[i] + (1 - beta1) * g
          @v[i] = beta2 * @v[i] + (1 - beta2) * (g * g)

          # Коррекция смещения (bias correction)
          m_hat = @m[i] / (1 - beta1**@t)
          v_hat = @v[i] / (1 - beta2**@t)

          # Обновление параметров
          param.data -= @lr * m_hat / (Numo::NMath.sqrt(v_hat) + @eps)
        end
      end

      def zero_grad
        @parameters.each { |p| p.grad = Numo::DFloat.zeros(*p.shape) }
      end
    end
  end
end