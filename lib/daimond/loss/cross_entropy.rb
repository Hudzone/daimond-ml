require_relative '../tensor'
require 'numo/narray'

module Daimond
  module Loss
    class CrossEntropyLoss
      def initialize
      end

      def forward(pred, target)
        # pred: [batch_size, 10] после softmax
        # target: [batch_size] метки классов (0-9)
        batch_size = pred.shape[0]

        # Вычисляем loss для мониторинга (не используется в backward)
        log_probs = Numo::NMath.log(pred.data)
        correct_log_probs = Numo::DFloat.zeros(batch_size)

        batch_size.times do |i|
          correct_log_probs[i] = log_probs[i, target.data[i]]
        end

        loss_value = -correct_log_probs.mean

        out = Tensor.new(Numo::DFloat[loss_value], prev: [pred], op: 'cross_entropy')

        # Backward: gradient of cross_entropy + softmax = pred - one_hot(target)
        out._backward = lambda do
          grad_input = pred.data.dup  # softmax output
          batch_size.times do |i|
            grad_input[i, target.data[i]] -= 1.0
          end
          grad_input /= batch_size
          pred.grad += grad_input
        end

        out
      end

      def call(pred, target)
        forward(pred, target)
      end
    end
  end
end