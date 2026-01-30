require 'numo/narray'

require_relative 'daimond/tensor'
require_relative 'daimond/nn/module'
require_relative 'daimond/nn/linear'
require_relative 'daimond/nn/functional'
require_relative 'daimond/optim/sgd'
require_relative 'daimond/loss/mse'
require_relative 'daimond/loss/cross_entropy'
require_relative 'daimond/data/mnist'
require_relative 'daimond/data/data_loader'

module Daimond
  VERSION = '0.1.0'

  def self.randn(*args)
    Tensor.randn(*args)
  end

  def self.zeros(*args)
    Tensor.zeros(*args)
  end
end