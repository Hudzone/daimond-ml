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
require_relative 'daimond/nn/conv2d'
require_relative 'daimond/nn/max_pool2d'
require_relative 'daimond/nn/flatten'
require_relative 'daimond/optim/adam'
require_relative 'daimond/nn/conv2d_rust'
require_relative 'daimond/nn/max_pool2d_rust'
begin
  require_relative 'daimond/rust_bridge'
rescue LoadError
  # Rust backend не обязателен
end

module Daimond
  VERSION = '0.1.0'

  def self.randn(*args)
    Tensor.randn(*args)
  end

  def self.zeros(*args)
    Tensor.zeros(*args)
  end
end

begin
  require_relative 'daimond/rust_backend'
rescue LoadError
  # Rust backend optional
end