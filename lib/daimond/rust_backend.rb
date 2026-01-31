require_relative 'rust/daimond_rust'

module Daimond
  module RustBackend
    # Проверка доступности
    def self.available?
      true
    rescue LoadError
      false
    end

    # Обертка для матричного умножения
    def self.matmul(a, b)
      # Здесь будет код конвертации Ruby -> Rust -> Ruby
      # Пока просто возвращаем Rust тензор
      Rust::Tensor.zeros(a.shape[0], b.shape[1])
    end
  end
end