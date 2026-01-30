require 'fileutils'

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

      # Сохранение модели
      def save(path)
        FileUtils.mkdir_p(File.dirname(path)) if File.dirname(path) != '.'

        # Сохраняем массив весов как массив Numo массивов
        params_data = @parameters.map { |p| p.data }
        File.open(path, 'wb') { |f| Marshal.dump(params_data, f) }

        puts "Model saved to #{path} (#{@parameters.length} parameters)"
      end

      # Загрузка модели
      def load(path)
        unless File.exist?(path)
          raise "Model file not found: #{path}"
        end

        params_data = File.open(path, 'rb') { |f| Marshal.load(f) }

        if params_data.length != @parameters.length
          raise "Parameter count mismatch: saved #{params_data.length} vs current #{@parameters.length}"
        end

        @parameters.each_with_index do |param, i|
          param.data = params_data[i]
          param.grad = Numo::DFloat.zeros(*param.data.shape)
        end

        puts "Model loaded from #{path}"
      end
    end
  end
end