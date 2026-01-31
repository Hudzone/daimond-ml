require 'csv'

module Daimond
  module Utils
    class TrainingLogger
      def initialize(filename = 'training_log.csv')
        @filename = filename
        @history = []
        @start_time = Time.now

        CSV.open(@filename, 'w') do |csv|
          csv << ['epoch', 'loss', 'accuracy', 'time_elapsed']
        end
      end

      def log(epoch, loss, accuracy)
        elapsed = Time.now - @start_time
        @history << {epoch: epoch, loss: loss, accuracy: accuracy, time: elapsed}

        CSV.open(@filename, 'a') do |csv|
          csv << [epoch, loss, accuracy, elapsed.round(2)]
        end
      end

      def plot_loss(width: 60, height: 10)
        return if @history.empty?

        losses = @history.map { |h| h[:loss] }
        min_loss = losses.min
        max_loss = losses.max
        range = max_loss - min_loss
        range = 1.0 if range == 0

        puts "\n📉 Loss Curve:"
        puts "-" * (width + 10)

        height.times do |row|
          y_val = max_loss - (row * range / height)
          line = sprintf("%6.3f |", y_val)
          line = line.ljust(width + 9)  # <-- Важно!

          @history.each_with_index do |h, i|
            x_pos = (i * (width - 1) / [@history.size - 1, 1].max).to_i
            idx = 8 + x_pos

            if idx < line.length && (h[:loss] - y_val).abs < (range / height / 2)
              line[idx] = "●"
            end
          end

          puts line
        end

        puts "       +" + "-" * width
        puts "       Epoch: 1" + " " * (width - 10) + "#{@history.size}"
      end

      def plot_accuracy(width: 60, height: 10)
        return if @history.empty?

        accs = @history.map { |h| h[:accuracy] }
        min_acc = [accs.min, 0].min
        max_acc = [accs.max, 100].max
        range = max_acc - min_acc
        range = 100 if range == 0

        puts "\n📈 Accuracy Curve:"
        puts "-" * (width + 10)

        height.times do |row|
          y_val = min_acc + ((height - row) * range / height)
          line = sprintf("%6.1f%%|", y_val)
          line = line.ljust(width + 9)  # <-- Важно!

          @history.each_with_index do |h, i|
            x_pos = (i * (width - 1) / [@history.size - 1, 1].max).to_i
            idx = 8 + x_pos

            if idx < line.length && (h[:accuracy] - y_val).abs < (range / height / 2)
              line[idx] = "★"
            end
          end

          puts line
        end

        puts "       +" + "-" * width
        puts "       Epoch: 1" + " " * (width - 10) + "#{@history.size}"
      end

      def summary
        return if @history.empty?

        first = @history.first
        last = @history.last

        puts "\n📊 Training Summary:"
        puts "=" * 50
        puts "Duration:        #{(last[:time] / 60).round(1)} minutes"
        puts "Epochs:          #{last[:epoch]}"
        puts "Initial Loss:    #{first[:loss].round(4)}"
        puts "Final Loss:      #{last[:loss].round(4)}"
        puts "Initial Acc:     #{first[:accuracy].round(2)}%"
        puts "Final Acc:       #{last[:accuracy].round(2)}%"
        puts "Improvement:     +#{(last[:accuracy] - first[:accuracy]).round(2)}%"
        puts "=" * 50
        puts "Log saved to:    #{@filename}"
      end
    end
  end
end