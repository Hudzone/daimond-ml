require_relative 'lib/daimond'

puts "=== Daimond Performance Benchmark ===\n\n"

size = 200  # Размер матрицы

# === RUBY VERSION ===
puts "Ruby backend (Numo::NArray):"
a_ruby = Daimond::Tensor.randn(size, size)
b_ruby = Daimond::Tensor.randn(size, size)

start = Time.now
result_ruby = a_ruby.dot(b_ruby)
ruby_time = Time.now - start

puts "  Time: #{ruby_time.round(4)}s"
puts "  Sample: #{result_ruby.data[0, 0].round(6)}\n\n"

# === RUST VERSION ===
puts "Rust backend (ndarray):"
a_arr = Array.new(size) { Array.new(size) { rand } }
b_arr = Array.new(size) { Array.new(size) { rand } }

start = Time.now
a_rust = Daimond::Rust::Tensor.from_array(a_arr)
b_rust = Daimond::Rust::Tensor.from_array(b_arr)
result_rust = a_rust.matmul(b_rust)
rust_time = Time.now - start

puts "  Time: #{rust_time.round(4)}s"
puts "  Sample: #{result_rust.to_a[0][0].round(6)}\n\n"

# === SPEEDUP ===
speedup = ruby_time / rust_time
puts "Speedup: #{speedup.round(1)}x"
puts "\n✅ Rust backend is #{speedup > 1 ? 'FASTER' : 'slower'} than Ruby!"