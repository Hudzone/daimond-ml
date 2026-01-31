require_relative 'lib/daimond'
require_relative 'lib/daimond/rust_bridge'

puts "Rust available: #{Daimond::Rust.available? ? '✅ YES' : '❌ NO'}"

if Daimond::Rust.available?
  a = Numo::DFloat[[1,2],[3,4]]
  b = Numo::DFloat[[5,6],[7,8]]
  result = Daimond::Rust.matmul_data(a, b)
  puts "MatMul result: #{result.to_a}"
  puts "✅ Rust backend working!"
end