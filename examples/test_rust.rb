require_relative '../lib/daimond'

if Daimond::RustBackend.available?
  puts "✅ Rust backend loaded successfully!"

  # Тест создания тензора
  tensor = Daimond::Rust::Tensor.zeros(3, 4)
  puts "Created tensor shape: #{tensor.shape}"
else
  puts "❌ Rust backend not available"
end