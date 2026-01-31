# Путь от текущей папки (корень проекта)
path = File.expand_path('lib/daimond/rust/daimond_rust.bundle', __dir__)

puts "Файл: #{path}"
puts "Существует: #{File.exist?(path)}"

begin
  require path
  puts "✅ Загружено!"
  puts Daimond.constants.sort
rescue LoadError => e
  puts "❌ Ошибка: #{e.message}"
end