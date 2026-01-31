lib = File.expand_path('lib', __dir__)
$LOAD_PATH.unshift(lib) unless $LOAD_PATH.include?(lib)

require 'daimond/version'

Gem::Specification.new do |spec|
  spec.name          = "daimond"
  spec.version       = Daimond::VERSION
  spec.authors       = ["Hudzone"]
  spec.email         = ["your-email@example.com"]

  spec.summary       = %q{Deep Learning framework for Ruby with Rust backend}
  spec.description   = %q{dAImond is a PyTorch-inspired deep learning framework for Ruby featuring automatic differentiation, neural networks, and a high-performance Rust backend for tensor operations. Achieves 89%+ accuracy on MNIST.}
  spec.homepage      = "https://github.com/Hudzone/daimond-ml"
  spec.license       = "MIT"

  spec.required_ruby_version = ">= 2.7.0"

  # Файлы для inclusion (всё кроме target/, .git и т.д.)
  spec.files = Dir.glob("lib/**/*") +
               Dir.glob("ext/**/*") +
               ["README.md", "README.ru.md", "README.ja.md", "CONTRIBUTIONG.md"]
  spec.files.reject! { |f| f.include?("target/") || f.include?(".git") }

  spec.bindir        = "exe"
  spec.executables   = spec.files.grep(%r{^exe/}) { |f| File.basename(f) }
  spec.require_paths = ["lib"]

  # Зависимости
  spec.add_dependency "numo-narray", "~> 0.9"

  # Для разработки (сборка Rust)
  spec.add_development_dependency "bundler", "~> 2.0"
  spec.add_development_dependency "rake", "~> 13.0"
  spec.add_development_dependency "rake-compiler", "~> 1.0"

  # Метаданные для RubyGems
  spec.metadata["homepage_uri"] = spec.homepage
  spec.metadata["source_code_uri"] = "https://github.com/Hudzone/daimond-ml"
  spec.metadata["bug_tracker_uri"] = "https://github.com/Hudzone/daimond-ml/issues"
end