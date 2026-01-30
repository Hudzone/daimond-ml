# Contributing to dAImond 💎

First off, thank you for considering contributing to dAImond! This is a passion project to bring deep learning to Ruby, and every contribution helps.

## Where to Start?

- **⭐ Star the repo** - It helps others discover the project
- **🐛 Report bugs** - Found an issue? Open an issue with details
- **📖 Improve docs** - Fix typos, improve examples, add translations
- **✨ Add features** - New layers, optimizers, or examples
- **🧪 Add tests** - We need them desperately!

## Development Setup

```bash
git clone https://github.com/yourusername/daimond.git
cd daimond
bundle install
ruby examples/mnist.rb  # Verify it works
```

## How to Contribute

**Reporting Bugs**

Use GitHub Issues and include:
Ruby version (ruby -v)
Error message with full backtrace
Minimal code to reproduce
Expected vs actual behavior
Suggesting Features

Open an issue with [Feature Request] in title. Explain:
What and why?
API design (how should it look?)
Are you willing to implement it?
Pull Request Process

Fork the repo and create your branch: git checkout -b feature/amazing-feature
Make changes following our style guide below
Test manually with MNIST at minimum: ruby examples/mnist.rb should still achieve 97%+
Update docs (this README, code comments, examples)
Commit with clear messages:
feat: Add BatchNorm layer
fix: Correct ReLU backward pass
docs: Update Japanese README
Push to your fork and open a Pull Request
Wait for review (usually within 24-48 hours)
Code Style Guide

**Ruby Style**

Follow Standard Ruby style guide
Max 100 characters per line
Prefer do...end for multiline blocks
Always use frozen_string_literal: true

**For New Layers (NN)**
If adding new layers (e.g., Conv2D, LSTM):
```ruby
module Daimond
  module NN
    class Conv2d < Module
      def initialize(in_channels, out_channels, kernel_size)
        super()
        # Initialize parameters as Tensors
        # Add to @parameters array
      end
      
      def forward(x)
        # Return Tensor
        # Store _backward lambda for autograd
      end
    end
  end
end
```

**Must include:**
- super() call in initialize
- Add parameters to @parameters
- Return Tensor from forward
- Implement _backward lambda with correct gradients
- Update lib/daimond.rb exports

**For Optimizers:**
```ruby
module Daimond
  module Optim
    class Adam < SGD  # Or Module
      def initialize(parameters, lr: 0.001, beta1: 0.9, beta2: 0.999)
        super()
        @parameters = parameters
        @lr = lr
        # Initialize moments
      end
      
      def step
        # Update parameters using gradients
      end
      
      def zero_grad
        @parameters.each { |p| p.grad = Numo::DFloat.zeros(*p.shape) }
      end
    end
  end
end
```

## Testing (Important!)
**1. Verify backward pass manually for new operations:**
```ruby
# Create simple test case
x = Daimond::Tensor.new([[1.0, 2.0]])
y = x.relu
y.backward!

# Check gradient is correct manually
puts x.grad.inspect
```
**2. Verify MNIST still trains to 95%+ accuracy**
**3. Add example script in examples/ if adding major features:**
- examples/conv2d_demo.rb
- examples/adam_comparison.rb

**Documentation**
- Code comments: Explain complex math in backward passes
- README updates: If adding public API
- Type hints: Optional, but helpful (e.g., # @param [Tensor] input)

## Performance Guidelines
dAImond is for education and prototyping, but we shouldn't be painfully slow:
- Prefer Numo operations over Ruby loops for math
- Minimize object allocation in hot loops (forward/backward)
- Profile before optimizing: Use ruby -rprofile if needed

Example:
```ruby
# ❌ Bad: Ruby loop
result = []
1000.times { |i| result << data[i] * 2 }

# ✅ Good: Numo vectorized
result = data * 2
```

## Recognition
- Contributors will be:
- Added to CONTRIBUTORS.md file
- Mentioned in release notes
- Forever appreciated! 🙏

## Questions
- Open a GitHub Discussion
- Or email: fyodor@hudzone.com

## Code of Conduct
Be nice

**Thank you for helping make dAImond shine! Glory Ruby.** 💎✨