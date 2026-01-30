# dAImond 💎

PyTorchにインスパイアされた、Rubyのためのディープラーニングフレームワーク。Rubyコミュニティへの愛を込めて、ゼロから作成しました。

[![Ruby](https://img.shields.io/badge/ruby-%23CC342D.svg?style=for-the-badge&logo=ruby&logoColor=white)](https://www.ruby-lang.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **なぜRuby？** 理由はわからない、ただこの言語が好きだから。dAImondはMLに幸せを取り戻します。

## 主な機能

- 🔥 **自動微分** - 計算グラフによる完全なautogradエンジン
- 🧠 **ニューラルネットワーク** - 線形レイヤー、活性化関数（ReLU、Sigmoid、Softmax、Tanh）
- 📊 **オプティマイザー** - Momentum付きSGD、学習率スケジューリング
- 🎯 **損失関数** - MSE、CrossEntropy
- 💾 **モデルの保存** - Marshalによる保存/読み込み
- 📈 **データローダー** - バッチ処理、シャッフル、MNIST対応
- ⚡ **高速バックエンド** - Numo::NArrayによるベクトル化演算（C速度）
- 🎨 **美しいAPI** - 慣用的なRuby DSL、メソッドチェーン

## インストール

Gemfileに追加：

```ruby
gem 'daimond'
```

または手動でインストール：
```ruby
gem install daimond
```

**依存関係:** Ruby 2.7+, numo-narray
クイックスタート
```ruby
require 'daimond'

# モデルの定義
class NeuralNet < Daimond::NN::Module
  attr_reader :fc1, :fc2
  
  def initialize
    super()
    @fc1 = Daimond::NN::Linear.new(784, 128)
    @fc2 = Daimond::NN::Linear.new(128, 10)
    @parameters = @fc1.parameters + @fc2.parameters
  end
  
  def forward(x)
    x = @fc1.forward(x).relu
    @fc2.forward(x).softmax
  end
end

# トレーニングループ
model = NeuralNet.new
optimizer = Daimond::Optim::SGD.new(model.parameters, lr: 0.1, momentum: 0.9)
criterion = Daimond::Loss::CrossEntropyLoss.new

# 順伝播 → 逆伝播 → 更新
loss = criterion.call(model.forward(input), target)
optimizer.zero_grad
loss.backward!
optimizer.step
```

## MNISTの例（97%の精度！）

60,000枚の手書き数字で分類器をトレーニング：
```ruby
ruby examples/mnist.rb
```

結果：
```text
Epoch 1/5: Loss = 0.2898, Accuracy = 91.35%
Epoch 2/5: Loss = 0.1638, Accuracy = 95.31%
Epoch 3/5: Loss = 0.1389, Accuracy = 96.2%
Epoch 4/5: Loss = 0.1195, Accuracy = 96.68%
Epoch 5/5: Loss = 0.1083, Accuracy = 97.12%
```

モデルの保存：
```ruby
model.save('models/mnist_model.bin')
```
読み込みと予測：
```ruby
model = NeuralNet.new
model.load('models/mnist_model.bin')
prediction = model.forward(test_image)
```

## パフォーマンス
純粋なRubyはPyTorch/CUDAより遅いですが、dAImondはプロトタイピングや小〜中規模のデータセットに対して合理的な速度を実現しています：
MNIST（60k画像）：現代のCPUで1エポックあたり約2〜3分
教育、研究、100万パラメーター未満のモデルに最適

## ロードマップ
- [x] コアautogradエンジン
- [x] 線形レイヤーと活性化関数
- [x] MNIST 97%精度
- [x] モデルのシリアライズ
- [ ] 畳み込みレイヤー（Conv2D）
- [ ] Batch NormalizationとDropout
- [ ] Adam/RMSpropオプティマイザー
- [ ] GPUサポート（OpenCL/CUDA経由FFI）
- [ ] ONNXエクスポート/インポート

## コントリビューション
どんなコントリビューターも歓迎します！CONTRIBUTING.mdをご覧ください。

## ライセンス
MIT License - LICENSEファイルを参照。