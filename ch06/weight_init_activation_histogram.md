## 全体の流れ

1. **活性化関数を用意**

   * `sigmoid`, `ReLU`, `tanh` を定義している（実際に使うのは下の切り替え部分）。

2. **入力データを生成**

   * `input_data = np.random.randn(1000, 100)`
   * 1000サンプル、各サンプルは100次元（特徴量100）。
   * 以降、層を通していくデータ `x` は基本的に形状 `(1000, 100)` のまま進む。

3. **ネットワーク構造の指定**

   * `node_num = 100`：各隠れ層のノード数（ここでは100）
   * `hidden_layer_size = 5`：隠れ層5層
   * `activations = {}`：各層の出力 `z` を保存する辞書

4. **5層ぶん順伝播（forward）**

   * 最初は `x = input_data`
   * 各層 `i` でやっていることは同じで、ざっくり **線形変換 → 活性化**：

     1. **前の層の出力を入力にする**

        * `i != 0` のとき `x = activations[i-1]`
     2. **重み行列を初期化**

        * `w = np.random.randn(node_num, node_num) * 1`
        * 形状は `(100, 100)`（ノード→ノードの写像）
        * コメントアウトされている他の候補は、スケール違い（小さい値、Xavier、Heなど）を試すため。
     3. **線形変換**

        * `a = np.dot(x, w)`
        * `x: (1000,100)` と `w: (100,100)` なので `a: (1000,100)`
     4. **活性化関数**

        * `z = sigmoid(a)`（今はsigmoid）
        * `z` も `(1000,100)`
     5. **保存**

        * `activations[i] = z`

5. **各層の出力分布をヒストグラムで描画**

   * 各層の `a`（ここでは辞書に入ってる `z`）を `flatten()` して1次元化し、ヒストグラムにする。
   * `range=(0,1)` になっているので **sigmoid前提**の表示（sigmoidの出力は0〜1）。
   * 5つのサブプロットに、1層目〜5層目の分布が横並びで表示される。


## 各行の意味

### 1) `* 1`

```python
w = np.random.randn(node_num, node_num) * 1
```

* $w_{ij}\sim N(0,1)$ なので $\mathrm{Var}(w)=1$。
* もし入力が $\mathrm{Var}(x)\approx 1$ なら
  $$\mathrm{Var}(a)\approx 100 \cdot 1 \cdot 1 = 100$$
  となり、$a$ のスケールが大きくなりやすい。
* sigmoid を使うと $|a|$ が大きいほど $z=\sigma(a)$ が $0$ か $1$ に寄って飽和しやすい（分布が端に張り付く）。

### 2) `* 0.01`

```python
w = np.random.randn(node_num, node_num) * 0.01
```

* 標準偏差が $0.01$ なので
  $$\mathrm{Var}(w)=(0.01)^2=10^{-4}$$
* よって
  $$\mathrm{Var}(a)\approx 100 \cdot 1 \cdot 10^{-4}=10^{-2}$$
  と小さくなり、$a$ が0付近に集まりやすい。
* sigmoid なら $\sigma(0)=0.5$ なので、出力 $z$ が $0.5$ 周辺に寄って変化が乏しくなりやすい。

### 3) `* sqrt(1.0 / node_num)`（Xavier系の発想：tanh/sigmoid向け）

```python
w = np.random.randn(node_num, node_num) * np.sqrt(1.0 / node_num)
```

* 標準偏差は $\sqrt{1/n}$（ここでは $n=100$ なので $0.1$）。
* したがって
  $$\mathrm{Var}(w)=\frac{1}{n}$$
* よって
  $$\mathrm{Var}(a)\approx n \cdot \mathrm{Var}(x)\cdot \frac{1}{n}\approx \mathrm{Var}(x)$$
  となり、層を跨いでも分散（スケール）を保ちやすい、という狙いがある。

### 4) `* sqrt(2.0 / node_num)`（He初期化：ReLU向け）

```python
w = np.random.randn(node_num, node_num) * np.sqrt(2.0 / node_num)
```

* 標準偏差は $\sqrt{2/n}$（$n=100$ なら約 $0.1414$）。
* したがって
  $$\mathrm{Var}(w)=\frac{2}{n}$$
* ReLU は負側を0にするため、出力の分散が減りやすい。そこで $2/n$ にして、ReLU後もスケールが落ちにくいよう補正する、という考え方だ。


