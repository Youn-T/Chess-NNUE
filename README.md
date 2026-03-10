# Chess NNUE
## How to install ?
Clone this repo and execute the entry point [train.py](/src/chess_nnue/train.py).
## Requirements 
Make sure you have installed cupy before using this NNUE.
## The architecture
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="/docs/NNUE_white.png">
  <source media="(prefers-color-scheme: light)" srcset="/docs/NNUE_black.png">
  <img alt="NNUE scheme" src="/docs/NNUE_black.png">
</picture>

## Maths
### Forward pass
#### Inputs:  
$X_b \in \mathbb{R}^{1\times40960} \ and \ X_w \in \mathbb{R}^{1\times40960}$  

#### Weights:  
$W_1 \in \mathbb{R}^{40960\times256}, \ W_2 \in \mathbb{R}^{512\times32}, \ W_3 \in \mathbb{R}^{32\times32} \ and \ W_4 \in \mathbb{R}^{322\times1}$

#### Biases:  
$b_1 \in \mathbb{R}^{1\times512}, \ b_2 \in \mathbb{R}^{1\times32}, \ b_3 \in \mathbb{R}^{1\times32} \ and \ b_4 \in \mathbb{R}^{1\times1}$

#### Functions:  
$ReLU(x) := max(0, x)$  
$\text{Leaky\_ReLU}(x) := max(\alpha x, x), \ \alpha \in ] 0 ; 1 [$  
$Sigmoid : \ \sigma(x):= \dfrac{1}{1 + e^{-x}}$  
  
#### Operations
$Z_1w \in \mathbb{R}^{1\times256}: Z_1 := X_w W_1 + b1$  
$A_1w \in \mathbb{R}^{1\times256}: A_1 := \text{Leaky\_ReLU}(Z_1w)$

$Z_1b \in \mathbb{R}^{1\times256}: Z_1 := X_b W_1 + b1$  
$A_1b \in \mathbb{R}^{1\times256}: A_1 := \text{Leaky\_ReLU}(Z_1b)$

$A_1 \in \mathbb{R}^{1\times512}: A_1 = (A_1w | A_1b)$

$Z_2 \in \mathbb{R}^{1\times32}: Z_2 := A_1 W_2 + b2$  
$A_2 \in \mathbb{R}^{1\times32}: A_2 := \text{Leaky\_ReLU}(Z_2)$  

$Z_3 \in \mathbb{R}^{1\times32}: Z_3 := A_2 W_3 + b3$  
$A_3 \in \mathbb{R}^{1\times32}: A_3 := \text{Leaky\_ReLU}(Z_3)$  

$Z_4 \in \mathbb{R}^{1\times1}: Z_4 := A_3 W_4 + b4$  
$A_4 \in \mathbb{R}^{1\times1}: A_4 := \sigma(Z_4)$