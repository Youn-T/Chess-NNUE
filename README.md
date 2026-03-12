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
$W_1 \in \mathbb{R}^{40960\times256}, \ W_2 \in \mathbb{R}^{512\times32}, \ W_3 \in \mathbb{R}^{32\times32} \ and \ W_4 \in \mathbb{R}^{32\times1}$

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

### Backward pass
#### Functions:
$\sigma^{\prime}(x) := \sigma(x) \times (1 - \sigma(x)) $  
Indeed :
$\sigma(x):= \dfrac{1}{1 + e^{-x}} \\
\sigma^{\prime}(x)= \dfrac{e^{-x}}{(1 + e^{-x})^{2}} \\ 
  = \dfrac{e^{-x}}{1 + e^{-x}} \times \dfrac{1}{1 + e^{-x}}\\
  = \dfrac{(1 + e^{-x}) -1}{1 + e^{-x}} \times \dfrac{1}{1 + e^{-x}}\\
  = (\dfrac{1 + e^{-x}}{1 + e^{-x}} - \dfrac{1}{1 + e^{-x}}) \times \dfrac{1}{1 + e^{-x}}\\
  = (1 - \dfrac{1}{1 + e^{-x}}) \times \dfrac{1}{1 + e^{-x}}\\
  = \sigma(x) \times (1 - \sigma(x)) $  

$BCE(y_i, \hat{y}_i) = y_i \ log(\hat{y_i}) + (1 - y_i) \ log(1 - \hat{y_i})$    

$ \dfrac{\partial{BCE}}{\partial{\hat{y}}} = \dfrac{y_i}{\hat{y}_i}- \dfrac{1 - y_i}{1 - \hat{y_i}} = \dfrac{\hat{y_i} - y_i}{\hat{y_i} (1 - \hat{y_i})}$  

Or $\hat{y_i} = \sigma(x)$  

Donc : $\dfrac{\partial BCE}{\partial x_i} =  \dfrac{\partial BCE}{\partial \sigma} \times \dfrac{\partial \sigma}{\partial x_i} = \dfrac{\sigma(x) - y_i }{\sigma(x) (1 - \sigma(x))} \times (\sigma(x) \times (1 - \sigma(x))) = \sigma(x) - y_i$

#### Opérations
$L = BCE(A_4) \ and \ A_4 = \sigma(Z_4)$  
$d_4 \in \mathbb{R}^{1 \times 1}: d_4 = \dfrac{\partial L}{\partial Z_4} = \dfrac{\partial L}{\partial A_4} \times \dfrac{\partial A_4}{\partial Z_4} = A_4 -Y$  
$d_3 \in \mathbb{R}^{1 \times 32}: d_3 = \dfrac{\partial L}{\partial Z_3} = d_4 \times \dfrac{\partial Z_4}{\partial A_3} \times \dfrac{\partial A_3}{\partial Z_3} = (d_4 \cdot  W_4^{T}) \ \odot \ \text{Leaky\_ReLU}^{\prime}(Z_3)$  
$d_2 \in \mathbb{R}^{1 \times 32}: d_2 = \dfrac{\partial L}{\partial Z_2} = d_3 \times \dfrac{\partial Z_3}{\partial A_2} \times \dfrac{\partial A_2}{\partial Z_2} = (d_3 \cdot W_3^{T}) \ \odot \ \text{Leaky\_ReLU}^{\prime}(Z_2)$  
$d_1 \in \mathbb{R}^{1 \times 512}: d_1 = \dfrac{\partial L}{\partial A_1} = d_2 \times \dfrac{\partial Z_2}{\partial A_1} = d_2 \cdot W_2^{T}$ 

$d'_{1b} \in \mathbb{R}^{1 \times 256}: d'_{1b} = \dfrac{\partial L}{\partial Z_{1b}} = d_1 \times \dfrac{\partial A_{1b}}{\partial Z_{1b}} = d_{1b} \ \odot \ \text{Leaky\_ReLU}^{\prime}(Z_{1b})$  
$d'_{1w} \in \mathbb{R}^{1 \times 256}: d'_{1w} = \dfrac{\partial L}{\partial Z_{1w}} = d_1 \times \dfrac{\partial A_{1w}}{\partial Z_{1w}} = d_{1w} \ \odot \ \text{Leaky\_ReLU}^{\prime}(Z_{1w})$  

<!-- $d_1 \in \mathbb{R}^{1 \times 32}: d_1 = \dfrac{\partial L}{\partial Z_1} = d_1 \times \dfrac{\partial Z_2}{\partial A_1} \times \dfrac{\partial A_1}{\partial Z_1} = (d_2 \cdot W_2^{T}) \ \odot \ \text{Leaky\_ReLU}^{\prime}(Z_1)$ -->

$dW_4 \in \mathbb{R}^{32 \times 1}: \dfrac{\partial L}{\partial W_4} = d_4 \times \dfrac{\partial Z_4}{\partial W_4} = A_3^T \cdot d_4$  
$dW_3 \in \mathbb{R}^{32 \times 32}: \dfrac{\partial L}{\partial W_3} = d_3 \times \dfrac{\partial Z_3}{\partial W_3} = A_2^T \cdot d_3$  
$dW_2 \in \mathbb{R}^{512 \times 32}: \dfrac{\partial L}{\partial W_2} = d_2 \times \dfrac{\partial Z_2}{\partial W_2} = A_1^T \cdot d_2$  

$dW_{1b} \in \mathbb{R}^{40960 \times 256}: \dfrac{\partial L}{\partial W_{1b}} = d'_{1b} \times \dfrac{\partial Z_{1b}}{\partial W_{1b}} = X^T_b \cdot d'_{1b}$  
$dW_{1w} \in \mathbb{R}^{40960 \times 256}: \dfrac{\partial L}{\partial W_{1w}} = d'_{1w} \times \dfrac{\partial Z_{1w}}{\partial W_{1w}} = X^T_b \cdot d'_{1w}$  
$dW_1 \in \mathbb{R}^{40960 \times 256} : dW_1 = dW_{1b} + dW_{1w}$
<!-- $dW_1 \in \mathbb{R}^{40960 \times 256}: \dfrac{\partial L}{\partial W_1} = d_1 \times \dfrac{\partial Z_1}{\partial W_1} = X^T \cdot d_1$   -->

$db_4 \in \mathbb{R}^{1 \times 1}: \dfrac{\partial L}{\partial b_4} = d_4 \times \dfrac{\partial Z_4}{\partial b_4} = d_4$  
$db_3 \in \mathbb{R}^{1 \times 32}: \dfrac{\partial L}{\partial b_3} = d_3 \times \dfrac{\partial Z_3}{\partial b_3} = d_3$  
$db_2 \in \mathbb{R}^{1 \times 32}: \dfrac{\partial L}{\partial b_2} = d_2 \times \dfrac{\partial Z_2}{\partial b_2} = d_2$  

$db_{1b} \in \mathbb{R}^{1 \times 256}: \dfrac{\partial L}{\partial b_{1b}} = d'_{1b} \times \dfrac{\partial Z_{1b}}{\partial b_{1b}} = d'_{1b}$  
$db_{1w} \in \mathbb{R}^{1 \times 256}: \dfrac{\partial L}{\partial b_{1w}} = d'_{1w} \times \dfrac{\partial Z_{1w}}{\partial b_{1w}} = d'_{1w}$  
$db_1 \in \mathbb{R}^{1 \times 256}: db_1 = db_{1b} + db_{1w}$  