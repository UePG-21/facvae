# facvae
A PyTorch inplementation of FactorVAE refering to ["FactorVAE: A Probabilistic Dynamic Factor Model Based on Variational Autoencoder for Predicting Cross-Sectional Stock Returns"](https://ojs.aaai.org/index.php/AAAI/article/view/20369)


## 1. Introduction
### 1.1. Abstract
As an asset pricing model in economics and finance, factor model has been widely used in quantitative investment. Towards building more effective factor models, recent years have witnessed the paradigm shift from linear models to more flexible nonlinear data-driven machine learning models. However, due to low signal-to-noise ratio of the financial data, it is quite challenging to learn effective factor models. In this paper, we propose a novel factor model, FactorVAE, as a probabilistic model with inherent randomness for noise modeling. Essentially, our model integrates the dynamic factor model (DFM) with the variational autoencoder (VAE) in machine learning, and we propose a prior-posterior learning method based on VAE, which can effectively guide the learning of model by approximating an optimal posterior factor model with future information. Particularly, considering that risk modeling is important for the noisy stock data, FactorVAE can estimate the variances from the distribution over the latent space of VAE, in addition to predicting returns. The experiments on the real stock market data demonstrate the effectiveness of FactorVAE, which outperforms various baseline methods.

### 1.2. Visualization
**1.2.1. Overall framework**
<div align=center>
    <img src="overall_framework.png" width="80%" height="80%">
</div>

**1.2.2. Encoder-decoder architecture**
<div align=center>
    <img src="encoder-decoder_architecture.png" width="50%" height="50%">
</div>

**1.2.3. Factor predictor with multi-head global attention mechanism**
<div align=center>
    <img src="multi-head_global_attention.png" width="50%" height="50%">
</div>


## 2. Notation
### 2.1. Scalar (constant)
- B: size of batch (arbitrary)
- N: size of stocks (arbitrary)
- T: size of time periods (arbitrary)
- C: size of characteristics
- H: size of hidden features
- M: size of portfolios
- K: size of factors
### 2.2. Tensor (variable)
- `x`: characteristics, B*N*T*C
- `y`: stock returns, B*N
- `z_post`: posterior latent factor returns, B*K 
- `z_prior`: prior latent factor returns, B*K 
- `y_hat`: reconstructed stock returns, B*N
- `e`: hidden features, B*N*H
- `mu_post`: mean vector of `z_post`, B*K
- `sigma_post`: std vector of `z_post`, B*K
- `mu_prior`: mean vector of `z_prior`, B*K
- `sigma_prior`: std vector of `z_prior`, B*K
- `mu_y`: mean vector of `y_hat`, B*N
- `Sigma_y`: cov matrix of `y_hat`, B*N*N


## 3. Module
### 3.1. feature_extractor.py
`FeatureExtractor` extracts stocks hidden features `e` from the historical sequential characteristics `x`.

### 3.2. factor_encoder.py
`FactorEncoder` extracts posterior factors `z_post`, a random vector following the independent Gaussian distribution, which can be described by the mean `mu_post` and the standard deviation `sigma_post`, from hidden features `e` and stock returns `y`.

### 3.3. factor_decoder.py
`FactorDecoder` calculates predicted stock returns `y_hat`, a random vector following the Gaussian distribution, which can be described by the mean `mu_y` and the covariance matrix `Sigma_y`, from distribution parameters of factor returns `z` (could be `z_post` or `z_prior`) and hidden features `e`.

### 3.4. factor_predictor.py
`FactorPredictor` extracts prior factor returns `z_prior`, a random vector following the independent Gaussian distribution, which can be described by the mean `mu_prior` and the standard deviation `sigma_prior`, from hidden features `e`.


## 4. Example
```Python
from facvae import FactorVAE, loss
```
