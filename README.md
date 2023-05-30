# facvae
A PyTorch inplementation of FactorVAE refering to ["FactorVAE: A Probabilistic Dynamic Factor Model Based on Variational Autoencoder for Predicting Cross-Sectional Stock Returns"](https://ojs.aaai.org/index.php/AAAI/article/view/20369)


## 1. Introduction
### 1.1. Abstract
As an asset pricing model in economics and finance, factor model has been widely used in quantitative investment. Towards building more effective factor models, recent years have witnessed the paradigm shift from linear models to more flexible nonlinear data-driven machine learning models. However, due to low signal-to-noise ratio of the financial data, it is quite challenging to learn effective factor models. In this paper, we propose a novel factor model, FactorVAE, as a probabilistic model with inherent randomness for noise modeling. Essentially, our model integrates the dynamic factor model (DFM) with the variational autoencoder (VAE) in machine learning, and we propose a prior-posterior learning method based on VAE, which can effectively guide the learning of model by approximating an optimal posterior factor model with future information. Particularly, considering that risk modeling is important for the noisy stock data, FactorVAE can estimate the variances from the distribution over the latent space of VAE, in addition to predicting returns. The experiments on the real stock market data demonstrate the effectiveness of FactorVAE, which outperforms various baseline methods.

### 1.2. Visualization
**1.2.1. Brief illustration**
<div align=center>
    <img src="illustration.png" width="40%" height="40%">
</div>

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
- `E`: size of epochs (arbitrary)
- `B`: size of batches (arbitrary)
- `N`: size of stocks (arbitrary)
- `T`: size of time periods (arbitrary)
- `C`: size of characteristics
- `H`: size of hidden features
- `M`: size of portfolios
- `K`: size of factors
### 2.2. Tensor (variable)
- `x`: characteristics, B\*N\*T\*C
- `y`: stock returns, B\*N
- `e`: hidden features, B\*N\*H
- `y_p`: portfolio returns, B\*M
- `z_post`: posterior latent factor returns, B\*K 
- `z_prior`: prior latent factor returns, B\*K 
- `alpha`: idiosyncratic returns, B\*N
- `beta`: factor exposures, B\*N\*K
- `y_hat`: reconstructed stock returns, B\*N
- `mu_post`: mean vector of `z_post`, B\*K
- `sigma_post`: std vector of `z_post`, B\*K
- `mu_prior`: mean vector of `z_prior`, B\*K
- `sigma_prior`: std vector of `z_prior`, B\*K
- `mu_alpha`: mean vector of `alpha`, B\*N
- `sigma_alpha`: std vector of `alpha`, B\*N
- `mu_y`: mean vector of `y_hat`, B\*N
- `Sigma_y`: cov matrix of `y_hat`, B\*N\*N
### 2.3. Distribution
- $p_{\theta}(y|x)$: true label, likelihood
- $q_{\phi}(z|x,y)$: encoder output, posterior distribution
- $q_{\phi}(z|x)$: predictor output, prior distribution
- $p_{\theta}(y|x,z)$: decoder output, conditional likelihood
- $f_{\phi,\theta}(y|x)$: predicted label, predicted likelihood


## 3. Module
### 3.1. \_\_init\_\_.py
`FactorVAE` (*top-level encapsulated class*) extracts effective factors from noisy market data. First, it obtain optimal factors by an encoder-decoder architecture with access to future data, and then train a factor predictor according a prior-posterior learning method, which extracts factors to approximate the optimal factors.

`PipelineFactorVAE` as a subclass of `Pipeline`, automates the training, validation, and testing process of the `FactorVAE`.

`loss_fn_vae()` gets the loss value of the model.

`bcorr()` calculates batch correlation between two vectors.

`gaussian_kld()` calculates KL divergence of two multivariate independent Gaussian distributions.


### 3.2. data.py
`RollingDataset` yields characteristics `x` in R^{N\*T\*C}, and future stock returns `y` in R^{N} in each iteration.

`change_freq()` changes the frequency of the panel data.

`shift_ret()` shifts returns to the previous period then drop NaN.

`wins_ret()` winsorizes returns.

`assign_label()` assigns labels based on different quantiles of the returns.

### 3.3. pipeline.py
`Pipeline` gives a general machine learning pipeline which automates the model training, validation, and testing process.

`set_seeds()` sets random seeds for all random processes.

### 3.4. backtesting.py
`Backtester` backtests cross-sectinal strategies, by the following procedure: 1) `factor` $\rightarrow$ `pos`; 2) `pos` + `ret` $\rightarrow$ `strat_ret`; 3) `strat_ret` $\rightarrow$ `nv`.

### 3.5. feature_extractor.py
`FeatureExtractor` extracts stocks hidden features `e` from the historical sequential characteristics `x`.

### 3.6. factor_encoder.py
`FactorEncoder` extracts posterior factors `z_post`, a random vector following the independent Gaussian distribution, which can be described by the mean `mu_post` and the standard deviation `sigma_post`, from hidden features `e` and stock returns `y`.

`PortfolioLayer` dynamically re-weights the portfolios on the basis of stock hidden features `e`.

`MappingLayer` maps `y_p` as the portfolio returns to the distribution of posterior factor returns `z_post`.

### 3.7. factor_decoder.py
`FactorDecoder` calculates predicted stock returns `y_hat`, a random vector following the Gaussian distribution, which can be described by the mean `mu_y` and the covariance matrix `Sigma_y`, from distribution parameters of factor returns `z` (could be `z_post` or `z_prior`) and hidden features `e`.

`AlphaLayer` outputs idiosyncratic returns `alpha` from the hidden features `e`.

`BetaLayer` calculates factor exposures `beta` from hidden feautres `e`.

### 3.8. factor_predictor.py
`FactorPredictor` extracts prior factor returns `z_prior`, a random vector following the independent Gaussian distribution, which can be described by the mean `mu_prior` and the standard deviation `sigma_prior`, from hidden features `e`.

`MultiheadGlobalAttention` implements a specific type of multi-head global attention.


## 4. Example
```Python
import numpy as np
import pandas as pd
import torch
from facvae import FactorVAE, PipelineFactorVAE
from facvae.backtesting import Backtester
from facvae.data import RollingDataset, change_freq, shift_ret, wins_ret
from facvae.pipeline import set_seeds
from torch.utils.data import DataLoader

if __name__ == "__main__":
    # constants
    E = 20
    B = 32
    N = 74
    T = 5
    C = 28
    H = 16
    M = 24
    K = 8
    h_prior_size = 32
    h_alpha_size = 16
    h_prior_size = 16
    partition = [0.8, 0.1, 0.1]
    lr = 0.01
    gamma = 2.0
    lmd = 0.5
    max_grad = None
    freq = "d"
    start_date = "2015-01-01"
    end_date = "2023-01-01"
    top_pct = 0.1
    wins_thresh = 0.25
    verbose_freq = None

    # data
    df = pd.read_pickle("df.pickle")
    df = df.loc[start_date:end_date]
    df = change_freq(df, freq)
    df = shift_ret(df)
    df = wins_ret(df, wins_thresh)

    # pipeline
    ds = RollingDataset(df, "ret", T)
    loss_kwargs = {"gamma": gamma, "lmd": lmd}
    eval_kwargs = {"df": df, "top_pct": top_pct}
    pl = PipelineFactorVAE(ds, partition, B, loss_kwargs, eval_kwargs)

    # search
    for i in range(2000):
        set_seeds(i)
        print("seed:", i)
        fv = FactorVAE(C, H, M, K, h_prior_size, h_alpha_size, h_prior_size)
        pl.train(fv, lr, E, max_grad, verbose_freq=verbose_freq)
        sr_valid = pl.validate(fv)
        sr_test = pl.test(fv)
        print(sr_valid)
        print(sr_test)
        if sr_valid > 1.5 and sr_test > 1.5:
            torch.save(fv, dir_result + f"model_{i}")

    # check
    model = torch.load(dir_result + "model_xxx")
    dl = DataLoader(ds, len(ds))
    x, y = next(iter(dl))
    mu_y, Sigma_y = model.predict(x)
    mu_y = mu_y.flatten().cpu().numpy()
    df["factor"] = np.nan
    df.iloc[-len(mu_y):, -1] = mu_y
    print(df)

    bt = Backtester("factor", top_pct=top_pct).feed(df).run()
    bt.report()
```