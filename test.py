import torch
from facvae import FactorVAE, loss_func

if __name__ == "__main__":
    import numpy as np
    import pandas as pd

    dir_main = "E:/Others/Programming/py_vscode/projects/signal_mixing/"
    dir_code = dir_main + "code/"
    dir_data = dir_main + "data/"
    dir_config = dir_main + "config/"
    dir_result = dir_main + "result/"

    H = 5
    N = 74
    T = 20
    C = 10
    M = 18
    K = 9
    h_proj_size = 6
    h_alpha_size = 6
    h_prior_size = 6

    df_l1 = pd.read_pickle(dir_data + "df_l1_comb.pickle").sort_index(level=0)
    x = (
        df_l1.loc[:"2015-01-30", :"maxretpayoff"]
        .values.reshape(T, N, C)
        .transpose(1, 0, 2)
        .reshape(1, N, T, C)
    )
    x = torch.tensor(x).float()
    y = df_l1.loc["2015-01-30", "ret"].values.reshape(1, N)
    y = torch.tensor(y).float()

    fv = FactorVAE(C, H, M, K, h_prior_size, h_alpha_size, h_prior_size)
    loss = loss_func(y, *fv(x, y))
    print(loss)
    print(fv(x, y))
    fv.predict(x)
