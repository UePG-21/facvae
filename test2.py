import pandas as pd
from facvae import FactorVAE
from facvae.backtesting import Backtester
from facvae.data import change_freq, get_dataloaders, shift_ret, assign_label
from facvae.pipeline import loss_func_vae, test_model, train_model

"""
total_norm = 0.0
for p in model.parameters():
    param_norm = p.grad.data.norm(2)
    total_norm += param_norm.item() ** 2
    total_norm = total_norm ** (1.0 / 2)
print("after", total_norm)

"""

if __name__ == "__main__":
    import numpy as np
    import pandas as pd
    import torch
    import torch.nn as nn

    # constants
    E = 100
    B = 16
    N = 74
    T = 5
    C = 1
    H = 64
    M = 32
    K = 32
    h_prior_size = 16
    h_alpha_size = 16
    h_prior_size = 16
    partition = [0.72, 0.0, 0.25]
    lr = 1e-3
    lmd = 0.0
    max_grad = 1
    freq = "d"
    quantiles = [0.2, 0.4]
    start_date = "2015-01-01"
    end_date = "2023-01-01"
    top_pct = 0.05

    # model
    fv = FactorVAE(C, H, M, K, h_prior_size, h_alpha_size, h_prior_size).to("cuda")

    # data
    df = pd.read_csv(r"E:\Others\Programming\py_vscode\modules\machine_learning\facvae\ALL_DATA.csv", parse_dates=["filing_date"])
    df.set_index(["filing_date", "ticker"], inplace=True)
    df.drop(["sector", "filing_date.1"], axis=1, inplace=True)
    df = pd.DataFrame(df.values)
    df.rename(columns={df.shape[1]-1:"ret"}, inplace=True)
    df = (df - df.mean()) / df.std()
    # df["ret"] = np.random.randn(len(df))
    print(df)


    X, y = df.iloc[:2000, :-1], df.iloc[:2000, -1]
    
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_squared_error
    from sklearn.tree import DecisionTreeRegressor

    rfr = RandomForestRegressor(n_estimators=100)
    dt = DecisionTreeRegressor(max_depth=6)
    model = dt
    model.fit(X, y)
    print(y)
    y_pred = model.predict(X)
    print(y_pred)
    print(mean_squared_error(y, y_pred))

    