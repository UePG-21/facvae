import pandas as pd

from facvae import FactorVAE
from facvae.backtesting import Backtester
from facvae.data import change_freq, get_dataloaders, shift_ret
from facvae.pipeline import test_model, train_model

"""
total_norm = 0.0
for p in model.parameters():
    param_norm = p.grad.data.norm(2)
    total_norm += param_norm.item() ** 2
    total_norm = total_norm ** (1.0 / 2)
print("after", total_norm)

"""

if __name__ == "__main__":
    # directories
    dir_main = "E:/Others/Programming/py_vscode/projects/signal_mixing/"
    dir_code = dir_main + "code/"
    dir_data = dir_main + "data/"
    dir_config = dir_main + "config/"
    dir_result = dir_main + "result/"

    # constants
    E = 20
    B = 8
    N = 74
    T = 20
    C = 283
    H = 64
    M = 24
    K = 16
    h_prior_size = 128
    h_alpha_size = 32
    h_prior_size = 32
    partition = [0.8, 0.0, 0.2]
    lr = 1e-4
    lmd = 1
    max_grad = None
    freq = "d"
    quantiles = [0.2, 0.4]
    start_date = "2015-01-01"
    end_date = "2023-01-01"
    top_pct = 0.1

    # model
    fv = FactorVAE(C, H, M, K, h_prior_size, h_alpha_size, h_prior_size).to("cuda")

    # data
    df = pd.read_pickle(dir_data + "df_l1_comb.pickle")
    df.sort_index(level=(0, 1), inplace=True)
    df = df.loc[start_date:end_date]
    df = change_freq(df, freq)
    df = shift_ret(df)
    # df, rets = assign_label(df, quantiles)
    print(df)
    dl_train, dl_valid, dl_test = get_dataloaders(df, "ret", T, B, partition)

    # train
    train_model(fv, dl_train, lr, E, lmd=lmd, max_grad=max_grad)

    # test
    loss = test_model(fv, dl_test)
    print("out-of-sample loss:", loss)

    # predict
    x, y = next(iter(dl_train))
    mu_y, Sigma_y, _, _, _, _ = fv(x, y)

    # print("Sigma_y")
    # print(Sigma_y)
    print("mu_y")
    print(mu_y)
    print("y")
    print(y)

    x, y = next(iter(dl_test))
    mu_y, Sigma_y = fv.predict(x)

    # print("Sigma_y")
    # print(Sigma_y)
    print("mu_y")
    print(mu_y)
    print("y")
    print(y)

    # backtest
    len_test = next(iter(dl_test))[1].shape[0]
    idx = pd.IndexSlice[df.index.get_level_values(0).unique()[-len_test:], :]
    df_test = df.loc[idx]

    df_test["factor"] = mu_y.flatten().cpu().numpy()

    bt = Backtester("factor", cost=0.0, top_pct=top_pct).feed(df_test).run()
    bt.report()
