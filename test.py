import pandas as pd
from facvae import FactorVAE
from facvae.backtesting import Backtester
from facvae.data import change_freq, get_dataloaders, shift_ret
from facvae.pipeline import test_model, train_model

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
    print(df)
    dl_train, dl_valid, dl_test = get_dataloaders(df, "ret", T, B, partition)

    # train
    train_model(fv, dl_train, lr, E, lmd, max_grad)

    # test
    loss = test_model(fv, dl_test)
    print("out-of-sample loss:", loss)

    # predict
    x, y = next(iter(dl_test))
    mu_y, Sigma_y = fv.predict(x)

    # backtest
    len_test = next(iter(dl_test))[1].shape[0]
    idx = pd.IndexSlice[df.index.get_level_values(0).unique()[-len_test:], :]
    df_test = df.loc[idx]

    df_test["factor"] = mu_y.flatten().cpu().numpy()

    bt = Backtester("factor", cost=0.0, top_pct=top_pct).feed(df_test).run()
    bt.report()
