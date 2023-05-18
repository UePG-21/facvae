import pandas as pd
from facvae import FactorVAE
from facvae.data import change_freq, shift_ret, get_dataloaders
from facvae.pipeline import loss_func_vae, test_model, train_model
from facvae.backtesting import Backtester


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
    E = 25
    B = 16
    N = 74
    T = 20
    C = 28
    H = 10
    M = 32
    K = 8
    h_prior_size = 16
    h_alpha_size = 16
    h_prior_size = 16
    partition = [0.7, 0.2, 0.1]
    lr = 0.0001
    lmd = 1.0
    max_grad = 1.0

    # model
    fv = FactorVAE(C, H, M, K, h_prior_size, h_alpha_size, h_prior_size).to("cuda")

    # data
    df = pd.read_pickle(dir_data + "df_l1_comb.pickle")
    df.sort_index(level=(0, 1), inplace=True)
    df = shift_ret(df)
    dl_train, dl_valid, dl_test = get_dataloaders(df, T, B, partition)

    # train
    train_model(fv, dl_train, lr, E, lmd=lmd, max_grad=max_grad)

    # test
    loss = test_model(fv, dl_test)
    print("out-of-sample loss:", loss)

    # predict
    x, y = next(iter(dl_test))
    mu_y, Sigma_y = fv.predict(x)

    # backtest
    len_test = next(iter(dl_test))[1].shape[0]
    idx = pd.IndexSlice[df.index.get_level_values(0).unique()[-len_test:], :]
    df = df.loc[idx]

    df["factor"] = mu_y.flatten().cpu().numpy()
    df = df[["factor", "ret"]]

    bt = Backtester("factor", top_pct=0.25).feed(df).run()
    bt.report()
