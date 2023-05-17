import pandas as pd
from facvae import FactorVAE
from facvae.data import get_dataloaders
from facvae.pipeline import test_model, train_model

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

    # model
    fv = FactorVAE(C, H, M, K, h_prior_size, h_alpha_size, h_prior_size)

    # data
    df = pd.read_pickle(dir_data + "df_l1_comb.pickle")
    dl_train, dl_valid, dl_test = get_dataloaders(df, T, B, partition)

    # train
    train_model(fv, dl_train, lr, E, 1.0)

    # test
    loss = test_model(fv, dl_test)

    # predict
    x, y = next(iter(dl_test))
    mu_y, Sigma_y = fv.predict(x)
    
    print(Sigma_y)
    print(mu_y)
    print(y)
    print(((mu_y - y) ** 2).mean())
