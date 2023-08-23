from models.Trans import *
import matplotlib.pyplot as plt

if __name__ == "__main__":
    args = {}
    args["input_size"] = 82
    args["trans_hidden_size"] = 512
    args["trans_kernel_size"] = 3
    args["seq_len"] = 10
    args["trans_n_heads"] = 8
    args["trans_n_layers"] = 3
    args["out_size"] = 1
    args["normalize"] = True
    args["window_size"] = 10
    args["y_dim"] = 1
    args["validation_ratio"] = 0.2
    args["batch_size"] = 256
    args["hidden_size"] = 128
    args["epochs"] = 100
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    raw_data =  pd.read_csv('/dunghc/time_series/nasdaq100_padding.csv').values
    print("raw_data: ", raw_data.shape)

    scale = StandardScaler().fit(raw_data)

    if args["normalize"]:
        data = scale.transform(raw_data)
    else:
        data = raw_data

    train_X, train_y, val_X, val_y = split_data(data, args["window_size"], args["y_dim"], args["validation_ratio"])
    train_X, train_y = torch.tensor(train_X, dtype=torch.float32), torch.tensor(train_y, dtype=torch.float32)
    val_X, val_y = torch.tensor(val_X[:4000], dtype=torch.float32), torch.tensor(val_y[:4000], dtype=torch.float32)
    print('train_X, train_y :', train_X.shape, train_y.shape)
    print('  val_X,   val_y :', val_X.shape, val_y.shape)
    val_X = val_X.to(device)

    model = Trans(args)
    model = model.to(device)
    model.load_state_dict(torch.load("/dunghc/deep-time-series/LSTM_v5.pth"))

    predicts = model(val_X)
    print(predicts.shape)
    print(val_y.shape)

    plt.figure(figsize=(40, 20))
    plt.plot(np.squeeze(predicts.detach().cpu().numpy()), "r", np.squeeze(val_y.detach().cpu().numpy()), "blue")
    plt.savefig("infer4.png")

    mse_val = mean_squared_error(np.squeeze(predicts.detach().cpu().numpy()), np.squeeze(val_y.detach().cpu().numpy()))
    mae_val = mean_absolute_error(np.squeeze(predicts.detach().cpu().numpy()), np.squeeze(val_y.detach().cpu().numpy()))
    mape_val = mean_absolute_percentage_error(np.squeeze(predicts.detach().cpu().numpy()), np.squeeze(val_y.detach().cpu().numpy()))

    print("mse_val: ", mse_val)
    print("mae_val: ", mae_val)
    print("mape_val: ", mape_val)
