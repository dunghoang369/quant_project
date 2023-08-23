import math
import torch
import tqdm
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.utils as utils
import torch.optim as optim
import torch.nn.functional as F
from models.drnn import DRNN
from torch.utils.data import TensorDataset
from get_rolling_window import rolling_window
import torch.optim.lr_scheduler as lr_scheduler
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error

def get_labels_from_features(features, window_size, y_dim):
    return features[window_size - 1:, -y_dim:]


def split_by_ratio(features, validation_ratio):
    length = len(features)
    validation_length = int(validation_ratio * length)

    return features[:-validation_length], features[-validation_length:]


def split_data(
    data,
    window_size,
    y_dim,
    validation_ratio
):
    train_data, val_data = split_by_ratio(data, validation_ratio)

    train_f, train_l = rolling_window(
        train_data, window_size, 1
    ), get_labels_from_features(train_data, window_size, y_dim)

    val_f, val_l = rolling_window(
        val_data, window_size, 1
    ), get_labels_from_features(val_data, window_size, y_dim)

    return train_f, train_l, val_f, val_l


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # print(x.shape, self.pe.shape)
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class Trans(nn.Module):

    def __init__(self, args):
        super(Trans, self).__init__()
        input_size = args["input_size"]
        trans_hidden_size = args["trans_hidden_size"]
        trans_kernel_size = args["trans_kernel_size"]
        seq_len = args["seq_len"]
        n_trans_head = args["trans_n_heads"]
        trans_n_layers = args["trans_n_layers"]
        out_size = args["out_size"]
        hidden_size = args["hidden_size"]

        self.conv = nn.Conv1d(input_size, trans_hidden_size, kernel_size=trans_kernel_size)
        self.pos_encoder = PositionalEncoding(trans_hidden_size, max_len=seq_len)
        self.transformer_layer = nn.TransformerEncoderLayer(d_model=trans_hidden_size, nhead=n_trans_head)
        self.transformer = nn.TransformerEncoder(self.transformer_layer, num_layers=trans_n_layers)
        self.lstm = nn.LSTM(trans_hidden_size, hidden_size, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hidden_size, out_size)
        self.kernel_size = trans_kernel_size
        self.drnn = DRNN(trans_hidden_size, hidden_size, 2, 0, "RNN")

    def forward(self, x):
        x = x.transpose(1, 2)
        x = F.pad(x, (self.kernel_size-1,0))
        x = self.conv(x).permute(2, 0, 1)
        x = self.pos_encoder(x)
        x = self.transformer(x).transpose(0, 1)[:, -1:]
        x = torch.permute(x, (1, 0, 2))
        # print("1:", x.shape)
        x = self.drnn(x)[1][-1]
        # print(x.shape)
        # print("x:", len(x))
        # DFDGDFDFD
        # print("2:", x.shape)
        # x = self.lstm(x)[1][0]
        # print(x.shape)
        # print("2: ", x.shape)
        output = self.fc(x)
        output = torch.permute(output, (1, 0, 2))
        # print("output:", output.shape)
        # output = torch.mean(output, 1)
        output = torch.squeeze(output, 1)
        # print("output:", output.shape)
        # print("output: ", output.shape)
        return output

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
    args["epochs"] = 50
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
    val_X, val_y = torch.tensor(val_X, dtype=torch.float32), torch.tensor(val_y, dtype=torch.float32)
    print('train_X, train_y :', train_X.shape, train_y.shape)
    print('  val_X,   val_y :', val_X.shape, val_y.shape)

    train_dataset = TensorDataset(train_X, train_y)
    valid_dataset = TensorDataset(val_X, val_y)
    train_loader = utils.data.DataLoader(train_dataset, batch_size=args["batch_size"], shuffle=True)
    valid_loader = utils.data.DataLoader(valid_dataset, batch_size=args["batch_size"], shuffle=False)

    model = Trans(args)
    model = model.to(device)

    params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.Adam(model.parameters(), lr=1e-5)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, "min", factor=0.5, patience=3, verbose=True)
    criterion = nn.MSELoss()

    best_mse = 0.0

    print("Start training ...\n" + "==================\n")
    for epoch in range(1, args["epochs"] + 1):
        head = 'epoch {:2}/{:2}'.format(epoch, args["epochs"])
        print(head + '\n' + '-'*(len(head)))

        model.train()
        loss = 0.0
        mse = 0.0
        mae = 0.0
        mape = 0.0

        for input_ids, labels in tqdm.tqdm(train_loader):
            input_ids, labels = input_ids.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(input_ids)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            loss += loss.item() * len(input_ids)
            mse += mean_squared_error(np.squeeze(outputs.detach().cpu().numpy()), np.squeeze(labels.detach().cpu().numpy()))
            mae += mean_absolute_error(np.squeeze(outputs.detach().cpu().numpy()), np.squeeze(labels.detach().cpu().numpy()))
            mape += mean_absolute_percentage_error(np.squeeze(outputs.detach().cpu().numpy()), np.squeeze(labels.detach().cpu().numpy()))
        
        epoch_loss = loss / len(train_dataset)
        epoch_mse = mse / len(train_dataset)
        epoch_mae = mae / len(train_dataset)
        epoch_mape = mape / len(train_dataset)
        print("{} - loss: {:.8f} - mse: {:.8f} - mae: {:.8f} - mape: {:.8f}".format("train", epoch_loss, epoch_mse, epoch_mae, epoch_mape))

        with torch.no_grad():
            model.eval()

            loss_val = 0.0
            mse_val = 0.0
            mae_val = 0.0
            mape_val = 0.0

            for input_ids, labels in tqdm.tqdm(valid_loader):
                input_ids, labels = input_ids.to(device), labels.float().to(device)
                outputs = model(x=input_ids)
                loss = criterion(outputs, labels)

                loss_val += loss.item() * len(input_ids)
                mse_val += mean_squared_error(np.squeeze(outputs.detach().cpu().numpy()), np.squeeze(labels.detach().cpu().numpy()))
                mae_val += mean_absolute_error(np.squeeze(outputs.detach().cpu().numpy()), np.squeeze(labels.detach().cpu().numpy()))
                mape_val += mean_absolute_percentage_error(np.squeeze(outputs.detach().cpu().numpy()), np.squeeze(labels.detach().cpu().numpy()))

            epoch_loss_val = loss_val / len(valid_dataset)
            epoch_mse_val = mse_val / len(valid_dataset)
            epoch_mae_val = mae_val / len(valid_dataset)
            epoch_mape_val = mape_val / len(valid_dataset)
            print("{} - loss: {:.8f} - mse: {:.8f} - mae: {:.8f} - mape: {:.8f}".format("valid", epoch_loss_val, epoch_mse_val, epoch_mae_val, epoch_mape_val))

            if best_mse < epoch_mse_val:
                best_mse = epoch_mse_val
                torch.save(model.state_dict(), "/dunghc/deep-time-series/LSTM_v5.pth")
        
    # train_test = train_X[:32]
    # print(train_test.shape)
    # out = model(train_test)
    
    # print(out.shape)


# /home/dunghc/time_series/nasdaq100_padding.csv