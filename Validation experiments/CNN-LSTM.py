import time
import tqdm
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from dataset import MLZDataset
from metrics import evaluate_PICP_WS_PL
from torch.utils.data import DataLoader, Dataset

class CNN_LSTM(nn.Module):
    def __init__(self,window, n_zone, n_kernels,drop_prob=0.1):
        super(CNN_LSTM, self).__init__()
        self.n_kernels =n_kernels
        self.window = window
        self.n_zone = n_zone
        self.drop_prob = drop_prob
        self.conv1= nn.Conv2d(1, n_kernels*2, (window, 1))
        self.conv2 = nn.Conv2d(1, n_kernels, (window, 1))
        self.pooling1 = nn.AdaptiveMaxPool2d((1, n_zone))
        self.lstm1 = nn.LSTM(input_size=n_kernels, hidden_size=n_kernels, num_layers=1, batch_first=True)
        self.linear1 = nn.Linear(n_kernels,n_kernels)
    def forward(self, x):
        x = x.view(-1, 1, self.window, self.n_zone)
        x2 = F.relu(self.conv1(x))
        x2 = x2.transpose(1,2)
        x2 = F.relu(self.conv2(x2))
        x2 = self.pooling1(x2)
        x2 = x2.transpose(3,1)
        x2 = x2.reshape(-1,1,self.n_kernels)
        _, (x3, _) = self.lstm1(x2)
        x4 = nn.Dropout(p=self.drop_prob)(x3)
        x4 = x4.reshape(1, -1, self.n_zone, self.n_kernels)
        x4 = self.linear1(x4)
        x5 = torch.squeeze(x4)
        return x5,

class QCNNLSTM(nn.Module):
    def __init__(self, batch_size=64, window=24, n_zone=20, n_kernels=64, drop_prob=0.1,
                 learning_rate=0.005, criterion='pinball_loss',lag=7,
                 device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                 quantiles=torch.tensor(np.array([0.95, 0.05])),
                 ):
        super(QCNNLSTM, self).__init__()

        # parameters from dataset
        self.lag = lag
        self.window = window
        self.n_zone = n_zone

        # hyperparameters of model
        self.total_loss = 0.
        self.device = device
        self.n_kernels = n_kernels
        self.drop_prob = drop_prob
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.quantiles = quantiles.to(device)
        self.n_quantile = self.quantiles.numel()
        assert self.n_quantile == 2

        # build model
        self.__build_model()
        self.to(device)

    def __build_model(self):
        self.cnnlstm = CNN_LSTM( window=self.window, n_zone=self.n_zone, n_kernels=self.n_kernels, drop_prob=self.drop_prob)
        self.dropout = nn.Dropout(p=self.drop_prob)
        self.active_func = nn.Tanh()
        self.W_output = nn.Linear(self.n_kernels, 2)

    def forward(self, x):
        cnnlstm_output, *_ = self.cnnlstm(x)
        # dropout
        drop_output = self.dropout(cnnlstm_output)
        # active function
        output = self.active_func(self.W_output(drop_output))
        return output

    def loss(self, labels, predictions):
        labels = labels * torch.ones((self.n_quantile)).to(self.device)
        e = predictions - labels
        return torch.sum(torch.mean(torch.maximum(self.quantiles * e, (self.quantiles - 1) * e), axis=0))

    def set_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
        self.optimizer, self.scheduler = optimizer, scheduler

    def training_step(self, data_batch):
        self.train()
        x, y = data_batch
        x, y = x.to(self.device), y.to(self.device)
        self.optimizer.zero_grad()
        y_hat = self.forward(x)
        loss = self.loss(y, y_hat)
        loss.backward()
        self.optimizer.step()
        self.total_loss += loss.cpu().item()

    def EVALUATE(self, eval_dataLoader):
        total_loss = 0.
        self.eval()
        with torch.no_grad():
            for data_batch in eval_dataLoader:
                x, y = data_batch
                x, y = x.to(self.device), y.to(self.device)
                y_hat = self.forward(x)
                loss = self.loss(y, y_hat)
                total_loss += loss.cpu().item()
        eval_loss = total_loss / len(eval_dataLoader)
        return eval_loss, y_hat, y

    def TRAIN(self, train_dataLoader, eval_dataLoader, test_dataLoader, epochs=1):
        self.set_optimizers()
        self.losses, self.val_losses = [], []
        for epoch in range(epochs):
            self.total_loss = 0.
            try:
                pbar = tqdm.tqdm(range(len(train_dataLoader)), ncols=75, ascii=True)
                for i, data_batch in enumerate(train_dataLoader):
                    self.training_step(data_batch)
                    pbar.set_description("Loss:{:>7.4f}".format(self.total_loss / (i + 1)))
                    pbar.update(1)
            finally:
                pbar.close()
            loss = self.total_loss / len(train_dataLoader)
            val_loss, y_hat, y = self.EVALUATE(eval_dataLoader)
            self.losses.append(loss)
            self.val_losses.append(val_loss)
            print('| epoch {:3d} | loss {:8.6f} | val_loss {:8.6f} |'.format(epoch + 1, loss, val_loss))
        # evaluate on the test set
        test_loss, test_y_hat, test_y = self.EVALUATE(test_dataLoader)
        self.y = np.array(test_y.cpu().reshape((-1, 1)))
        self.y_hat = np.array(test_y_hat.cpu().reshape((-1, 2)))
        PICP, WS, QS, MPIW = evaluate_PICP_WS_PL(self.y_hat, self.y, np.array(model.quantiles.cpu()))
        return PICP, WS, MPIW, test_loss

zone_set = [5, 10, 15, 20]
lag_set = [3, 5, 7]
year = [2012, 2017]
for i in range(len(year)):
    for j in range(len(zone_set)):
        for k in range(len(lag_set)):
            batch_size = 512
            torch.manual_seed(101)
            torch.cuda.manual_seed(101)
            np.random.seed(101)
            model = QCNNLSTM(batch_size=batch_size, n_zone=zone_set[j], window=24, n_kernels=64,lag=lag_set[k])
            rdt1 = MLZDataset(window=model.window, lag=model.lag, data_name=str(year[i]) + '_' + str(model.n_zone),
                              set_type='train', data_dir='./data')
            rdv1 = MLZDataset(window=model.window, lag=model.lag, data_name=str(year[i]) + '_' + str(model.n_zone),
                              set_type='validation', data_dir='./data')
            rdT1 = MLZDataset(window=model.window, lag=model.lag, data_name=str(year[i]) + '_' + str(model.n_zone),
                              set_type='test', data_dir='./data')
            train_dataLoader = DataLoader(dataset=rdt1, batch_size=batch_size, shuffle=True)
            eval_dataLoader = DataLoader(dataset=rdv1, batch_size=batch_size, shuffle=True)
            test_dataLoader = DataLoader(dataset=rdT1, batch_size=batch_size, shuffle=True)
            PICP, WS, MPIW, test_loss = model.TRAIN(train_dataLoader, eval_dataLoader, test_dataLoader, 1)
