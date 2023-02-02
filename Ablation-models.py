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


class Global_CNN(nn.Module):
    def __init__(
            self,
            window, n_zone, n_kernels, drop_prob=0.1):
        super(Global_CNN, self).__init__()
        self.window = window
        self.n_zone = n_zone
        self.drop_prob = drop_prob
        self.conv2 = nn.Conv2d(1, n_kernels, (window, 1))

    def forward(self, x):
        x = x.view(-1, 1, self.window, self.n_zone)
        x2 = F.relu(self.conv2(x))
        x2 = nn.Dropout(p=self.drop_prob)(x2)
        x = torch.squeeze(x2, 2)
        x = torch.transpose(x, 1, 2)
        return x,


class Local_CNN(nn.Module):
    def __init__(
            self,
            window, n_zone, n_kernels, drop_prob=0.1):
        super(Local_CNN, self).__init__()
        self.window = window
        self.n_zone = n_zone
        self.drop_prob = drop_prob
        self.conv1 = nn.Conv2d(1, n_kernels, (window, 1))
        self.pooling1 = nn.AdaptiveMaxPool2d((1, n_zone))

    def forward(self, x):
        x = x.view(-1, 1, self.window, self.n_zone)
        x1 = F.relu(self.conv1(x))
        x1 = self.pooling1(x1)
        x1 = nn.Dropout(p=self.drop_prob)(x1)
        x = torch.squeeze(x1, 2)
        x = torch.transpose(x, 1, 2)
        return x,


class AMPO(nn.Module):
    def __init__(self, in_size):
        super(AMPO, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.hidden_size = in_size * 2
        self.project = nn.Sequential(
            nn.Linear(in_size, self.hidden_size),  
            nn.Linear(self.hidden_size, self.hidden_size * 2), 
            nn.Tanh(),  # tanh(W*Z+b)
            nn.Linear(self.hidden_size * 2, 1, bias=False),  
        )

    def forward(self, z): 
        w1 = self.max_pool(z)  
        w = self.project(z)  
        w2 = self.avg_pool(z)
        w3 = torch.cat([w1, w2], dim=2)
        beta = torch.softmax(w, dim=1)  
        w3 = torch.softmax(w3, dim=1)
        beta = beta + w3
        out = (beta * z).sum(2)  
        return torch.squeeze(out)


class AR(nn.Module):
    def __init__(self, window):
        super(AR, self).__init__()
        self.linear = nn.Linear(window, 1)

    def forward(self, x):
        x = torch.transpose(x, 1, 2)
        x = self.linear(x)
        x = torch.transpose(x, 1, 2)
        return x

class Model(nn.Module):
    def __init__(self, batch_size=64, window=32, n_zone=20, n_kernels=32,drop_prob=0.1,learning_rate=0.005,
                 lag=7,device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                 quantiles=torch.tensor(np.array([0.95, 0.05]))):
        super(Model, self).__init__()

        self.batch_size = batch_size
        self.device = device
        self.quantiles = quantiles.to(device)
        self.n_quantile = self.quantiles.numel()
        assert self.n_quantile == 2
        # parameters from dataset
        self.window = window
        self.lag = lag
        self.n_zone = n_zone
        self.n_kernels = n_kernels
        # hyperparameters of model
        self.drop_prob = drop_prob
        self.learning_rate = learning_rate
        self.total_loss = 0.
        # build model
        self.__build_model()
        self.to(device)

    #CNN
    '''
    def __build_model(self):
        self.lCNN = Local_CNN(
            window=self.window, n_zone=self.n_zone, n_kernels=self.n_kernels,
            drop_prob=self.drop_prob)

        self.ar = AR(window=self.window)
        self.dropout = nn.Dropout(p=self.drop_prob)
        self.active_func = nn.Tanh()
        self.W_output = nn.Linear(self.n_kernels, 2)

    def forward(self, x):
        #Nonlinear component
        lCNN_output, *_ = self.lCNN(x)
        # dropout
        drop_output = self.dropout(lCNN_output)
        # active function
        active_output = self.active_func(self.W_output(drop_output))
        nonlinear_output = torch.transpose(active_output, 1, 2)
        output = torch.transpose(nonlinear_output, 1, 2)
        return output
    '''

    #CNN_AR
    '''
    def __build_model(self):
        self.lCNN = Local_CNN(
            window=self.window, n_zone=self.n_zone, n_kernels=self.n_kernels,
            drop_prob=self.drop_prob)

        self.ar = AR(window=self.window)
        self.dropout = nn.Dropout(p=self.drop_prob)
        self.active_func = nn.Tanh()
        self.W_output = nn.Linear(self.n_kernels, 2)
        self.ar1 = AR(window=self.window)
        self.ar2 = AR(window=self.window)

    def forward(self, x):
        #Nonlinear component
        lCNN_output, *_ = self.lCNN(x)  
        # dropout
        drop_output = self.dropout(lCNN_output)
        # active function
        active_output = self.active_func(self.W_output(drop_output))
        nonlinear_output = torch.transpose(active_output, 1, 2)
        #Linear component
        ar_output1 = self.ar1(x)
        ar_output2 = self.ar2(x)
        linear_output = torch.cat((ar_output1, ar_output2), 1)
        #Integration
        output = nonlinear_output + linear_output
        output = torch.transpose(output, 1, 2)
        return output
    '''

    #GDCNN_AR
    '''
    def __build_model(self):
        self.gCNN = Global_CNN(
            window=self.window, n_zone=self.n_zone, n_kernels=self.n_kernels,
            drop_prob=self.drop_prob)
        self.lCNN = Local_CNN(
            window=self.window, n_zone=self.n_zone, n_kernels=self.n_kernels,
            drop_prob=self.drop_prob)

        self.ar = AR(window=self.window)
        self.dropout = nn.Dropout(p=self.drop_prob)
        self.active_func = nn.Tanh()
        self.W_output1 = nn.Linear(self.n_kernels, 1)
        self.W_output2 = nn.Linear(self.n_kernels, 1)
        self.W_output3 = nn.Linear(self.n_kernels, 2)
        self.ar1 = AR(window=self.window)
        self.ar2 = AR(window=self.window)

    def forward(self, x):
        #Nonlinear component
        # Dual CNN
        gCNN_output, *_ = self.gCNN(x)  
        lCNN_output, *_ = self.lCNN(x) 
        # gated mechanisim
        gate_output = torch.tanh(gCNN_output) * torch.sigmoid(lCNN_output)
        # dropout
        drop_output = self.dropout(gate_output)
        # active function
        active_output1 = self.active_func(self.W_output1(drop_output))
        active_output2 = self.active_func(self.W_output2(drop_output))
        gdcnn_output = torch.stack([active_output1, active_output2], dim=2)
        nolinear_output = torch.transpose(torch.squeeze(gdcnn_output), 1, 2)    
        #Linear component
        ar_output1 = self.ar1(x)
        ar_output2 = self.ar2(x)
        linear_output = torch.cat((ar_output1, ar_output2), 1)
        #Integration
        output = nonlinear_output + linear_output
        output = torch.transpose(output, 1, 2)
        return output
    '''

    #GDCNN_AR_AMPO
    '''
    def __build_model(self):
        self.gCNN = Global_CNN(
            window=self.window, n_zone=self.n_zone, n_kernels=self.n_kernels,
            drop_prob=self.drop_prob)
        self.lCNN = Local_CNN(
            window=self.window, n_zone=self.n_zone, n_kernels=self.n_kernels,
            drop_prob=self.drop_prob)

        self.ar = AR(window=self.window)
        self.dropout = nn.Dropout(p=self.drop_prob)
        self.active_func = nn.Tanh()
        self.W_output1 = nn.Linear(self.n_kernels, self.n_kernels)
        self.W_output2 = nn.Linear(self.n_kernels, self.n_kernels)
        self.W_output3 = nn.Linear(self.n_kernels, 2)
        self.ar1 = AR(window=self.window)
        self.ar2 = AR(window=self.window)
        self.attention = AMPO(self.n_kernels)

    def forward(self, x):
        #Nonlinear component
        # Dual CNN
        gCNN_output, *_ = self.gCNN(x)  
        lCNN_output, *_ = self.lCNN(x) 
        # gated mechanisim
        gate_output = torch.tanh(gCNN_output) * torch.sigmoid(lCNN_output)
        # dropout
        drop_output = self.dropout(gate_output)
        # active function
        active_output1 = self.active_func(self.W_output1(drop_output))
        active_output2 = self.active_func(self.W_output2(drop_output))
        # attention mechanisim with pooling operation
        ampo_output = self.attention(torch.stack([active_output1, active_output2], dim=2))
        nonlinear_output = self.active_func(self.W_output3(ampo_output))
        nonlinear_output = torch.transpose(nonlinear_output, 1, 2)
        #Linear component
        ar_output1 = self.ar1(x)
        ar_output2 = self.ar2(x)
        linear_output = torch.cat((ar_output1, ar_output2), 1)
        #Integration
        output = nonlinear_output + linear_output
        output = torch.transpose(output, 1, 2)
        return output 
    '''


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
        #evaluate on the test set
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
            model = Model(batch_size=batch_size, n_zone=zone_set[j], window=24, lag=lag_set[k])
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


