import numpy as np
import torch.optim as optim
import torchmetrics
import pytorch_lightning as pl

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

WINDOW_SIZE = 32


class PoseDataset(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.y = Y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

openpose_to_detectron_mapping = [0, 1, 28, 29, 26, 27, 32, 33, 30, 31, 8, 9, 2, 3, 10, 11, 4, 5, 12, 13, 6, 7, 20, 21, 14, 15, 22, 23, 16, 17, 24, 25, 18, 19]

class PoseDataModule(pl.LightningDataModule):
    def __init__(self, data_root, batch_size):
        super().__init__()
        self.data_root = data_root
        self.batch_size = batch_size
        self.X_train_path = self.data_root + "X_train.txt"
        self.X_test_path = self.data_root + "X_test.txt"
        self.y_train_path = self.data_root + "Y_train.txt"
        self.y_test_path = self.data_root + "Y_test.txt"


    def convert_to_detectron_format(self, row):
        row = row.split(',')
        temp = row[:2] + row[4:]
        temp = [temp[i] for i in openpose_to_detectron_mapping]
        return temp

    def load_X(self, X_path):
        file = open(X_path, 'r')
        X = np.array(
            [elem for elem in [
                self.convert_to_detectron_format(row) for row in file
            ]],
            dtype=np.float32
        )
        file.close()
        blocks = int(len(X) / WINDOW_SIZE)
        X_ = np.array(np.split(X, blocks))
        return X_

    def load_y(self, y_path):
        file = open(y_path, 'r')
        y = np.array(
            [elem for elem in [
                row.replace('  ', ' ').strip().split(' ') for row in file
            ]],
            dtype=np.int32
        )
        file.close()
        return y - 1

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        X_train = self.load_X(self.X_train_path)
        X_test = self.load_X(self.X_test_path)
        y_train = self.load_y(self.y_train_path)
        y_test = self.load_y(self.y_test_path)
        self.train_dataset = PoseDataset(X_train, y_train)
        self.val_dataset = PoseDataset(X_test, y_test)

    def train_dataloader(self):
        train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True
        )
        return train_loader

    def val_dataloader(self):
        val_loader = torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False
        )
        return val_loader

TOT_ACTION_CLASSES = 6

class ActionClassificationLSTM(pl.LightningModule):
    def __init__(self, input_features, hidden_dim, learning_rate=0.001):
        super().__init__()
        self.save_hyperparameters()
        self.lstm = nn.LSTM(input_features, hidden_dim, batch_first=True)
        self.linear = nn.Linear(hidden_dim, TOT_ACTION_CLASSES)

    def forward(self, x):
        lstm_out, (ht, ct) = self.lstm(x)
        return self.linear(ht[-1])

    def training_step(self, batch, batch_idx):
        x, y = batch
        y = torch.squeeze(y)
        y = y.long()
        y_pred = self(x)
        loss = F.cross_entropy(y_pred, y)
        prob = F.softmax(y_pred, dim=1)
        pred = prob.data.max(dim=1)[1]
        acc = torchmetrics.functional.accuracy(pred, y)
        dic = {
            'batch_train_loss': loss,
            'batch_train_acc': acc
        }
        self.log('batch_train_loss', loss, prog_bar=True)
        self.log('batch_train_acc', acc, prog_bar=True)
        return {'loss': loss, 'result': dic}

    def training_epoch_end(self, training_step_outputs):
        avg_train_loss = torch.tensor([x['result']['batch_train_loss'] for x in training_step_outputs]).mean()
        avg_train_acc = torch.tensor([x['result']['batch_train_acc'] for x in training_step_outputs]).mean()
        self.log('train_loss', avg_train_loss, prog_bar=True)
        self.log('train_acc', avg_train_acc, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y = torch.squeeze(y)
        y = y.long()
        y_pred = self(x)
        loss = F.cross_entropy(y_pred, y)
        prob = F.softmax(y_pred, dim=1)
        pred = prob.data.max(dim=1)[1]
        acc = torchmetrics.functional.accuracy(pred, y)
        dic = {
            'batch_val_loss': loss,
            'batch_val_acc': acc
        }
        self.log('batch_val_loss', loss, prog_bar=True)
        self.log('batch_val_acc', acc, prog_bar=True)
        return dic

    def validation_epoch_end(self, validation_step_outputs):
        avg_val_loss = torch.tensor([x['batch_val_loss']
                                     for x in validation_step_outputs]).mean()
        avg_val_acc = torch.tensor([x['batch_val_acc']
                                    for x in validation_step_outputs]).mean()
        self.log('val_loss', avg_val_loss, prog_bar=True)
        self.log('val_acc', avg_val_acc, prog_bar=True)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-15, verbose=True)
        return {"optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "interval": "epoch", "frequency": 1, "monitor": "val_loss"}}