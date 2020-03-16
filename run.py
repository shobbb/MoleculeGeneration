import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import torch.utils.data as data
import pickle as p
import sys

import deepsets
from denoise import torch_sigma
from qm9_fetcher import Qm9Dataset, Qm9RotInvDataset

#################### Settings ##############################
num_epochs = 250
batch_size = 32
network_dim = 64  #For 5000 points use 512, for 1000 use 256, for 100 use 256
#################### Settings ##############################

device = torch.device("cuda")
kwargs = {'num_workers': 1, 'pin_memory': True, 'drop_last': True}
model_kwargs = {
    deepsets.DTanh: {
        'pool': 'max1',
        'x_dim': 3,
        'd_dim': network_dim,
    },
    torch_sigma:{
        "batch_size": batch_size,
        "f1": 512,
        "x_dim": 3,
    },
}


class MoleculeClassification(object):
    def __init__(
        self, load_model=False, exp_name='Molecule_Classification', 
        model=deepsets.DTanh, weight_xentp=False, fetcher=Qm9Dataset
    ):
        #Data loader
        self.train_loader = data.DataLoader(
            fetcher(), batch_size=batch_size, shuffle=True, **kwargs
        )
        self.valid_loader = data.DataLoader(
            fetcher(data_type='valid'), batch_size=batch_size, shuffle=True, **kwargs
        )
        self.test_loader = data.DataLoader(
            fetcher(data_type='test'), batch_size=batch_size, shuffle=True, **kwargs
        )
        self.exp_name = exp_name

        #Setup network
        self.model = model(**model_kwargs[model]).cuda()
        if load_model:
            self.model.load_state_dict(torch.load(f'./models/{self.exp_name}.pt'))
        if weight_xentp:
            weights = np.array([1.96692393, 2.84026181, 16.0560000, 13.0218978, 2293.71429])
            weights /= weights.max()
            weights = torch.FloatTensor(weights).cuda()
            self.L = nn.CrossEntropyLoss(weight=weights).cuda()
        else:
            self.L = nn.CrossEntropyLoss().cuda()
        self.optimizer = optim.Adam([{'params':self.model.parameters()}], lr=1e-3, weight_decay=1e-7, eps=1e-3)
        self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=list(range(400,num_epochs,400)), gamma=0.1)
        #self.optimizer = optim.Adamax([{'params':self.model.parameters()}], lr=5e-4, weight_decay=1e-7, eps=1e-3) # optionally use this for 5000 points case, but adam with scheduler also works

    def train(self):
        self.model.train()
        print('Begin Training')
        loss_val = float('inf')
        for j in range(num_epochs):
            sys.stdout.flush()
            counts = 0
            sum_acc = 0.0
            for x, y in self.train_loader:
                counts += len(y) * 18
                X = x.to(device).float()
                Y = y.to(device).long()
                self.optimizer.zero_grad()
                f_X = self.model(X)
                f_X = f_X.reshape((f_X.size()[0] * f_X.size()[1], -1))
                Y = Y.reshape((Y.size()[0] * Y.size()[1], -1)).squeeze()
                loss = self.L(f_X, Y)
                loss_val = loss.data.cpu().numpy()
                sum_acc += (f_X.max(dim=1)[1] == Y).float().sum().data.cpu().numpy()
                loss.backward()
                # deepsets.clip_grad(self.model, 5)
                self.optimizer.step()
                del X,Y,f_X,loss
            train_acc = sum_acc/counts
            self.scheduler.step()
            if j%10==9:
                print(f'\tAfter epoch {j+1} Train Accuracy: {train_acc}')
                self.valid()

    def valid(self):
        self.model.eval()
        counts = 0
        sum_acc = 0.0
        for x, y in self.valid_loader:
            counts += len(y) * 18
            X = x.to(device).float()
            Y = y.to(device).long()
            f_X = self.model(X)
            f_X = f_X.reshape((f_X.size()[0] * f_X.size()[1], -1))
            Y = Y.reshape((Y.size()[0] * Y.size()[1], -1)).squeeze()
            sum_acc += (f_X.max(dim=1)[1] == Y).float().sum().data.cpu().numpy()
            del X,Y,f_X
        test_acc = sum_acc/counts
        print('\tValid Accuracy: {0:0.3f}'.format(test_acc))
        return test_acc

    def test(self):
        self.model.eval()
        counts = 0
        sum_acc = 0.0
        guesses = np.array([])
        for x, y in self.test_loader:
            counts += len(y) * 18
            X = x.to(device).float()
            Y = y.to(device).long()
            f_X = self.model(X)
            f_X = f_X.reshape((f_X.size()[0] * f_X.size()[1], -1))
            Y = Y.reshape((Y.size()[0] * Y.size()[1], -1)).squeeze()

            # get predictions
            pred = np.stack(
                [(f_X.max(dim=1)[1] == Y).long().cpu().numpy(), Y.data.cpu().numpy()]
            ).T
            guesses = np.concatenate([guesses, pred]) if guesses.any() else pred


            sum_acc += (f_X.max(dim=1)[1] == Y).float().sum().data.cpu().numpy()
            del X,Y,f_X
        test_acc = sum_acc/counts
        print('Test Accuracy: {0:0.3f}'.format(test_acc))
        return test_acc, guesses
    
    def save(self):
        torch.save(self.model.state_dict(), open(f'./models/{self.exp_name}.pt', 'wb'))

if __name__ == "__main__":
    exp = 'dmps'
    t = MoleculeClassification(
        load_model=False, exp_name=exp, fetcher=Qm9Dataset, 
        weight_xentp=False, model=torch_sigma,
    )
    print("Training for Molecule Atom Classification")
    t.train()
    acc, guesses = t.test()
    t.save()
    with open('./results/{}_predictions.p'.format(exp), 'wb') as f:
        p.dump(guesses, f)