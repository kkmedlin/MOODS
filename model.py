import torch
import torch.nn as nn
from torch import nn
from torch import optim
import torch.nn.functional as F
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, matthews_corrcoef
import numpy as np
import math

#define device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class Basic_DNN(nn.Module):
    def __init__(self, X, lr = 0.0001):
        super().__init__()
        #print("X shape 1:", X.shape[1])
        self.hidden1 = nn.Linear(X.shape[1],256)
        self.hidden2 = nn.Linear(256, 128)
        self.relu = nn.ReLU()
        self.output  = nn.Linear(128, 2)    
        self.softmax = nn.Softmax(dim=1)
        self.optimizer = optim.Adam(self.parameters(), lr=lr) 
    
    def predict(self, xb):
        #forward = self.forward(xb)
        #hat = torch.argmax(self.forward(xb), dim=1)
        #print("predict argmax forward hat:", hat)
        #print("hat shape:", hat.shape)
        return torch.argmax(self.forward(xb), dim=1)

    def forward(self, x):  
        x = self.hidden1(x)  
        x = self.hidden2(x)
        x = self.relu(x)
        x = self.output(x)
        #print("output:", x)
        x = self.softmax(x) 
        #print("softmax:", x)
        #print("softmax shape:", x.shape)
        return x
    
    def zValue(self, x):  
        x = self.hidden1(x)  
        x = self.hidden2(x)
        x = self.relu(x)
        x = self.output(x)
        #print("output:", x)
        #print("softmax:", x)
        #print("softmax shape:", x.shape)
        return x
    
    def metrics(self, xb, yb): 
        yhat = self.predict(xb.to(device)).detach().cpu()
        F1_score_2 = f1_score(yb, yhat)
        precision_2 = precision_score(yb, yhat, zero_division=1)
        recall_2 = recall_score(yb, yhat)
        auc = roc_auc_score(yb, yhat)
        accuracy = ( (yhat == yb).float().mean())
        MCC = matthews_corrcoef(yb, yhat)
        return accuracy, F1_score_2, precision_2, recall_2, auc, MCC
    
    def accuracy(self, xb, yb):
        yhat = self.predict(xb.to(device)).detach().cpu()
        accuracy = ( yhat == yb).float().mean()
        return accuracy
    
    def effOneMinority(self, xb, yb):
        yhat = self.predict(xb.to(device)).detach().cpu()
        F1_scoreMinority = f1_score(yb, yhat)
        return F1_scoreMinority
    
    def effOneAverage(self, xb, yb):
        yhat = self.predict(xb.to(device)).detach().cpu()
        F1_scoreAverage = f1_score(yb, yhat, average='macro')
        #print("F1:",F1_scoreAverage)
        #print("F1 type:",F1_scoreAverage.dtype)
        return F1_scoreAverage
    
    def precisionMinority(self, xb, yb):
        yhat = self.predict(xb.to(device)).detach().cpu()
        precisionMinority = precision_score(yb, yhat, zero_division=0.0)
        #print("precisionMinority:",precisionMinority)
        return precisionMinority

    def precisionAverage(self, xb, yb):
        yhat = self.predict(xb.to(device)).detach().cpu()
        precisionAverage = precision_score(yb, yhat, zero_division=0.0, average='macro')
        #print("precisionAverage:",precisionAverage)
        #print("precision type:",precisionAverage.dtype)
        return precisionAverage
        
    def recallMinority(self, xb, yb):
        yhat = self.predict(xb.to(device)).detach().cpu()
        recallMinority = recall_score(yb, yhat)
        #print("recallMinority:",recallMinority)
        return recallMinority

    def recallAverage(self, xb, yb):
        yhat = self.predict(xb.to(device)).detach().cpu()
        recallAverage = recall_score(yb, yhat, average='macro')
        #print("recallAverage:",recallAverage)
        return recallAverage
    
    def areaUnder(self, xb, yb):
        yhat = self.predict(xb.to(device)).detach().cpu()
        auc = roc_auc_score(yb, yhat)
        return auc
    
    def loss(self, xb, yb):
        #non-normalized version for computing majority loss
        return F.cross_entropy(self.forward(xb.to(device)), yb.to(device), reduction='none').detach().cpu()
    
    def loss_J(self, xb, yb):
        #normalized for optimizing inner loss function of full training sample
        return F.cross_entropy(self.forward(xb.to(device)), yb.to(device))
    
    def fit(self, loader, epochs = 0):
        norm2GradientSquared = 1
        norm2Gradient = 0
        arrayLoss = []
        while math.sqrt(norm2GradientSquared) >10e-6 and epochs <2000:
            for _, batch in enumerate(loader):
                x, y = batch['x'], batch['y']
                loss = self.loss_J(x,y) 
                arrayLoss.append(loss.detach().cpu().numpy())
                self.optimizer.zero_grad()
                loss.backward()
                theta = self.parameters()
                
                norm2GradientSquared = 0.0
                #print("epoch:", epochs)
                for param in theta:
                    #print("theta:", theta)
                    #print("param:", param)
                    norm2GradientSquared += (torch.linalg.norm(param.grad))**2
                    #print("param.grad:", param.grad)
                    #paramGrad = param.grad
                    #print("param shape:", param.grad.shape)
                    #print("norm2GradientSquared:", norm2GradientSquared)
                
                norm2Gradient = torch.sqrt(norm2GradientSquared.detach().cpu())
                #print()
                self.optimizer.step()   
            epochs = epochs + 1
        #return paramGrad
        return norm2Gradient
    