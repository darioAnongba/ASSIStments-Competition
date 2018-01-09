
# coding: utf-8

# In[ ]:


import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import WeightedRandomSampler, Sampler
from torch.autograd import Variable
from torch.nn.init import xavier_normal, kaiming_normal
import torch.nn.functional as F
import numpy as np
import pandas as pd
from tqdm import tqdm, tqdm_notebook
from sklearn.metrics import accuracy_score, roc_auc_score
import matplotlib.pyplot as plt
import random

DATA_DIR = 'Data/'
SEED = 7
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

categorical_features = set(['SY ASSISTments Usage',
                        'skill',
                        'problemId',
                        'assignmentId',
                        'assistmentId',
                        'problemType'])


# In[ ]:


pickle_train = open(DATA_DIR + "student_train_logs.pickle","rb")
train = pickle.load(pickle_train)

train[9][0].head()


# ### Parameters

# In[ ]:


input_dim = train[9][0].shape[1] + train[9][1].shape[1]
input_dim


# In[ ]:


class TrainingSet(Dataset):
    def __init__(self):
        self.idx = list(train.keys())
        self.sequences = train
        self.weights = [0.3 if x[2] == 0 else 0.8 for x in self.sequences.values()]
        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, id):
        student_id = self.idx[id]
        
        actions = self.sequences[student_id][0].as_matrix().astype(np.float32)
        fixed = self.sequences[student_id][1].as_matrix().astype(np.float32)
        target = np.asarray([self.sequences[student_id][2]]).astype(np.float32)
        seq = np.hstack([fixed, actions])

        return student_id, seq, target

    
train_dataset = TrainingSet()


# In[ ]:


class RNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers, bi, use_gpu=True):
        super(RNN, self).__init__()

        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.bi = bi
        self.output_dim = output_dim
        self.use_gpu = use_gpu
        
        self.gru = nn.GRU(input_size=input_dim,
                          hidden_size=hidden_dim,
                          num_layers=n_layers,
                          dropout=0.25,
                          bidirectional=bi)
        if bi:
            self.decoder = nn.Linear(hidden_dim*2, output_dim)
        else:
            self.decoder = nn.Linear(hidden_dim, output_dim)

        
    def weights_init(self):
        for name, param in self.named_parameters():
            if param.dim() >= 2:
                kaiming_normal(param)


    def init_hidden(self, batch_size):
        if self.bi:
            if self.use_gpu:
                return Variable(torch.zeros(self.n_layers*2, batch_size, self.hidden_dim)).cuda()
            else:
                return Variable(torch.zeros(self.n_layers*2, batch_size, self.hidden_dim))
        
        if self.use_gpu:
            return Variable(torch.zeros(self.n_layers, batch_size, self.hidden_dim)).cuda()
        else:
            return Variable(torch.zeros(self.n_layers, batch_size, self.hidden_dim))
        
    
    def forward(self, actions):
        batch_size = actions.size(1)
        hidden_state = self.init_hidden(batch_size)
        out, _ = self.gru(actions, hidden_state)
        out = out[-1,:,:]                                                                                                         
        out = self.decoder(out)                                                                                                   
        out = out.view(batch_size, self.output_dim)                                                                              
        return out
    
    def step(self, inp, target):                                                                                                        
        self.zero_grad()                                                                                                           
        output = self.forward(inp)                                                                                                      
        loss = self.criterion(output, target.float())                                                                      
        loss.backward()                                                                                                                 
        self.optimizer.step()                                                                                                           
        return loss.data[0], F.sigmoid(output)
    
    def fit(self, train_dataset):
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = optim.Adamax(self.parameters(), lr=1e-3)
        
        sampler = WeightedRandomSampler(train_dataset.weights, num_samples=len(train_dataset))
        loader = DataLoader(train_dataset, batch_size=1, num_workers=4, sampler=sampler)
        e_losses = []
        e_accs = []
        e_aucs = []
        e_bar = tqdm_notebook(range(10))
        for e in e_bar:
            self.train()
            e_loss = 0
            preds = []
            targets = []
            for i, (_, seq, label) in enumerate(tqdm_notebook(loader, leave=False)):
                seq = seq.permute(1,0,2)
                
                if self.use_gpu:
                    seq_var = Variable(seq).cuda()
                    label_var = Variable(label).cuda()
                else:
                    seq_var = Variable(seq)
                    label_var =  Variable(label)
                
                loss, output = self.step(seq_var, label_var)
                
                e_loss += loss
                preds.append(output.squeeze().cpu().data[0])
                targets.append(label.numpy()[0,0])
            preds = np.array(preds)
            targets = np.array(targets)
            auc = roc_auc_score(targets, preds)
            preds[preds >= 0.5] = 1
            preds[preds < 0.5] = 0
            
            acc = accuracy_score(preds, targets)
            e_losses.append(e_loss / (i+1))
            e_accs.append(acc)
            e_aucs.append(auc)
            e_bar.set_postfix(acc=acc, e_loss=e_losses[-1], auc=auc)
            
            
        return e_losses, e_accs, e_aucs


# In[ ]:


model = RNN(input_dim=input_dim, hidden_dim=256, output_dim=1, n_layers=3, bi=False, use_gpu=True)
model.weights_init()
model.cuda()


# In[ ]:


e_losses, e_accs, e_aucs = model.fit(train_dataset)

plt.plot(e_losses)
plt.plot(e_accs)
plt.plot(e_aucs)
plt.show()

