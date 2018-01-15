
# coding: utf-8

# In[ ]:


import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import WeightedRandomSampler, Sampler
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import pandas as pd
from tqdm import tqdm, tqdm_notebook
from sklearn.metrics import accuracy_score, roc_auc_score
import random

DATA_DIR = 'Data/'
SEED = 7
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

categorical_features = set(['skill',
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


validation_set_size = 30
input_dim = train[9][0].shape[1]
fixed_dim = train[9][1].shape[0]
hidden_dim = 256
dropout = 0.25
n_layers = 3
bidirectional = True
use_gpu = True
learning_rate = 1e-3
epochs = 20
num_workers = 4


# In[ ]:


validation = {k:v for k, v in random.sample(train.items(), validation_set_size)}
train_truncated = { k : train[k] for k in set(train) - set(validation) }


# In[ ]:


class DataSet(Dataset):
    def __init__(self, sequences):
        self.idx = list(sequences.keys())
        self.sequences = sequences
    
    
    def __len__(self):
        return len(self.sequences)
    
    
    def __getitem__(self, id):
        student_id = self.idx[id]
        
        dynamic = self.sequences[student_id][0].as_matrix().astype(np.float32)
        fixed = self.sequences[student_id][1].as_matrix().astype(np.float32)
        target = np.asarray([self.sequences[student_id][2]]).astype(np.float32)

        return student_id, dynamic, fixed, target


# In[ ]:


train_dataset = DataSet(train_truncated)
validation_dataset = DataSet(validation)


# In[ ]:


class RNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, fixed_dim, n_layers, bi, use_gpu, output_dim=1, batch_size=1):
        super(RNN, self).__init__()

        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.bi = bi
        self.output_dim = output_dim
        self.use_gpu = use_gpu
        self.batch_size = batch_size
        
        self.gru = nn.GRU(input_size=input_dim,
                          hidden_size=hidden_dim,
                          num_layers=n_layers,
                          dropout=dropout,
                          bidirectional=bi)
        if bi:
            self.decoder = nn.Linear(hidden_dim*2 + fixed_dim, output_dim)
        else:
            self.decoder = nn.Linear(hidden_dim + fixed_dim, output_dim)
        

    def init_hidden(self):
        if self.bi:
            if self.use_gpu:
                return Variable(torch.zeros(self.n_layers*2, self.batch_size, self.hidden_dim)).cuda()
            else:
                return Variable(torch.zeros(self.n_layers*2, self.batch_size, self.hidden_dim))
        
        if self.use_gpu:
            return Variable(torch.zeros(self.n_layers, self.batch_size, self.hidden_dim)).cuda()
        else:
            return Variable(torch.zeros(self.n_layers, self.batch_size, self.hidden_dim))

    
    def forward(self, actions, fixed):
        hidden_state = self.init_hidden()
        out, _ = self.gru(actions, hidden_state)
        out = self.decoder(torch.cat([out[-1, :, :], fixed], dim=1))                                                                                                   
        out = out.view(self.batch_size, self.output_dim)
        
        return out
    
    
    def step(self, dynamic, fixed, target):                                                                                                        
        self.zero_grad()                                                                                                           
        output = self.forward(dynamic, fixed)                                                                                                      
        loss = self.criterion(output, target.float())                                                                      
        loss.backward()                                                                                                                 
        self.optimizer.step()
        
        return loss.data[0], F.sigmoid(output)
    
    
    def evaluate_val(self, dataset):
        loader = DataLoader(dataset, self.batch_size, num_workers=num_workers)
        
        y_preds = []
        y_true = []
        
        for i, (_, actions, fixed, target) in enumerate(tqdm(loader, leave=False)):
            y_true.append(target.float())
            
            actions = actions.permute(1, 0, 2)
            
            if self.use_gpu:
                actions = Variable(actions).cuda()
                fixed = Variable(fixed).cuda()
            else:
                actions = Variable(actions)
                fixed = Variable(fixed)
                
            output = self.forward(actions, fixed)
            output = F.sigmoid(output)
            
            if self.use_gpu:
                y_preds.append(output.squeeze().cpu().data[0])
            else:
                y_preds.append(output.squeeze().data[0])
                
        return y_true, y_preds
    
    
    def fit(self, train_dataset):
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = optim.Adamax(self.parameters(), lr=learning_rate)
        
        loader = DataLoader(train_dataset, batch_size=self.batch_size, num_workers=num_workers, shuffle=True)
        
        e_losses = []
        e_accs = []
        e_aucs = []
        
        e_val_accs = []
        e_val_aucs = []
        
        e_bar = tqdm_notebook(range(epochs))
        for e in e_bar:
            self.train()
            e_loss = 0
            preds = []
            targets = []
            val_preds = []
            
            for i, (_, seq, fixed, label) in enumerate(tqdm_notebook(loader, leave=False)):
                seq = seq.permute(1,0,2)
                
                if self.use_gpu:
                    seq_var = Variable(seq).cuda()
                    fixed_var = Variable(fixed).cuda()
                    label_var = Variable(label).cuda()
                else:
                    seq_var = Variable(seq)
                    fixed_var = Variable(fixed)
                    label_var =  Variable(label)
                
                loss, output = self.step(seq_var, fixed_var, label_var)
                e_loss += loss
                
                preds.append(output.squeeze().cpu().data[0])
                targets.append(label.float())
            
            preds = np.array(preds)
            targets = np.array(targets)
            auc = roc_auc_score(targets, preds)
            
            preds[preds >= 0.5] = 1
            preds[preds < 0.5] = 0
            acc = accuracy_score(preds, targets)
            
            e_losses.append(e_loss / (i+1))
            e_accs.append(acc)
            e_aucs.append(auc)

            # Validation set accuracy and AUC
            val_acc = None
            val_auc = None
            if validation_dataset is not None:
                val_targets, val_preds = self.evaluate_val(validation_dataset)
                val_targets = np.array(val_targets)
                val_preds = np.array(val_preds)
                val_auc = roc_auc_score(val_targets, val_preds)

                val_preds[val_preds >= 0.5] = 1
                val_preds[val_preds < 0.5] = 0
                val_acc = accuracy_score(val_preds, val_targets)

                e_val_accs.append(val_acc)
                e_val_aucs.append(val_auc)
            
            e_bar.set_postfix(acc=acc, e_loss=e_losses[-1], auc=auc, val_acc=val_acc, val_auc=val_auc)
      
        return e_losses, e_accs, e_aucs, e_val_accs, e_val_aucs


# In[ ]:


model = RNN(input_dim=input_dim,
            hidden_dim=hidden_dim,
            fixed_dim=fixed_dim,
            n_layers=n_layers,
            bi=bidirectional,
            use_gpu=use_gpu)

if use_gpu:
    model.cuda()
    
model


# In[ ]:


e_losses, e_accs, e_aucs, e_val_accs, e_val_aucs = model.fit(train_dataset)


# In[ ]:


print('Accuracies')
print(e_accs, e_val_accs)

print('ROC AUC')
print(e_aucs, e_val_aucs)

