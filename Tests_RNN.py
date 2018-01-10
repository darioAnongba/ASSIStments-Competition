
# coding: utf-8

# In[1]:


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
get_ipython().magic('matplotlib inline')
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


# In[2]:


pickle_train = open(DATA_DIR + "student_train_logs.pickle","rb")
train = pickle.load(pickle_train)

train[9][0].head()


# ### Parameters

# In[3]:


input_dim = train[9][0].shape[1] + train[9][1].shape[1]
validation_set_size = 80


# In[4]:


validation = {k:v for k, v in random.sample(train.items(), validation_set_size)}
train_truncated = { k : train[k] for k in set(train) - set(validation) }


# In[15]:


class DataSet(Dataset):
    def __init__(self, sequences):
        self.idx = list(sequences.keys())
        self.sequences = sequences
        #self.weights = [0.3 if x[2] == 0 else 0.8 for x in self.sequences.values()]
        
    def __len__(self):
        return 5
        #return len(self.sequences)
    
    def __getitem__(self, id):
        student_id = self.idx[id]
        
        actions = self.sequences[student_id][0].as_matrix().astype(np.float32)
        fixed = self.sequences[student_id][1].as_matrix().astype(np.float32)
        target = np.asarray([self.sequences[student_id][2]]).astype(np.float32)
        seq = np.hstack([fixed, actions])

        return student_id, seq, target


# In[16]:


train_dataset = DataSet(train_truncated)
validation_dataset = DataSet(validation)


# In[23]:


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
    
    
    def evaluate_val(self, dataset):
        loader = DataLoader(dataset, batch_size=1, num_workers=4)
        
        y_preds = []
        y_true = []
        
        for i, (_, actions, target) in enumerate(tqdm_notebook(loader, leave=False)):
            y_true.append(target.numpy()[0,0])
            
            actions = actions.permute(1, 0, 2)
            
            if self.use_gpu:
                actions = Variable(actions).cuda()
            else:
                actions = Variable(actions)
                
            output = self.forward(actions)
            output = F.sigmoid(output)
            
            if self.use_gpu:
                y_preds.append(output.squeeze().cpu().data[0])
            else:
                y_preds.append(output.squeeze().data[0])
                
        return y_true, y_preds
    
    
    def predict(self, test_set):
        loader = DataLoader(test_set, batch_size=1, num_workers=4)
        
        preds = []
        
        for i, actions in enumerate(tqdm_notebook(loader, leave=False)):
            actions = actions.permute(1, 0, 2)
            
            if self.use_gpu:
                actions = Variable(actions).cuda()
            else:
                actions = Variable(actions)
            
            output = self.forward(actions)
            output = F.sigmoid(output)
            
            if self.use_gpu:
                preds.append(output.squeeze().cpu().data[0])
            else:
                preds.append(output.squeeze().data[0])
                
        return preds
    
    
    def fit(self, train_dataset):
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = optim.Adamax(self.parameters(), lr=1e-3)
        
        #sampler = WeightedRandomSampler(train_dataset.weights, num_samples=len(train_dataset))
        loader = DataLoader(train_dataset, batch_size=1, num_workers=4, shuffle=True)
        
        e_losses = []
        e_accs = []
        e_aucs = []
        
        e_val_accs = []
        e_val_aucs = []

        e_bar = tqdm_notebook(range(10))
        
        for e in e_bar:
            self.train()
            e_loss = 0
            preds = []
            targets = []
            val_preds = []
            
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
      
        return e_losses, e_accs, e_aucs


# In[26]:


model = RNN(input_dim=input_dim, hidden_dim=256, output_dim=1, n_layers=3, bi=True, use_gpu=False)
#model.weights_init()
#model.cuda()


# In[27]:


e_losses, e_accs, e_aucs = model.fit(train_dataset)

