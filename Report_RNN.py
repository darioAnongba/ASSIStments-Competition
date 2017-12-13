
# coding: utf-8

# # ASSISTments Data Mining Competition 2017 - Optional Semester Project

# ## Imports and constants

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


# Constants

# In[ ]:


DATA_DIR = 'Data/'

def init_seed(seed, use_gpu=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if use_gpu:
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


# ### Parameters

# In[ ]:


learning_rate = 1e-3
epochs = 10
dropout = 0.25
threshold = 20
n_layers = 3
hidden_dim = 256
validation_set_size = 80
balanced_data = False
bidirectional = True
use_gpu = True
seed = 7


# ## Loading the training data
# 
# The training data is stored in pickles as a dictionary where the keys are the student ids and and the value is tuple of **(sequence, fixed_features, target)**.
# 
# The **sequence** is the sequence of actions of the specific student stored as a dataframe and sorted by time.
# The **fixed_feature** are the student features that are static through time like the average correctness, the school he's attending or his MCAS grade.
# The **target** is either yes (1) or no (0) the student has done a carrer in STEM

# In[ ]:


pickle_train = open(DATA_DIR + "student_train_logs.pickle","rb")
train = pickle.load(pickle_train)

train[9][0].head()


# In[ ]:


dynamic_dim = train[9][0].shape[1]
fixed_dim = train[9][1].shape[1]
print(dynamic_dim, fixed_dim)
input_dim = dynamic_dim + fixed_dim
input_dim


# ## Removing features that are out of range
# 
# We discard columns that have a range that is too big outside [0, 1]

# In[ ]:


columns_discard = []

for i in range(train[9][0].shape[1]):
    v_min = 0
    v_max = 1000000
    for k, v in train.items():
        name = v[0].columns[i]
        v_min = min(v_min, v[0][name].min())
        v_max = max(v_min, v[0][name].max())

    if v_max - v_min > threshold:
        columns_discard.append(name)

print('Columns to discard with threshold:', threshold)
print('Number of columns discarded:', len(columns_discard))
input_dim = input_dim - len(columns_discard)
columns_discard


# In[ ]:


for k, v in train.items():
    train[k] = (v[0].drop(columns_discard, axis=1), v[1], v[2])


# ## Creating a validation set
# 
# We will simply take randomly 50 students as validation set

# In[ ]:


validation = {k:v for k, v in random.sample(train.items(), validation_set_size)}
train_truncated = { k : train[k] for k in set(train) - set(validation) }


# ## Truncating features out of range

# ## Creating Data Loaders

# Now, we implement data loaders that will enable us to iterate through our data and transform our data into tensors to be used with pyTorch.
# 
# With PyTorch, every DataLoader can be set a sampler that will define how the data is being sampled. Here we implement a Random weighted sampler in order to traverse our data randomly and also be able to define if a class should be sampled more often than its appearance in the dataset

# In[ ]:


class FullTrainingSet(Dataset):
    def __init__(self, balance=False):
        self.idx = list(train.keys())
        self.sequences = train
        
        if balance:
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


# In[ ]:


class TruncatedTrainingSet(Dataset):
    def __init__(self, balance=False):
        self.idx = list(train_truncated.keys())
        self.sequences = train_truncated
        
        if balance:
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


# In[ ]:


class ValidationSet(Dataset):
    def __init__(self, balance=False):
        self.idx = list(validation.keys())
        self.sequences = validation
        
        if balance:
            self.weights = [0.3 if x[2] == 0 else 0.8 for x in self.sequences.values()]
        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, id):
        student_id = self.idx[id]
        
        actions = self.sequences[student_id][0].as_matrix().astype(np.float32)
        fixed = self.sequences[student_id][1].as_matrix().astype(np.float32)
        target = np.asarray([self.sequences[student_id][2]]).astype(np.float32)
        seq = np.hstack([fixed, actions])
        
        return seq, target


# Testing that our sampler works fine

# In[ ]:


full_train_dataset = FullTrainingSet(balance=balanced_data)
train_dataset = TruncatedTrainingSet(balance=balanced_data)
validation_dataset = ValidationSet(balance=balanced_data)


# ## Building our RNN
# 
# Our RNN is composed of an LSTM that takes as data our sequence of actions per student. The LSTM outputs a number of hidden parameters on which we append our fixed features. We then fave a fully connected layer that takes as input our concatenated features and outputs a value for each class. Finally, we use the sigmoid function to output a probability for each class

# In[ ]:


class RNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers, bi, use_gpu=False, seed=7, dropout=0.25):
        super(RNN, self).__init__()

        init_seed(seed, use_gpu)
        
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.bi = bi
        self.output_dim = output_dim
        self.use_gpu = use_gpu
        
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            dropout=dropout,
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
        if self.use_gpu:
            if self.bi:
                return Variable(torch.zeros(self.n_layers*2, batch_size, self.hidden_dim)).cuda()
            else:
                return Variable(torch.zeros(self.n_layers, batch_size, self.hidden_dim)).cuda()
        else:
            if self.bi:
                return Variable(torch.zeros(self.n_layers*2, batch_size, self.hidden_dim))
            else:
                return Variable(torch.zeros(self.n_layers, batch_size, self.hidden_dim))

        return Variable(torch.zeros(self.n_layers, batch_size, self.hidden_dim)).cuda()

    
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

    
    def evaluate_val(self, dataset, balanced=False):
        if balanced:
            sampler = WeightedRandomSampler(dataset.weights, num_samples=len(train_dataset))
            loader = DataLoader(dataset, batch_size=1, num_workers=4, sampler=sampler)
        else:
            loader = DataLoader(dataset, batch_size=1, num_workers=4, shuffle=True)
        
        y_preds = []
        y_true = []
        
        for i, (actions, target) in enumerate(tqdm_notebook(loader, leave=False)):
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
                
    
    def fit(self, train_dataset, validation_dataset=None, epochs=10, balanced=False):
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = optim.Adamax(self.parameters(), lr=learning_rate)
        
        if balanced:
            sampler = WeightedRandomSampler(train_dataset.weights, num_samples=len(train_dataset))
            loader = DataLoader(train_dataset, batch_size=1, num_workers=4, sampler=sampler)
        else:
            loader = DataLoader(train_dataset, batch_size=1, num_workers=4, shuffle=True)

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
            
            for i, (_, seq, label) in enumerate(tqdm_notebook(loader, leave=False)):
                seq = seq.permute(1, 0, 2)
                
                if self.use_gpu:
                    seq_var = Variable(seq).cuda()
                    label_var = Variable(label).cuda()
                else:
                    seq_var = Variable(seq)
                    label_var = Variable(label)
                    
                loss, output = self.step(seq_var, label_var)
                e_loss += loss
                
                if self.use_gpu:
                    preds.append(output.squeeze().cpu().data[0])
                else:
                    preds.append(output.squeeze().data[0])
                    
                targets.append(label.numpy()[0,0])
             
            # Train set loss, accuracy and AUC
            targets = np.array(targets)
            preds = np.array(preds)
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
                val_targets, val_preds = self.evaluate_val(validation_dataset, balanced=balanced)
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


# ## Training the network

# In order to train our RNN, we use our previously defined DataLoader and try to minimize the Mean squared error using the MSELoss pyTorch loss function

# ### Defining parameters

# In[ ]:


model = RNN(
    input_dim=input_dim,
    hidden_dim=hidden_dim,
    output_dim=1,
    n_layers=n_layers,
    bi=bidirectional,
    use_gpu=use_gpu,
    seed=seed,
    dropout=dropout
)
model.weights_init()

# If not using CUDA, comment this line
model.cuda()


# In[ ]:


e_losses, e_accs, e_aucs, e_val_accs, e_val_aucs = model.fit(train_dataset, validation_dataset, epochs=epochs, balanced=balanced_data)


# In[ ]:


print('Training set loss, accuracy and AUC')
plt.plot(e_losses)
plt.plot(e_accs)
plt.plot(e_aucs)
plt.show()


# In[ ]:


print('Validation set accuracy and AUC')

plt.plot(e_val_accs)
plt.plot(e_val_aucs)
plt.show()


# ## Predict test data

# In[ ]:


pickle_test = open(DATA_DIR + "student_test_logs.pickle","rb")
test = pickle.load(pickle_test)

test[9][0].head()


# In[ ]:


for k, v in test.items():
    test[k] = (v[0].drop(columns_discard, axis=1), v[1])


# In[ ]:


class TestSet(Dataset):
    def __init__(self):
        self.idx = list(test.keys())
        self.sequences = test
        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, id):
        student_id = self.idx[id]
        
        actions = self.sequences[student_id][0].as_matrix().astype(np.float32)
        fixed = self.sequences[student_id][1].as_matrix().astype(np.float32)
        seq = np.hstack([fixed, actions])
        
        return seq


# In[ ]:


test_set = TestSet()


# In[ ]:


model.predict(test_set)

