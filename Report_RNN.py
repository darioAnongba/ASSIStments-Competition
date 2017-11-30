
# coding: utf-8

# # ASSISTments Data Mining Competition 2017 - Optional Semester Project

# ## Imports and constants

# In[2]:


import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import pandas as pd
from tqdm import tqdm_notebook
from sklearn.metrics import accuracy_score, roc_auc_score

DATA_DIR = 'Data/'


# ## Loading the training data
# 
# The training data is stored in pickles as a dictionary where the keys are the student ids and and the value is tuple of **(sequence, fixed_features, target)**.
# 
# The **sequence** is the sequence of actions of the specific student stored as a dataframe and sorted by time.
# The **fixed_feature** are the student features that are static through time like the average correctness, the school he's attending or his MCAS grade.
# The **target** is either yes (1) or no (2) the student has done a carrer in STEM

# In[5]:


pickle_train = open(DATA_DIR + "student_train_logs.pickle","rb")
train = pickle.load(pickle_train)

train[9][0].head()


# In[6]:


(train[9][1], train[9][2])


# ## Creating a Data Loader

# Now, we implement a data loader that will enable us to iterate through our data and transform our data into tensors to be used with pyTorch.
# 
# With PyTorch, every DataLoader can be set a sampler that will define how the data is being sampled. Here we implement a Random weighted sampler in order to traverse our data randomly and also be able to define if a class should be sampled more often than its appearance in the dataset

# In[7]:


class TrainingSet(Dataset):
    def __init__(self):
        self.idx = list(train.keys())
        self.sequences = train
        self.weights = [1] * len(train)
        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, id):
        student_id = self.idx[id]
        
        actions = self.sequences[student_id][0].as_matrix().astype(np.float32)
        fixed = self.sequences[student_id][1].as_matrix().astype(np.float32)
        target = np.asarray([self.sequences[student_id][2]]).astype(np.float32)
        
        return student_id, actions, fixed, target


# Testing that our sampler works fine

# In[8]:


train_dataset = TrainingSet()
sampler = WeightedRandomSampler(train_dataset.weights, num_samples=len(train_dataset))
train_loader = DataLoader(train_dataset, sampler=sampler, batch_size=1, num_workers=4)

nStudents = {}
nClasses = [0] * 2
for i, (sid, actions, fixed, target) in enumerate(tqdm_notebook(train_loader)):
    sid = sid.numpy()[0]
    if not sid in nStudents:
        nStudents[sid] = 1
    else:
        nStudents[sid] += 1
    nClasses[int(target.numpy()[0][0])] += 1

print('The distribution of classes is:', nClasses)


# ## Building our RNN
# 
# Our RNN is composed of an LSTM that takes as data our sequence of actions per student. The LSTM outputs a number of hidden parameters on which we append our fixed features. We then fave a fully connected layer that takes as input our concatenated features and outputs a value for each class. Finally, we use the sigmoid function to output a probability for each class

# In[9]:


class LSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, fixed_dim, output_dim, batch_size=1, n_layers=1, use_gpu=False):
        super(LSTMClassifier, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.n_layers = n_layers
        
        self.lstm = nn.LSTM(input_size=input_dim,
                            hidden_size=hidden_dim,
                            num_layers=n_layers,
                            dropout=0.25,
                            bidirectional=True)
        self.hidden2label = nn.Linear(hidden_dim*2 + fixed_dim, output_dim)

    def init_hidden(self):
        if use_gpu:
            h0 = Variable(torch.zeros(self.n_layers*2, self.batch_size, self.hidden_dim)).cuda()
            c0 = Variable(torch.zeros(self.n_layers*2, self.batch_size, self.hidden_dim)).cuda()
        else:
            h0 = Variable(torch.zeros(self.n_layers*2, self.batch_size, self.hidden_dim))
            c0 = Variable(torch.zeros(self.n_layers*2, self.batch_size, self.hidden_dim))
        return (h0, c0)

    def forward(self, actions, fixed):
        hidden_state = self.init_hidden()
        out, _ = self.lstm(actions, hidden_state)
        y = self.hidden2label(torch.cat([out[-1, :, :], fixed], dim=1))
        y = F.sigmoid(y)
        return y


# ## Training the network

# In order to train our RNN, we use our previously defined DataLoader and try to minimize the Mean squared error using the MSELoss pyTorch loss function

# ### Defining parameters

# In[10]:


input_dim = train[9][0].shape[1]
n_hidden = 64
fixed_dim = train[9][1].shape[0]
output_dim = 1
batch_size = 1
n_layers = 1

print(input_dim, n_hidden, fixed_dim, output_dim, batch_size, n_layers)


# In[11]:


train_dataset = TrainingSet()
sampler = WeightedRandomSampler(train_dataset.weights, num_samples=len(train_dataset))
train_loader = DataLoader(train_dataset, sampler=sampler, batch_size=1, num_workers=4)

lstm = LSTMClassifier(input_dim, n_hidden, fixed_dim, output_dim, batch_size, n_layers).cuda()

criterion = nn.MSELoss()
learning_rate = 0.0001 # If you set this too high, it might explode. If too low, it might not learn
optimizer = optim.Adam(lstm.parameters(), lr=learning_rate)

losses = []
accs = []
aucs = []
epochs = 5
batch = 20
bar1 = tqdm_notebook(range(epochs))

for epoch in bar1:
    train_loss = 0
    train_acc = []
    train_auc = []
    
    for i, (sid, actions, fixed, target) in enumerate(tqdm_notebook(train_loader)):
        actions = Variable(actions).permute(1,0,2)
        fixed = Variable(fixed)
        target = Variable(target)

        lstm.zero_grad()
        output = lstm(actions, fixed)
        loss = criterion(output, target)
        loss.backward()
        
        # Gradient clipping
        # torch.nn.utils.clip_grad_norm(lstm.parameters(), 0.25)
        if i % batch == 0:
            optimizer.step()
        
        train_loss += loss.data[0]
        
        _, argmax = output.data.max(1)
        y_preds = argmax.view(-1).numpy()
        y_true = target.data.view(-1).numpy()

        train_acc.append(accuracy_score(y_true, y_preds))
        train_auc.append(roc_auc_score(y_true, y_preds))

    train_loss /= i+1
    
    losses.append(train_loss)
    accs.append(np.mean(train_acc))
    aucs.append(np.mean(train_auc))
    
    bar1.set_postfix(loss=losses[-1], acc=accs[-1], auc=aucs[-1])


# In[ ]:


import matplotlib.pyplot as plt

plt.plot(losses)
plt.plot(accs)
plt.show()


# ## Predict test data
