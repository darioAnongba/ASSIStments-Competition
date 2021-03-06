# coding: utf-8

# # ASSISTments Data Mining Competition 2017 - Report of RNN Model

# ## Imports and constants

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
from sklearn.metrics import accuracy_score, roc_auc_score, mean_squared_error
import random
import pickle

DATA_DIR = 'Data/'
RESULTS_DIR = 'Results/'
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

def save_pickle(dict_data, name):
    pickle_out = open(RESULTS_DIR + name + '.pickle', 'wb')
    pickle.dump(dict_data, pickle_out)
    pickle_out.close()

# ## Loading the training data
# 
# The training data is stored in pickles as a dictionary where the keys are student ids and and values are tuples of **(Sequence of dynamic features, array of fixed features, target)**:
# 
# - **Dynamic features**: Features that can potentially contain different values per action in a sequence. Stored as a pandas dataframe sorted by time.
# - **Fixed features**: Features that are unchanged through time. Ex: Average correctness, the school id or MCAS grade. Stored as a pandas Series.
# - **target**: Either yes (1) or no (0) the student has chosen a carrer in STEM.

pickle_train = open(DATA_DIR + "student_train_logs.pickle","rb")
train = pickle.load(pickle_train)

# ### Parameters
# 
# Here we define the parameters of the whole system, each parameter can be tweaked as needed:
# 
# ** System parameters**
# - **Validation_set_size** (int): The size of the validation set. Due to the poor amount of data that we are provided, the size of the validation set should be small.
# - **use_gpu** (bool): Use CUDA or not
# - **num_workers** (int): Maximum number of threads on the CPU.
# - **epochs** (int): Number of epochs
#
# ** Model (RNN) parameters**
# - **dynamic_dim** (int): Number of dynamic features (that change at every action)
# - **fixed_dim** (int): Number of the static features (that are unchanged at every action)
# - **hidden_dim** (int): Number of features to be output by the Recurrent neural network 
# - **n_layers** (int): Number of layers of the RNN
# - **bidirectional** (bool): To either use a bidirectional RNN or not
# - **dropout** (float in [0, 1]): Dropout of the RNN
# - **learning_rate** (float in [0, 1]): Learning rate

parameters = {
    'validation_set_size': 30,
    'dynamic_dim': train[9][0].shape[1],
    'fixed_dim': train[9][1].shape[0],
    'hidden_dim': 32,
    'dropout': 0.1,
    'n_layers': 3,
    'bidirectional': True,
    'use_gpu': True,
    'learning_rate': 1e-3,
    'epochs': 30,
    'num_workers': 4
}


# ## Creating the validation set
#
# We create the validation set by randomly selecting `validation_set_size` students from the total training set and discarding those students from the actual training set.

validation = {k:v for k, v in random.sample(train.items(), parameters['validation_set_size'])}
train_truncated = { k : train[k] for k in set(train) - set(validation) }

# ## Creating a Data Loader
# 
# Now, we implement a data loader that will enable us to iterate through our data and transform our data into tensors to be used with pyTorch.
# 
# With PyTorch, every DataLoader can be set a sampler that will define how the data is being sampled.
# Here we do not define any sampler but it could be a good idea to create a random weighted sampler in order to balance.

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


train_dataset = DataSet(train_truncated)
validation_dataset = DataSet(validation)


# ## Building the RNN model
# 
# Our RNN is composed of an LSTM or GRU that takes as data our sequence of actions per student. The LSTM outputs a number of hidden parameters on which we append our fixed features. We then fave a fully connected layer that takes as input our concatenated features and outputs a value. Finally, we use the sigmoid function to output a probability that the student will effectively do a career in STEM.
# 
# Here are the functions explained:
# - **init**: Initializes the model by setting the parameters, creating a RNN layer and a Linear layer (fully connected).
# - **init_hidden**: Creates a zeroed initial hidden state.
# - **forward**: Actual RNN computation.
# - **step**: A step in the training process.
# - **evaluate_val**: Validation set results on the current model.
# - **fit**: Complete training process.

class RNN(nn.Module):
    def __init__(self,
                 input_dim,
                 hidden_dim,
                 fixed_dim,
                 n_layers,
                 bi,
                 use_gpu,
                 dropout,
                 output_dim=1,
                 batch_size=1):
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
        loader = DataLoader(dataset, self.batch_size, num_workers=parameters['num_workers'])
        
        y_preds = []
        y_true = []
        
        self.eval()
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
        self.optimizer = optim.Adamax(self.parameters(), lr=parameters['learning_rate'])
        
        loader = DataLoader(train_dataset, batch_size=self.batch_size, num_workers=parameters['num_workers'], shuffle=True)
        
        e_losses = []
        e_accs = []
        e_aucs = []
        e_mse = []

        e_val_accs = []
        e_val_aucs = []
        e_val_mse = []
        
        e_bar = tqdm(range(parameters['epochs']))
        for e in e_bar:
            self.train()
            e_loss = 0
            preds = []
            targets = []
            val_preds = []
            
            for i, (_, seq, fixed, label) in enumerate(tqdm(loader, leave=False)):
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
            mse = mean_squared_error(targets, preds)
            
            preds[preds >= 0.5] = 1
            preds[preds < 0.5] = 0
            acc = accuracy_score(preds, targets)
            
            e_losses.append(e_loss / (i+1))
            e_accs.append(acc)
            e_aucs.append(auc)
            e_mse.append(mse)

            # Validation set accuracy, AUC and RMSE
            val_acc = None
            val_auc = None
            val_mse = None
            if validation_dataset is not None:
                val_targets, val_preds = self.evaluate_val(validation_dataset)
                val_targets = np.array(val_targets)
                val_preds = np.array(val_preds)
                val_auc = roc_auc_score(val_targets, val_preds)
                val_mse = mean_squared_error(val_targets, val_preds)

                val_preds[val_preds >= 0.5] = 1
                val_preds[val_preds < 0.5] = 0
                val_acc = accuracy_score(val_preds, val_targets)

                e_val_accs.append(val_acc)
                e_val_aucs.append(val_auc)
                e_val_mse.append(val_mse)
            
            e_bar.set_postfix(acc=acc,
                              e_loss=e_losses[-1],
                              auc=auc,
                              mse=mse,
                              val_acc=val_acc,
                              val_auc=val_auc,
                              val_mse=val_mse)
      
        return e_losses, e_accs, e_aucs, e_mse, e_val_accs, e_val_aucs, e_val_mse


# ## Training the network

# In order to train our RNN, we use our previously defined DataLoader and try to minimize error. The optimization function chosen is **Adamax** and the loss function is **Binary Cross-Entropy with Logits**.

for layers in [3]:
    parameters['n_layers'] = layers
    
    for h_dim in [20, 24, 28, 32, 36, 40, 44]:
        parameters['hidden_dim'] = h_dim

        for dropout in [0, 0.1, 0.2]:
            parameters['dropout'] = dropout

            model = RNN(input_dim=parameters['dynamic_dim'],
                        hidden_dim=parameters['hidden_dim'],
                        fixed_dim=parameters['fixed_dim'],
                        n_layers=parameters['n_layers'],
                        bi=parameters['bidirectional'],
                        use_gpu=parameters['use_gpu'],
                        dropout=parameters['dropout'])

            if parameters['use_gpu']:
                model.cuda()

            print('Running with parameters:')
            print(parameters)
            e_losses, e_accs, e_aucs, e_mse, e_val_accs, e_val_aucs, e_val_mse = model.fit(train_dataset)

            # ## Storing results in pickles
            # 
            # The results are stored in a dictionary with the following entries:
            # - **parameters**: A dictionary of parameters for the given result
            # - **losses**: The losses over time
            # - **accs**: Accuracies over time for the training set
            # - **aucs**: ROC AUC over time for the training set
            # - **mse**: Mean Squared Error over time for the training set
            # - **val_accs**: Accuracies over time for the validation set
            # - **val_aucs**: ROC AUC over time for the validation set
            # - **val_mse**: Mean Squared error over time for the validation set

            data_to_store = {
                'parameters': parameters,
                'losses': e_losses,
                'accs': e_accs,
                'aucs': e_aucs,
                'mse': e_mse,
                'val_accs': e_val_accs,
                'val_aucs': e_val_aucs,
                'val_mse': e_val_mse
            }

            pickle_name = 'results_' + str(layers) + '_' + str(h_dim) + '_' + str(dropout)
            save_pickle(data_to_store, pickle_name)
            print('File stored: ' + pickle_name + '.pickle')
            print()
