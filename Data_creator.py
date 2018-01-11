
# coding: utf-8

# # ASSISTments Data Mining Competition 2017 - Optional Semester Project

# ## Imports

# In[1]:


import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
import torch
import torch.nn as nn
from torch.autograd import Variable


# In[2]:


DATA_DIR = 'Data/'


# ## Loading the data

# We choose the columns to keep and load all the data into dataframes

# In[3]:


categorical_features = set(['skill',
                        'problemId',
                        'assignmentId',
                        'assistmentId',
                        'problemType'])

numerical_features = set(['NumActions',
                      'attemptCount',
                      'consecutiveErrorsInRow',
                      'frPast5HelpRequest',
                      'frPast5WrongCount',
                      'frPast8HelpRequest',
                      'frPast8WrongCount',
                      'frTimeTakenOnScaffolding',
                      'frTotalSkillOpportunitiesScaffolding',
                      'hintCount',
                      'hintTotal',
                      'past8BottomOut',
                      'sumRight',
                      'sumTimePerSkill',
                      'timeSinceSkill',
                      'timeTaken',
                      'totalFrAttempted',
                      'totalFrPastWrongCount',
                      'totalFrSkillOpportunities',
                      'totalFrSkillOpportunitiesByScaffolding',
                      'totalFrTimeOnSkill',
                      'totalTimeByPercentCorrectForskill'])

columns_not_keep = set([
    'SY ASSISTments Usage',
    'Prev5count',
    'prev5count',
    'endTime',
    'responseIsChosen',
    'sumTime3SDWhen3RowRight',
    'Ln',
    'Ln-1',
    'actionId'
])


# In[4]:


column_names = set(['AveCarelessness', 'AveCorrect', 'AveKnow', 'AveResBored', 'AveResConf',
       'AveResEngcon', 'AveResFrust', 'AveResGaming', 'AveResOfftask',
       'ITEST_id', 'Ln', 'Ln-1', 'NumActions', 'Prev5count', 'RES_BORED',
       'RES_CONCENTRATING', 'RES_CONFUSED', 'RES_FRUSTRATED', 'RES_GAMING',
       'RES_OFFTASK', 'SY ASSISTments Usage', 'actionId', 'assignmentId',
       'assistmentId', 'attemptCount', 'bottomHint', 'confidence(BORED)',
       'confidence(CONCENTRATING)', 'confidence(CONFUSED)',
       'confidence(FRUSTRATED)', 'confidence(GAMING)', 'confidence(OFF TASK)',
       'consecutiveErrorsInRow', 'correct', 'endTime',
       'endsWithAutoScaffolding', 'endsWithScaffolding', 'frIsHelpRequest',
       'frIsHelpRequestScaffolding', 'frPast5HelpRequest', 'frPast5WrongCount',
       'frPast8HelpRequest', 'frPast8WrongCount', 'frTimeTakenOnScaffolding',
       'frTotalSkillOpportunitiesScaffolding', 'frWorkingInSchool',
       'helpAccessUnder2Sec', 'hint', 'hintCount', 'hintTotal', 'manywrong',
       'original', 'past8BottomOut', 'prev5count', 'problemId', 'problemType',
       'responseIsChosen', 'responseIsFillIn', 'scaffold', 'skill',
       'startTime', 'stlHintUsed', 'sumRight', 'sumTime3SDWhen3RowRight',
       'sumTimePerSkill', 'timeGreater10SecAndNextActionRight',
       'timeGreater5Secprev2wrong', 'timeOver80', 'timeSinceSkill',
       'timeTaken', 'totalFrAttempted', 'totalFrPastWrongCount',
       'totalFrPercentPastWrong', 'totalFrSkillOpportunities',
       'totalFrSkillOpportunitiesByScaffolding', 'totalFrTimeOnSkill',
       'totalTimeByPercentCorrectForskill'])


# In[5]:


columns_keep = column_names - columns_not_keep


# In[6]:


student_logs = pd.concat([
    pd.read_csv(DATA_DIR + 'student_log_' + str(i) + '.csv', usecols=columns_keep) for i in range(1, 11)
], ignore_index=True)

student_logs.head()


# ## Remove Na

# In[7]:


print(student_logs.isnull().any().any())
student_logs = student_logs.fillna(0)
print(student_logs.isnull().any().any())


# ## Scaling numerical features

# In[8]:


scaler = StandardScaler()
numerical_features = list(numerical_features)
student_logs[numerical_features] = scaler.fit_transform(student_logs[numerical_features])


# ## Encoding categorical features

# In[9]:


class MultiColumnLabelEncoder:
    def __init__(self,columns = None):
        self.columns = columns # array of column names to encode

    def fit(self,X,y=None):
        return self # not relevant here

    def transform(self,X):
        '''
        Transforms columns of X specified in self.columns using
        LabelEncoder(). If no columns specified, transforms all
        columns in X.
        '''
        output = X.copy()
        if self.columns is not None:
            for col in self.columns:
                output[col] = LabelEncoder().fit_transform(output[col])
        else:
            for colname,col in output.iteritems():
                output[colname] = LabelEncoder().fit_transform(col)
        return output

    def fit_transform(self,X,y=None):
        return self.fit(X,y).transform(X)


# In[10]:


student_logs_categorical = MultiColumnLabelEncoder(columns = categorical_features).fit_transform(student_logs)
student_logs_categorical.head()


# ## Creating embeddings for categorical features

# In[11]:


dim_embeddings = 3

for column in categorical_features:
    categories_idx = []
    embeddings = []

    for idx in student_logs_categorical[column]:
        categories_idx.append(idx.item())

    embeds = nn.Embedding(len(categories_idx), dim_embeddings) # 3 dimensional embeddings
    lookup_tensor = torch.LongTensor(np.asarray(categories_idx))
    embed = embeds(Variable(lookup_tensor))

    df = pd.DataFrame(embed.data.numpy()).rename(columns={'0': column+'_0', '1': column+'_1', '2':column+'_2'})
    df.columns = [column+'_0', column+'_1', column+'_2']
    student_logs_categorical = pd.concat([student_logs_categorical, df], axis=1)

student_logs_categorical.drop(categorical_features, axis=1, inplace=True)
student_logs_categorical.head()


# ## Keeping only students whom we have labels

# In[12]:


train_labels = pd.read_csv('Data/training_label.csv', index_col='ITEST_id', na_values=-999).sort_index()
train_labels.drop_duplicates(subset=None, keep='first', inplace=True)

train_labels = train_labels.fillna(train_labels['MCAS'].median())
train_labels.head()


# In[13]:


test_labels = pd.read_csv(DATA_DIR + 'validation_test_label.csv', index_col='ITEST_id', na_values=-999).sort_index()
test_labels = test_labels.fillna(train_labels['MCAS'].median())
test_labels.head()


# We only keep actions for students for which we have labels in train_labels and test_labels. We also sort by student ID and by startTime of in order to have a chronological suite of actions

# In[14]:


student_logs_categorical = student_logs_categorical.sort_values(by=['ITEST_id', 'startTime'])
del student_logs_categorical['startTime']


# In[15]:


train_idx = train_labels.index.values
test_idx = test_labels.index.values

student_train_logs = student_logs_categorical[student_logs_categorical['ITEST_id'].isin(train_idx)]
student_test_logs = student_logs_categorical[student_logs_categorical['ITEST_id'].isin(test_idx)]
print('Training data shape:', student_train_logs.shape)
print('Test data shape:', student_test_logs.shape)


# In[16]:


print('Number of students train:', student_train_logs.ITEST_id.unique().shape)
print('Number of students test:', student_test_logs.ITEST_id.unique().shape)


# In[17]:


student_train_logs.head()


# ## Creating a dictionary of sequences 

# In[18]:


fixed_features = ['NumActions',
                  'AveKnow',
                  'AveCarelessness',
                  'AveCorrect',
                  'AveResBored',
                  'AveResEngcon',
                  'AveResConf',
                  'AveResFrust',
                  'AveResOfftask',
                  'AveResGaming']

def create_dict(idx, labels, is_train=True):
    dict_data = {}

    for i in idx:
        sequence = student_logs_categorical[student_logs_categorical['ITEST_id'] == i]
        sequence = sequence.drop(['ITEST_id'], axis=1)
        fixed = sequence[fixed_features]
        fixed = fixed.assign(MCAS=labels.loc[i].MCAS, SchoolId=labels.loc[i].SchoolId)
        sequence = sequence.drop(fixed_features, axis=1)
        
        if is_train:
            target = train_labels.loc[i].isSTEM
            dict_data[i] = (sequence, fixed, target)
        else:
            dict_data[i] = (sequence, fixed)
        
    return dict_data

dict_train = create_dict(train_idx, train_labels)
dict_test = create_dict(test_idx, test_labels, False)


# In[19]:


print(len(dict_train))
print(len(dict_test))


# ## Saving data in pickles

# Finally we save the data into pickles to use them later

# In[20]:


import pickle

def save_pickle(dict_data, name):
    pickle_out = open(DATA_DIR + name + '.pickle', 'wb')
    pickle.dump(dict_data, pickle_out)
    pickle_out.close()


# In[21]:


save_pickle(dict_train, 'student_train_logs')
save_pickle(dict_test, 'student_test_logs')


# ## Aggregated data

# In[22]:


student_train_logs_agg = pd.concat([student_train_logs.groupby('ITEST_id').mean(), student_train_logs.groupby('ITEST_id').std().add_suffix('_std')], axis=1)
student_test_logs_agg = pd.concat([student_test_logs.groupby('ITEST_id').mean(), student_test_logs.groupby('ITEST_id').std().add_suffix('_std')], axis=1)

student_train_logs_agg = student_train_logs_agg.loc[:, (student_train_logs_agg != 0.0).any(axis=0)]
student_test_logs_agg = student_test_logs_agg.loc[:, (student_test_logs_agg != 0.0).any(axis=0)]


# In[23]:


student_train_logs_agg['isSTEM'] = train_labels.apply(lambda row: row.isSTEM, axis=1)
student_train_logs_agg['MCAS'] = train_labels.apply(lambda row: row.MCAS, axis=1).fillna(train_labels.MCAS.mean())
student_train_logs_agg['SchoolId'] = train_labels.apply(lambda row: row.SchoolId, axis=1)

student_test_logs_agg['MCAS'] = test_labels.apply(lambda row: row.MCAS, axis=1).fillna(train_labels.MCAS.mean())
student_test_logs_agg['SchoolId'] = test_labels.apply(lambda row: row.SchoolId, axis=1)

student_test_logs_agg.to_pickle(DATA_DIR + 'student_test_logs_agg')
student_train_logs_agg.to_pickle(DATA_DIR + 'student_train_logs_agg')

