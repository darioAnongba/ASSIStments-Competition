
# coding: utf-8

# # ASSISTments Data Mining Competition 2017 - Optional Semester Project
# 
# ## Data Preprocessing
# 
# The purpose of this notebook is to explain and handle the data preprocessing needed to fuel the Deep Learning Model used for this project. The Model can be found in the "`Report`" notebook.

# ## Imports

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
import torch
import torch.nn as nn
from torch.autograd import Variable

DATA_DIR = 'Data/'

# ## Loading the data

# The data is composed of multiple types of features that we need to differentiate. The types are the following:
# 
# - **Floating averages**: Values in the range [0, 1] representing averages. *These values can be used without transformation*. Ex: `AveCorrect`
# - **Categorical**: Values representing categories or indexes. These values need to be treated carefully because a value of "1184832848" represents an index and not a numerical value. This means that *we need to transform these values in order to use them in a neural network using embeddings*. Ex: `assignmentId`
# - **Numerical**: Values representing integers. *These values need to be scaled* to a range (like [0, 1] or [-1, 1]) in order to be used in a neural network. Ex: `timeTaken`
# - **Binary**: Values representing booleans. These values can be used without *transformation*. Ex: `correct`
# 
# In addition, some values are discarded because of missing values, non-usefulness (their information is already contained in another feature) or uniqueness (`actionId` is different for each action so there is no information to extract from that feature).

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
    'actionId'])

columns_keep = column_names - columns_not_keep


student_logs = pd.concat([
    pd.read_csv(DATA_DIR + 'student_log_' + str(i) + '.csv', usecols=columns_keep) for i in range(1, 11)
], ignore_index=True)

student_logs.head()


# ## Remove Na
# 
# We remove actions containing NA values (normally, there shouldn't be any)

student_logs = student_logs.fillna(0)
print('NA values removed')

# ## Scaling numerical features
# 
# We need to scale numerical values using different scaling methods:
# - **StandardScaler**: Standardize features by removing the mean and scaling to unit variance.
# - **MaxMinScaler**: Transforms features by scaling each feature to a given range.
# 
# After testing with both methods, we chose a MaxMinScaler with range [-1, 1]

#scaler = MinMaxScaler((-1, 1))
scaler = StandardScaler()
numerical_features = list(numerical_features)
student_logs[numerical_features] = scaler.fit_transform(student_logs[numerical_features])

# ## Encoding categorical features
# 
# We need to encode categorical features into integer indexes. There are two types of categorical features:
# - **Indexes**: Integers used to define a feature of an action like the problem id or the assignment id (`assignmentId`)
# - **Categories**: Strings used to state that an action is part of a given category, like the skill being tested (`skill`)
# 
# With both types the problem of label encoding is the same as we need to transform the values into integers spanning from 0 to the amount of different categories for that feature.

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


student_logs_categorical = MultiColumnLabelEncoder(columns = categorical_features).fit_transform(student_logs)

# ## Creating embeddings for categorical features
# 
# Now that each category has been encoded into an integer index, we can apply vector embeddings methods to transform each value into a X dimensional vector. We chose vectors of size 3.
# 
# We also need to drop the "old" column containing the indexes and only keep the new embedding features

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

# ## Keeping only students for whom we have labels
# 
# Some students do not have any labels assigned so they are useless for supervised learning

train_labels = pd.read_csv('Data/training_label.csv', index_col='ITEST_id', na_values=-999).sort_index()
train_labels.drop_duplicates(subset=None, keep='first', inplace=True)

train_labels = train_labels.fillna(train_labels['MCAS'].median())

test_labels = pd.read_csv(DATA_DIR + 'validation_test_label.csv', index_col='ITEST_id', na_values=-999).sort_index()
test_labels = test_labels.fillna(train_labels['MCAS'].median())

# We only keep actions for students for which we have labels in train_labels and test_labels. We also sort by student ID and by startTime of in order to have a chronological suite of actions

student_logs_categorical = student_logs_categorical.sort_values(by=['ITEST_id', 'startTime'])
del student_logs_categorical['startTime']

train_idx = train_labels.index.values
test_idx = test_labels.index.values

student_train_logs = student_logs_categorical[student_logs_categorical['ITEST_id'].isin(train_idx)]
student_test_logs = student_logs_categorical[student_logs_categorical['ITEST_id'].isin(test_idx)]
print('Training data shape:', student_train_logs.shape)
print('Test data shape:', student_test_logs.shape)

print('Number of students train:', student_train_logs.ITEST_id.unique().shape)
print('Number of students test:', student_test_logs.ITEST_id.unique().shape)

# ## Creating a dictionary of sequences 
# 
# Instead of storing all action sequences into a single dataframe, we separate actions par student and create a dictionary from student id to a sequence of actions. To be exact, the values of the dictionary are arrays of size 3 containing:
# - **Sequence of dynamic features**: Features that are different for every student action. The result is a pandas dataframe.
# - **Sequence of static features**: Features that stay the same for every action of a student (averages, school id and MCAS). The result is a Pandas Series.
# - **label**: If yes or no the student has chosen a STEM career. The result is a boolean (0 or 1).

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
        fixed = fixed.assign(MCAS=labels.loc[i].MCAS, SchoolId=labels.loc[i].SchoolId).iloc[0]
        sequence = sequence.drop(fixed_features, axis=1)
        
        if is_train:
            target = train_labels.loc[i].isSTEM
            dict_data[i] = (sequence, fixed, target)
        else:
            dict_data[i] = (sequence, fixed)
        
    return dict_data

dict_train = create_dict(train_idx, train_labels)
dict_test = create_dict(test_idx, test_labels, False)

# ## Saving data in pickles

# Finally we save the data into pickles to use them later.

import pickle

def save_pickle(dict_data, name):
    pickle_out = open(DATA_DIR + name + '.pickle', 'wb')
    pickle.dump(dict_data, pickle_out)
    pickle_out.close()

save_pickle(dict_train, 'student_train_logs')
save_pickle(dict_test, 'student_test_logs')

print('Data successfully saved into pickles')