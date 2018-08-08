


import random
import os
import sys
import copy
import math
import time
import pickle
import numpy as np
import pandas as pd
from deap import tools
from deap import base, creator
from six.moves.urllib.request import urlretrieve


# ### Constants Decleration

output_file = 'output'



class Tools :
    
    data_root = ''
    root_url = ''
    chromosomes = {}
    
    @staticmethod
    def download_progress_hook(count, blockSize, totalSize):
        """A hook to report the progress of a download. This is mostly intended for users with
        slow internet connections. Reports every 5% change in download progress."""
        global last_percent_reported
        percent = int(count * blockSize * 100 / totalSize)

        if last_percent_reported != percent:
            if percent % 5 == 0:
                sys.stdout.write("%s%%" % percent)
                sys.stdout.flush()
            else:
                sys.stdout.write(".")
                sys.stdout.flush()
            last_percent_reported = percent
    
    @staticmethod
    def read_df(filename, expected_bytes=None, force=False):
        """Download a file if not present, and make sure it's the right size."""
        dest_filename = os.path.join(Tools.data_root, filename)
        direc = dest_filename[:dest_filename.rfind('/')]
        if not os.path.exists(direc):
            os.makedirs(direc)
        if force or not os.path.exists(dest_filename):
            print('Attempting to download:', filename) 
            filename, _ = urlretrieve(Tools.root_url + filename, dest_filename, reporthook=Tools.download_progress_hook)
            print('\nDownload Complete!')        
        return np.array(pd.read_csv(filename, header=None))
    
    @staticmethod
    def keras_model(input_dim,hiddenNum=40,lr=.1,m=.5) :
        model = Sequential()
        model.add(Dense(hiddenNum, input_dim=input_dim, kernel_initializer='normal', activation='sigmoid'))
        model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
        sgd = SGD(lr=lr, momentum=m)
        # loss could be "mse" too
        model.compile(loss='binary_crossentropy',metrics=['accuracy','binary_accuracy'],optimizer=sgd)
        return lambda: model
    
    @staticmethod
    def save_to_file(path,data) :
        with open(path + '.pkl', 'wb') as f:
            pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
    @staticmethod    
    def load_from_file(path) :
        with open(path + '.pkl', 'rb') as f:
            return pickle.load(f)


# ### Creating Types

# In[11]:


creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)


# ### Qbit Class Implementation

# In[12]:


class Qbit :
    def __init__(self) :
        self.a = random.random()
        self.b = np.sqrt(1 - self.a**2)
        self.bit = None
    
    def __str__(self) :
        return '({}, {})'.format(self.a,self.b)


# ### Reading German Dataset

# In[13]:


x_data = np.array(pd.read_excel(german_data,header=None))
y_data = np.array(pd.read_excel(german_label,header=None))
print('Dataset Shape : {}\nDataset Labels Shape : {}'.format(x_data.shape,y_data.shape))


# ### Initialization

# In[14]:


toolbox = base.Toolbox()
toolbox.register("attribute", Qbit)
toolbox.register("individual", tools.initRepeat, creator.Individual,
                 toolbox.attribute, n=x_data.shape[1]) # Length of each chromosome : Number of Features of Dataset
toolbox.register("population", tools.initRepeat, list, toolbox.individual)


# ### Operators Implementation 

# #### Converting To Bit

# In[15]:


def toBit(ind) :
    "transform qbit to zero or one"
    for qb in ind :
        if np.random.random() < qb.a**2 :
            qb.bit = 0
        else :
            qb.bit = 1
    return ind
toolbox.register("toBit",toBit)


# #### Mutation

# In[16]:


def mutate(ind, indpb) :
    for qb in ind :
        if indpb > np.random.random() :
            qb.a,qb.b = qb.b,qb.a
    return ind
toolbox.register("mutate", mutate, indpb=1)


# #### Crossover

# In[17]:


toolbox.register("mate", tools.cxTwoPoint)


# #### Rotation

# In[18]:


def rotate(ind,b_ind,isGreater) :
    for qb,b_qb in zip(ind,b_ind) :
        dt = 0
        sign = 0
        ri = qb.bit
        bi = b_qb.bit
        positive = qb.a * qb.b > 0
        aZero = not qb.a
        bZero = not qb.b
        # initializing angle and sign of rotation 
        if(isGreater) :
            if not ri and bi :
                dt = np.pi * .05
                if aZero :
                    sign = 1
                elif bZero :
                    sign = 0
                elif positive :
                    sign = -1
                else :
                    sign = 1
            elif ri and not bi :
                dt = np.pi * .025
                if aZero :
                    sign = 0
                elif bZero :
                    sign = 1
                elif positive :
                    sign = 1
                else :
                    sign = -1
            elif ri and bi :
                dt = np.pi * .025
                if aZero :
                    sign = 0
                elif bZero :
                    sign = 1
                elif positive :
                    sign = 1
                else :
                    sign = -1
        else :
            if ri and not bi :
                dt = np.pi * .01
                if aZero :
                    sign = 1
                elif bZero :
                    sign = 0
                elif positive :
                    sign = -1
                else :
                    sign = 1
            elif ri and bi :
                dt = np.pi * .005
                if aZero :
                    sign = 0
                elif bZero :
                    sign = 1
                elif positive :
                    sign = 1
                else :
                    sign = -1

        t = sign * dt
        qb.a,qb.b = np.dot(
            np.array([[np.cos(t),-np.sin(t)],[np.sin(t),np.cos(t)]]),np.array([qb.a,qb.b])
        )
toolbox.register("rotate", rotate)



if __name__ == '__main__' :
    print(Tools.load_from_file('output(0)'))
