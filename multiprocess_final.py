
# coding: utf-8

# # Quantum Inspired Genetic Algorithm For Credit Scoring Problem

# # To Do List
# * Create Diffrent Version Of Algorithm : Diffrent Crossover,Mutation and ...p
# * Add Feature Restriction(Crossover and Population should be modified.)
# * Add Feature Preprocessing
# * MinMaxScaler +
# * Add MultiProcessing

# # Done So Far
# * All The Main Functions Implemented
# * all functions debugged
# * Whole Algorithm Implemented

# ## contents
# * Preconfiguration
#     * Importing Libraries
#     * Constants Decleration
#         * Genetic Algorithm Configurations
#         * Neural Network Configurations
#         * Reduced Features Announced By Credit Scorring Essay
#         * Datasets Path
#     * Tools Class Implementation
#     * Creating Types
#     * Qbit Class Implementation
#     * Reading Data
#     * Initialization
#     * Operators
#         * Converting To Bit
#         * Mutation
#         * Crossover
#         * Selection
#         * Rotation
#         * Catastroph
#         * Fitness Calculation
#         
# * Implementation
#     * Quantum Algorithm

# ## Preconfiguration

# ### Importing Libraries

# In[2]:


from __future__ import print_function
import random
import os
import sys
import copy
import math
import time
import pickle
import contextlib
import numpy as np
import pandas as pd
import multiprocessing as mp
from deap import tools
from deap import base, creator
from sklearn.cluster import KMeans
from multiprocessing import Pool
from six.moves.urllib.request import urlretrieve
from keras.models import Sequential
from keras.layers import Input, Dense, Activation
from keras.optimizers import adam, SGD
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import KFold, cross_val_score


# ### Constants Decleration

# In[5]:


output_file = 'output'
# pop_size = 100
# gen_num = 100 # it could be 300 too
# n_max = 15
# m_max = 25
# pc = 0.9
# pm = 0.01
# pcc = (1 - pc) * random.random() + pc
# pmm = (2*pm - pm) * random.random() + pm


# #### Genetic Algorithm Configurations`

# In[6]:


pc = 0.9
pm = 0.01
genetic_config = { 
    'max_feature_num': 12,
    'min_feature_num': 5,
    'pop_size': 100,
    'iter_num': 100,
    'n_max': 15,
    'm_max': 25,
    'pm': 0.01,
    'pc': 0.9,
    'pmm': (2*pm - pm) * random.random() + pm,
    'pcc': (1 - pc) * random.random() + pc,
#     'selection' : '',
#     'crossover' : '',
#     'mutation' : '',
#     'rotation' : True,
}


# #### Neural Network Configurations


nn_config = {
    # 'lr': np.random.uniform(0.3, 1.0),
    # 'train_cycles': np.random.uniform(300, 600),
    # 'm': np.random.uniform(0.2, 0.7)
    'm':.7,
    'train_cycles':600,
    'lr': .3
}
print(nn_config)

# #### Reduced Features Announced By Credit Scorring Essay

# In[8]:


reduced_feature_config = {
    'IG': [ 0,  1,  2,  6,  9, 10, 11, 19, 20, 21, 22, 24],
    'gain_ratio': [ 0,  1,  2,  4,  9, 10, 19, 20, 21, 22, 24, 29],
    'correlation': [ 0,  1,  2,  4,  6,  9, 10, 11, 19, 20, 22, 24],
    'voting': [ 0,  1,  2,  9, 10, 19, 20, 22, 24],
    'current_solution': [ 0,  1,  2,  3, 10, 12, 13, 14, 17, 19, 20, 27, 29]
}
reduced_feature_subset = []
for key in reduced_feature_config : 
    reduced_feature_subset += reduced_feature_config[key]
reduced_feature_subset_rank = {}
for feature in reduced_feature_subset:
    if feature in reduced_feature_subset_rank:
        reduced_feature_subset_rank[feature] += 1
    else :
        reduced_feature_subset_rank[feature] = 1
reduced_feature_subset = sorted(list(set(reduced_feature_subset)))


# #### Datasets Path

# In[9]:


data_root = 'data'
german_data = os.path.join(data_root,'GermanCreditInput.xls')
german_label = os.path.join(data_root,'GermanCreditOutputClass1columnknn.xls')
australian_dataset = os.path.join(data_root,'australian dataset.xlsx')


# ### Tools Class Implementation

# In[10]:


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


# #### Fitness Calculation

# In[19]:


def evaluate(ind,X,Y,train_cycles=450,lr=.1,m=.5) :
    """Train one layer feedforward neural network
    Args :
       X : training data
       Y : training label
       hiddenNum : number of hidden units of hidden layer
       trainCycles : number of training cycles
       lr : learning rate of nueral network
       m: momentum of neural network
    Returns :
       'float' accuracy
    """
    sel_features = np.array([qb.bit for qb in ind])
    hiddenNum = len(sel_features) + np.sum(sel_features)
    string_arr = ''.join(map(str, 1*sel_features))
    if string_arr not in Tools.chromosomes :
        model = Tools.keras_model(np.sum(sel_features),int(hiddenNum),lr,m)
        classifier = KerasClassifier(build_fn=model, epochs=int(train_cycles),batch_size=int(X.shape[0]),verbose=0)
        Tools.chromosomes[string_arr] = np.mean(cross_val_score(classifier, X[:,sel_features==1], Y, cv=10,verbose=0))
    
    return (Tools.chromosomes[string_arr],)

toolbox.register("evaluate", evaluate,X=x_data,Y=y_data,**nn_config)


# #### Selection

# In[20]:


def select(pop,pop_size) :    
    # Roulette selection
    offsprings = list(map(toolbox.clone,tools.selRoulette(pop,pop_size)))
    # Elite selection
    max_os_fit = np.max([ind.fitness.values[0] for ind in offsprings])
    max_pop_fit = np.max([ind.fitness.values[0] for ind in pop])
    replace_choices = list(range(pop_size))
    
    if max_pop_fit > max_os_fit :
        for ind in sorted(pop, key=lambda x: x.fitness.values[0],reverse=True) :
            if ind.fitness.values[0] > max_os_fit :
                choice = np.random.choice(replace_choices)
                offsprings[choice] = toolbox.clone(ind)
                replace_choices.remove(choice) # To Stop replacing the best ones that we already replaced
            else :
                break;
                
    return offsprings 

toolbox.register("select", select, pop_size=genetic_config['pop_size'])


# In[21]:


# pop = toolbox.population(n=4)
# CXPB, MUTPB, NGEN = 0.5, 0.2, 40
# best_ind = None
# # Evaluate the entire population
# for ind in pop:
#     toolbox.toBit(ind)
#     ind.fitness.values = toolbox.evaluate(ind)
#     if not best_ind or best_ind.fitness.values[0] < ind.fitness.values[0] :
#         best_ind = ind
        
# offspring = toolbox.select(pop)
# for qb in offspring[0] :
#     print(qb)
# toolbox.rotate(offspring[0],best_ind,offspring[0].fitness.values[0] < best_ind.fitness.values[0])
# print('-------------------------')
# for qb in offspring[0] :
#     print(qb)


# #### Catastroph

# In[22]:


def catastrophe(best_ind,pop_size) :
    pop = toolbox.population(n=pop_size)
    pop[np.random.randint(0,pop_size)] = toolbox.clone(best_ind)
    return pop
toolbox.register("catastrophe", catastrophe, pop_size=genetic_config['pop_size'])


# ## Implementation

# ### Quantum Algorithm

# In[23]:


def main(pop_size,iter_num,n_max,m_max,
        max_feature_num,min_feature_num,
        pm,pc,pmm,pcc):
    
    best_fits = np.array([])
    best_same_iter = 0
    best_ind = None
    current_best_ind = None
    
    pop = toolbox.population(n=pop_size)
    
    # Evaluate the entire population
    for i,ind in enumerate(pop):
        print('%{}'.format(float(i)/pop_size))
        toolbox.toBit(ind)
        ind.fitness.values = toolbox.evaluate(ind)
        if not best_ind or best_ind.fitness.values[0] < ind.fitness.values[0] :
            best_ind = ind
            current_best_ind = ind
    best_fits = np.append(best_fits,best_ind.fitness.values[0])
        
    for generation in range(1,iter_num) :   
        print('--------------------generation : {} ------------------'.format(generation))
        print('best fitness : {}'.format(best_ind.fitness.values[0]))
        print('best current fit : {}'.format(current_best_ind.fitness.values[0]))
        if best_same_iter < n_max :
            offspring = toolbox.select(pop)
            
            # Apply crossover on the offspring
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < pc:
                    toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values
                    
            # Apply mutation on the offspring      
            for mutant in offspring:
                if random.random() < pm:
                    toolbox.mutate(mutant)
                    del mutant.fitness.values

            # Evaluate Individual 
            current_best_ind = None
            for i,ind in enumerate(offspring):
                print('%{}'.format(float(i)/pop_size))
                toolbox.toBit(ind)
                ind.fitness.values = toolbox.evaluate(ind)
                if not current_best_ind or current_best_ind.fitness.values[0] < ind.fitness.values[0] :
                    current_best_ind = ind
            
            if current_best_ind.fitness.values[0] > best_ind.fitness.values[0] :
                best_ind = toolbox.clone(current_best_ind)
                best_same_iter = 0
            if best_same_iter < m_max :
                for ind in offspring :
                    toolbox.rotate(ind,best_ind,best_ind.fitness.values[0] > ind.fitness.values[0])
            else :
                offspring[:] = toolbox.catastrophe(best_ind)
                best_same_iter = 0
        else :
            offspring = toolbox.select(pop)

            # Apply crossover on the offspring
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < pcc:
                    toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values

            # Apply mutation on the offspring      
            for mutant in offspring:
                if random.random() < pmm:
                    toolbox.mutate(mutant)
                    del mutant.fitness.values

            # Evaluate Individual 
            current_best_ind = None
            for ind in offspring:
                toolbox.toBit(ind)
                ind.fitness.values = toolbox.evaluate(ind)
                if not current_best_ind or current_best_ind.fitness.values[0] < ind.fitness.values[0] :
                    current_best_ind = ind

            if current_best_ind.fitness.values[0] > best_ind.fitness.values[0] :
                best_ind = toolbox.clone(current_best_ind)
                best_same_iter = 0
            if best_same_iter < m_max :
                for ind in offspring :
                    toolbox.rotate(ind,best_ind,best_ind.fitness.values[0] > ind.fitness.values[0])
            else :
                offspring[:] = toolbox.catastrophe(best_ind)
                best_same_iter = 0
                
        pop[:] = offspring
        # Evaluate Individual 
        current_best_ind = None
        for i,ind in enumerate(offspring):
            print('%{}'.format(float(i)/pop_size))
            toolbox.toBit(ind)
            ind.fitness.values = toolbox.evaluate(ind)
            if not current_best_ind or current_best_ind.fitness.values[0] < ind.fitness.values[0] :
                current_best_ind = ind
            
        best_fits = np.append(best_fits,current_best_ind.fitness.values[0])
        if current_best_ind.fitness.values[0] > best_ind.fitness.values[0] :
            best_ind = toolbox.clone(current_best_ind)
            best_same_iter = 0
        else :
            best_same_iter += 1
    return best_fits,best_ind


# ## Multiprocess Quantum Algorithm

# In[3]:


def multiprocess_main(pop_size,iter_num,n_max,m_max,
        max_feature_num,min_feature_num,
        pm,pc,pmm,pcc):
    
    best_fits = np.array([])
    best_same_iter = 0
    best_ind = None
    current_best_ind = None
    
    pop = toolbox.population(n=pop_size)
    
    # Evaluate the entire population
    for i,ind in enumerate(pop):
        toolbox.toBit(ind)
    with contextlib.closing(Pool(processes=25)) as pool:
        fitnesses = pool.map_async(toolbox.evaluate, (ind for ind in pop))
        fitnesses = fitnesses.get()
    for ind,fitness in zip(pop,fitnesses):
        ind.fitness.values = fitness
        if not best_ind or best_ind.fitness.values[0] < ind.fitness.values[0] :
            best_ind = ind
            current_best_ind = ind
    best_fits = np.append(best_fits,best_ind.fitness.values[0])
        
    for generation in range(1,iter_num) :   
        print('--------------------generation : {} ------------------'.format(generation))
        print('best fitness : {}'.format(best_ind.fitness.values[0]))
        print('best current fit : {}'.format(current_best_ind.fitness.values[0]))
        if best_same_iter < n_max :
            offspring = toolbox.select(pop)
            
            # Apply crossover on the offspring
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < pc:
                    toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values
                    
            # Apply mutation on the offspring      
            for mutant in offspring:
                if random.random() < pm:
                    toolbox.mutate(mutant)
                    del mutant.fitness.values

            start = time.time()
            # Evaluate Individual 
            current_best_ind = None
            for i,ind in enumerate(offspring):
                toolbox.toBit(ind)
            
            with contextlib.closing(Pool(processes=25)) as pool:
                fitnesses = pool.map_async(toolbox.evaluate, (ind for ind in offspring))
                fitnesses = fitnesses.get()
                
            for ind,fitness in zip(offspring,fitnesses):
                ind.fitness.values = fitness
                if not current_best_ind or current_best_ind.fitness.values[0] < ind.fitness.values[0] :
                    current_best_ind = ind
            print('First Evaluation Time : {}'.format(time.time() - start))
            if current_best_ind.fitness.values[0] > best_ind.fitness.values[0] :
                best_ind = toolbox.clone(current_best_ind)
                best_same_iter = 0
            if best_same_iter < m_max :
                for ind in offspring :
                    toolbox.rotate(ind,best_ind,best_ind.fitness.values[0] > ind.fitness.values[0])
            else :
                offspring[:] = toolbox.catastrophe(best_ind)
                best_same_iter = 0
        else :
            offspring = toolbox.select(pop)

            # Apply crossover on the offspring
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < pcc:
                    toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values

            # Apply mutation on the offspring      
            for mutant in offspring:
                if random.random() < pmm:
                    toolbox.mutate(mutant)
                    del mutant.fitness.values
            
            start = time.time()
            # Evaluate Individual 
            current_best_ind = None
            for i,ind in enumerate(offspring):
                toolbox.toBit(ind)
            
            with contextlib.closing(Pool(processes=25)) as pool:
                fitnesses = pool.map_async(toolbox.evaluate, (ind for ind in offspring))
                fitnesses = fitnesses.get()
                
            for ind,fitness in zip(offspring,fitnesses):
                ind.fitness.values = fitness
                if not current_best_ind or current_best_ind.fitness.values[0] < ind.fitness.values[0] :
                    current_best_ind = ind
            print('First Evaluation Time : {}'.format(time.time() - start))
            if current_best_ind.fitness.values[0] > best_ind.fitness.values[0] :
                best_ind = toolbox.clone(current_best_ind)
                best_same_iter = 0
            if best_same_iter < m_max :
                for ind in offspring :
                    toolbox.rotate(ind,best_ind,best_ind.fitness.values[0] > ind.fitness.values[0])
            else :
                offspring[:] = toolbox.catastrophe(best_ind)
                best_same_iter = 0
                
        pop[:] = offspring

        # Evaluate Individual 
        start = time.time()
        current_best_ind = None
        for i,ind in enumerate(pop):
            toolbox.toBit(ind)

        with contextlib.closing(Pool(processes=25)) as pool:
            fitnesses = pool.map_async(toolbox.evaluate, (ind for ind in pop))
            fitnesses = fitnesses.get()
        print(fitnesses)
        for ind,fitness in zip(pop,fitnesses):
            ind.fitness.values = fitness
            if not current_best_ind or current_best_ind.fitness.values[0] < ind.fitness.values[0] :
                current_best_ind = ind
        print('Second Evaluation Time : {}'.format(time.time() - start))    
        best_fits = np.append(best_fits,current_best_ind.fitness.values[0])
        if current_best_ind.fitness.values[0] > best_ind.fitness.values[0] :
            best_ind = toolbox.clone(current_best_ind)
            best_same_iter = 0
        else :
            best_same_iter += 1
    return best_fits,best_ind


# In[ ]:


if __name__ == '__main__' :
    out = multiprocess_main(**genetic_config)
    Tools.save_to_file(output_file,out)

