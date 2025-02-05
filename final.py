
# coding: utf-8

# # Quantum Inspired Genetic Algorithm For Credit Scoring Problem

# # To Do List
# * Create Diffrent Version Of Algorithm : Diffrent Crossover,Mutation and ...p
# * Add Feature Restriction(Crossover and Population should be modified.)
# * Add Feature Preprocessing
# * MinMaxScaler +
# * Change the k and get the result again
# * Neural network optimization
#     * Using Diffrent Optimizers
#     * Use Diffrent Architecture
# * Speed Up Convergence
#     * Manipulating Population Number & Generation Number
#     * If Population Fitness didn't changed after N generation, end it

# # Done So Far
# * All The Main Functions Implemented
# * all functions debugged
# * Whole Algorithm Implemented
# * MultiProcessing Added
# * Run Multipletimes and average them all

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

# In[7]:


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
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import MinMaxScaler

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 



output_file = 'output'
chromosome_file = 'chromosomes'
skf = StratifiedKFold(n_splits=10)



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
mask = np.zeros(30)
for key in reduced_feature_subset_rank :
    mask[key] = reduced_feature_subset_rank[key]
reduced_feature_subset = sorted(list(set(reduced_feature_subset)))
mask += 1



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
    'mask_best_num':1,
    'mask_evapuration_rate':.1,
    'mask_update_rate':.5,
    'epsilon':0,
    'chrom_mask': mask,
    'workers':50
}


nn_config = {
    # 'lr': np.random.uniform(0.3, 1.0),
    # 'train_cycles': np.random.uniform(300, 600),
    # 'm': np.random.uniform(0.2, 0.7)
    'm':.7,
    'train_cycles':600,
    'lr': .3
}
print(nn_config)




data_root = 'data'
german_data = os.path.join(data_root,'GermanCreditInput.xls')
german_label = os.path.join(data_root,'GermanCreditOutputClass1columnknn.xls')
australian_dataset = os.path.join(data_root,'australian dataset.xlsx')


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
        return model
    
    @staticmethod
    def save_to_file(path,data) :
        with open(path + '.pkl', 'wb') as f:
            pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

    @staticmethod    
    def load_from_file(path) :
        with open(path + '.pkl', 'rb') as f:
            return pickle.load(f)


creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)


class Qbit :
    def __init__(self) :
        self.a = random.random()
        self.b = np.sqrt(1 - self.a**2)
        self.bit = None
    
    def __str__(self) :
        return '({}, {})'.format(self.a,self.b)


x_data = np.array(pd.read_excel(german_data,header=None))
y_data = np.array(pd.read_excel(german_label,header=None))
mx = MinMaxScaler()
mx.fit(x_data)
x_data = mx.transform(x_data)
print('Dataset Shape : {}\nDataset Labels Shape : {}'.format(x_data.shape,y_data.shape))


toolbox = base.Toolbox()
toolbox.register("attribute", Qbit)
toolbox.register("individual", tools.initRepeat, creator.Individual,
                 toolbox.attribute, n=x_data.shape[1]) # Length of each chromosome : Number of Features of Dataset
toolbox.register("population", tools.initRepeat, list, toolbox.individual)



def toBit(ind) :
    "transform qbit to zero or one"
    for qb in ind :
        if np.random.random() < qb.a**2 :
            qb.bit = 0
        else :
            qb.bit = 1
    return ind
toolbox.register("toBit",toBit)

def mutate(ind) :
    rnd = np.random.randint(len(ind))
    ind[rnd].a,ind[rnd].b = ind[rnd].b,ind[rnd].a
    return ind
toolbox.register("mutate", mutate)

toolbox.register("mate", tools.cxTwoPoint)

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

def evaluate(ind,X,Y,train_cycles=600,lr=.3,m=.7) :
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
    
    sel_features = np.array(ind).astype(np.int32)
    hiddenNum = len(sel_features) + np.sum(sel_features)
    string_arr = ''.join(map(str, 1*sel_features))
    sum_val_acc = 0
    p_X = X[:,sel_features==1]
    if string_arr not in Tools.chromosomes :
        for train_index, test_index in skf.split(X, Y):
            model = Tools.keras_model(np.sum(sel_features),int(hiddenNum),lr,m)
            hist = model.fit(p_X[train_index,:],Y[train_index,:], validation_data=(p_X[test_index,:],Y[test_index,:]),epochs=int(train_cycles),batch_size=int(X.shape[0]),verbose=0)
            ev = model.evaluate(p_X[test_index,:],Y[test_index,:],verbose=0)
            sum_val_acc += ev[1]
            del model
        Tools.chromosomes[string_arr] = sum_val_acc/10
    return (Tools.chromosomes[string_arr],)

toolbox.register("evaluate", evaluate,X=x_data,Y=y_data,**nn_config)

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


def catastrophe(best_ind,pop_size) :
    pop = toolbox.population(n=pop_size)
    pop[np.random.randint(0,pop_size)] = toolbox.clone(best_ind)
    return pop
toolbox.register("catastrophe", catastrophe, pop_size=genetic_config['pop_size'])


# #### Mask update and collapse

# In[30]:


def mask_update(mask,best_ind,mask_update_rate=.5,mask_evapuration_rate=.1,inline=True) :
    for i in range(len(mask)) :
        mask[i] *= (1-mask_evapuration_rate)
        mask[i] += best_ind[i].bit*mask_update_rate


# In[31]:


def mask_collapse(mask,epsilon=1) :
    collapsed_mask = np.ndarray(len(mask))
    max_val = max(mask) +epsilon
    for i in range(len(mask)) :
        collapsed_mask[i] = 1 if np.random.random() < mask[i]/max_val else 0
    return collapsed_mask


# #### Evaluating Whole Population

# In[81]:


def evaluate_pop(pop,collapsed_mask=None,multiprocessing=False,workers=20) :
    best_ind = None
    # Collapsing Individual Bits
    for i,ind in enumerate(pop):
        toolbox.toBit(ind)
    # Masking Individual Bit Values
    print(pop[0].fitness.values)
    if type(collapsed_mask) != type(None):
        masked_pop = [[qb.bit and bit for qb,bit in zip(ind,collapsed_mask)] for ind in pop]
    else :
        masked_pop = [[qb.bit for qb in ind] for ind in pop]
    print(np.sum([[qb.bit for qb in ind] for ind in pop],axis=1))
    print(np.sum(masked_pop,axis=1))
    if multiprocessing :
        fitnesses = None
        with contextlib.closing(Pool(processes=workers)) as pool:
            fitnesses = pool.map_async(toolbox.evaluate, (ind for ind in masked_pop))
            fitnesses = fitnesses.get()
        for ind,fitness in zip(pop,fitnesses):
            ind.fitness.values = fitness
            if not best_ind or best_ind.fitness.values[0] < ind.fitness.values[0] :
                best_ind = ind
        Tools.save_to_file(chromosome_file,Tools.chromosomes)
        with open('fitnesses.txt','a') as f :
            f.write(str(fitnesses) + '\n')
    else :
        # Evaluate Individual
        for ind,mask_ind,i in zip(pop,masked_pop,range(len(pop))):
            print('%{}'.format(float(i)/len(pop)))
            ind.fitness.values = toolbox.evaluate(mask_ind)
            if not best_ind or best_ind.fitness.values[0] < ind.fitness.values[0] :
                best_ind = ind
            if i % 10 == 0 :
                Tools.save_to_file(chromosome_file,Tools.chromosomes)
    print(pop[0].fitness.values)
    return best_ind


# ## Implementation

# ### Quantum Algorithm

# In[27]:


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


# ## Quantum MultiProcess Algorithm

# In[101]:


def multiprocess_main(pop_size,iter_num,n_max,m_max,
                    max_feature_num,min_feature_num,
                    pm,pc,pmm,pcc,chrom_mask,mask_best_num=1,
                    mask_evapuration_rate=.1,mask_update_rate=.5,
                    epsilon=1,multiprocessing=False,workers=20):
    if os.path.exists(chromosome_file+'.pkl') :
        Tools.chromosomes = Tools.load_from_file(chromosome_file)
        print('chromosomes loaded : {}'.format(Tools.chromosomes))
    best_fits = np.array([])
    best_same_iter = 0
    best_ind = None
    current_best_ind = None
    mask = copy.deepcopy(chrom_mask)
    
    pop = toolbox.population(n=pop_size)
    
    # Callapse Mask
    collapsed_mask = mask_collapse(mask,epsilon)
    print(collapsed_mask)
    # Evaluating whole population
    best_ind = current_best_ind = evaluate_pop(pop,collapsed_mask,multiprocessing,workers)
    
    # Updating And Evapurating Mask Values
    mask_update(mask,best_ind,mask_evapuration_rate=mask_evapuration_rate,mask_update_rate=mask_update_rate)
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
            
            # Callapse Mask
            collapsed_mask = mask_collapse(mask,epsilon)
            
            # Evaluating whole population
            current_best_ind = evaluate_pop(offspring,collapsed_mask,multiprocessing,workers)
            
            # Updating And Evapurating Mask Values
            mask_update(mask,best_ind,mask_evapuration_rate=mask_evapuration_rate,mask_update_rate=mask_update_rate)
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
            current_best_ind = None
            # Callapse Mask
            collapsed_mask = mask_collapse(mask,epsilon)
            
            # Evaluating whole population
            current_best_ind = evaluate_pop(offspring,collapsed_mask,multiprocessing,workers)
            
            # Updating And Evapurating Mask Values
            mask_update(mask,best_ind,mask_evapuration_rate=mask_evapuration_rate,mask_update_rate=mask_update_rate)
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
        
        # Callapse Mask
        collapsed_mask = mask_collapse(mask,epsilon)

        # Evaluating whole population
        current_best_ind = evaluate_pop(pop,collapsed_mask,multiprocessing,workers)
        
        # Updating And Evapurating Mask Values
        mask_update(mask,best_ind,mask_evapuration_rate=mask_evapuration_rate,mask_update_rate=mask_update_rate)
        print('Second Evaluation Time : {}'.format(time.time() - start))    
        best_fits = np.append(best_fits,current_best_ind.fitness.values[0])
        if current_best_ind.fitness.values[0] > best_ind.fitness.values[0] :
            best_ind = toolbox.clone(current_best_ind)
            best_same_iter = 0
        else :
            best_same_iter += 1
    return best_fits,best_ind

def new_multiprocess_main(pop_size, iter_num, n_max, m_max,
                          max_feature_num, min_feature_num,
                          pm, pc, pmm, pcc, chrom_mask, mask_best_num=1,
                          mask_evapuration_rate=.1, mask_update_rate=.5,
                          epsilon=1, multiprocessing=False, workers=20,
                          have_catas=True):

    best_fits = np.array([])
    best_same_iter = 0
    best_ind = None
    current_best_ind = None
    mask = copy.deepcopy(chrom_mask)

    pop = toolbox.population(n=pop_size)

    # Callapse Mask
    collapsed_mask = mask_collapse(mask, epsilon)
    print(collapsed_mask)
    # Evaluating whole population
    best_ind = current_best_ind = evaluate_pop(
        pop, collapsed_mask, multiprocessing, workers
        )

    # Updating And Evapurating Mask Values
    mask_update(mask, best_ind, mask_evapuration_rate=mask_evapuration_rate,
                mask_update_rate=mask_update_rate)
    best_fits = np.append(best_fits, best_ind.fitness.values[0])

    for generation in range(1, iter_num):
        print('--------------------generation : {} ------------------'.format(generation))
        print('best fitness : {}'.format(best_ind.fitness.values[0]))
        print('best current fit : {}'.format(
            current_best_ind.fitness.values[0]))
        p_mutate = pm if best_same_iter < n_max else pmm
        p_mate = pc if best_same_iter < n_max else pcc
        offspring = toolbox.select(pop)

        # Apply crossover on the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < p_mate:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        # Apply mutation on the offspring
        for mutant in offspring:
            if random.random() < p_mutate:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        pop[:] = offspring

        # Evaluate Individual
        start = time.time()

        # Callapse Mask
        collapsed_mask = mask_collapse(mask, epsilon)

        # Evaluating whole population
        current_best_ind = evaluate_pop(
            pop, collapsed_mask, multiprocessing, workers)

        # Updating And Evapurating Mask Values
        mask_update(mask, best_ind, mask_evapuration_rate=mask_evapuration_rate,
                    mask_update_rate=mask_update_rate)

        print('Second Evaluation Time : {}'.format(time.time() - start))

        best_fits = np.append(best_fits, current_best_ind.fitness.values[0])

        if current_best_ind.fitness.values[0] > best_ind.fitness.values[0]:
            best_ind = toolbox.clone(current_best_ind)
            best_same_iter = 0
        else:
            best_same_iter += 1

        if best_same_iter >= m_max and have_catas:
            offspring[:] = toolbox.catastrophe(best_ind)
            best_same_iter = 0

    return best_fits, best_ind
# In[47]:


def multiple_run(genetic_config,number_of_run=10) :
    if os.path.exists(chromosome_file+'.pkl') :
        Tools.chromosomes = Tools.load_from_file(chromosome_file)
    out = [None for _ in range(number_of_run)]
    for i in range(number_of_run) :
        out[i] = multiprocess_main(**genetic_config)
        print('{} run ended with fitness : {} '.format(i,out[i]))
        Tools.save_to_file('{}({})'.format(output_file,i),out[i])
    return np.average([o[1].fitness.values[0] for o in out]) 


# In[102]:


if __name__ == '__main__' :
    out = new_multiprocess_main(**genetic_config,multiprocessing=True)
    Tools.save_to_file(output_file+'1',out)


# In[99]:


# if __name__ == '__main__' :
#     out = multiple_run(genetic_config)
#     print('Average is : {}'.format(out))
#     Tools.save_to_file('multi_run_' + output_file,out)

