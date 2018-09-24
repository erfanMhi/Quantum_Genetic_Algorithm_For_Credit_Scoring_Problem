def new_multiprocess_main(pop_size,iter_num,n_max,m_max,
                    max_feature_num,min_feature_num,
                    pm,pc,pmm,pcc,chrom_mask,mask_best_num=1,
                    mask_evapuration_rate=.1,mask_update_rate=.5,
                    epsilon=1):
    
    best_fits = np.array([])
    best_same_iter = 0
    best_ind = None
    current_best_ind = None
    mask = copy.deepcopy(chrom_mask)
    
    pop = toolbox.population(n=pop_size)
    
    # Callapse Mask
    collapsed_mask = mask_collapse(mask,epsilon)
    # Collapsing Individual Bits
    for i,ind in enumerate(pop):
        toolbox.toBit(ind)
    # Masking Individual Bit Values
    masked_pop = [[qb.bit and bit for qb in zip(ind,collapsed_mask)] for ind in pop]
    # Calculating Fitness Of Individuals In Parallel
    with contextlib.closing(Pool(processes=25)) as pool:
        fitnesses = pool.map_async(toolbox.evaluate, (ind for ind in masked_pop))
        fitnesses = fitnesses.get()
    for ind,fitness in zip(pop,fitnesses):
        ind.fitness.values = fitness
        if not best_ind or best_ind.fitness.values[0] < ind.fitness.values[0] :
            best_ind = ind
            current_best_ind = ind
    # Updating And Evapurating Mask Values
    mask_update(mask,best_ind,mask_evapuration_rate=.1,mask_update_rate=.5)
    best_fits = np.append(best_fits,best_ind.fitness.values[0])
    
    
    for generation in range(1,iter_num) :   
        print('--------------------generation : {} ------------------'.format(generation))
        print('best fitness : {}'.format(best_ind.fitness.values[0]))
        print('best current fit : {}'.format(current_best_ind.fitness.values[0])) 
        p_mate = pm if 
        p_mutate = pc if best_same_iter < n_max else pcc
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

            
            # start = time.time()
            # current_best_ind = None
            # # Callapse Mask
            # collapsed_mask = mask_collapse(mask,epsilon)
            # # Collapsing Individual Bits
            # for i,ind in enumerate(pop):
            #     toolbox.toBit(ind)
            # # Masking Individual Bit Values
            # masked_offspring = [[qb.bit and bit for qb in zip(ind,collapsed_mask)] for ind in offspring]
            # # Calculating Fitness Of Individuals In Parallel
            # with contextlib.closing(Pool(processes=20)) as pool:
            #     fitnesses = pool.map_async(toolbox.evaluate, (ind for ind in masked_offspring))
            #     fitnesses = fitnesses.get()
                
            # for ind,fitness in zip(offspring,fitnesses):
            #     ind.fitness.values = fitness
            #     if not current_best_ind or current_best_ind.fitness.values[0] < ind.fitness.values[0] :
            #         current_best_ind = ind
            # # Updating And Evapurating Mask Values
            # mask_update(mask,best_ind,mask_evapuration_rate=.1,mask_update_rate=.5)
            # print('First Evaluation Time : {}'.format(time.time() - start))
            # if current_best_ind.fitness.values[0] > best_ind.fitness.values[0] :
            #     best_ind = toolbox.clone(current_best_ind)
            #     best_same_iter = 0
            # if best_same_iter < m_max :
            #     for ind in offspring :
            #         toolbox.rotate(ind,best_ind,best_ind.fitness.values[0] > ind.fitness.values[0])
            # else :
            #     offspring[:] = toolbox.catastrophe(best_ind)
            #     best_same_iter = 0
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
            
            # start = time.time()
            # current_best_ind = None
            # # Callapse Mask
            # collapsed_mask = mask_collapse(mask,epsilon)
            # # Collapsing Individual Bits
            # for i,ind in enumerate(pop):
            #     toolbox.toBit(ind)
            # # Masking Individual Bit Values
            # masked_offspring = [[qb.bit and bit for qb in zip(ind,collapsed_mask)] for ind in offspring]
            # # Calculating Fitness Of Individuals In Parallel
            # with contextlib.closing(Pool(processes=20)) as pool:
            #     fitnesses = pool.map_async(toolbox.evaluate, (ind for ind in masked_offspring))
            #     fitnesses = fitnesses.get()
                
            # for ind,fitness in zip(offspring,fitnesses):
            #     ind.fitness.values = fitness
            #     if not current_best_ind or current_best_ind.fitness.values[0] < ind.fitness.values[0] :
            #         current_best_ind = ind
            # # Updating And Evapurating Mask Values
            # mask_update(mask,best_ind,mask_evapuration_rate=.1,mask_update_rate=.5)
            # print('First Evaluation Time : {}'.format(time.time() - start))
            # if current_best_ind.fitness.values[0] > best_ind.fitness.values[0] :
            #     best_ind = toolbox.clone(current_best_ind)
            #     best_same_iter = 0
            # if best_same_iter < m_max :
            #     for ind in offspring :
            #         toolbox.rotate(ind,best_ind,best_ind.fitness.values[0] > ind.fitness.values[0])
            # else :
            #     offspring[:] = toolbox.catastrophe(best_ind)
            #     best_same_iter = 0
                
        pop[:] = offspring

        # Evaluate Individual 
        start = time.time()
        current_best_ind = None
        # Callapse Mask
        collapsed_mask = mask_collapse(mask,epsilon)
        # Collapsing Individual Bits
        for i,ind in enumerate(pop):
            toolbox.toBit(ind)
        # Masking Individual Bit Values
        masked_pop = [[qb.bit and bit for qb in zip(ind,collapsed_mask)] for ind in pop]

        with contextlib.closing(Pool(processes=20)) as pool:
            fitnesses = pool.map_async(toolbox.evaluate, (ind for ind in masked_pop))
            fitnesses = fitnesses.get()
        print(fitnesses)
        for ind,fitness in zip(pop,fitnesses):
            ind.fitness.values = fitness
            if not current_best_ind or current_best_ind.fitness.values[0] < ind.fitness.values[0] :
                current_best_ind = ind
        # Updating And Evapurating Mask Values
        mask_update(mask,best_ind,mask_evapuration_rate=.1,mask_update_rate=.5)
        print('Second Evaluation Time : {}'.format(time.time() - start))    
        best_fits = np.append(best_fits,current_best_ind.fitness.values[0])
        if current_best_ind.fitness.values[0] > best_ind.fitness.values[0] :
            best_ind = toolbox.clone(current_best_ind)
            best_same_iter = 0
        else :
            best_same_iter += 1
    return best_fits,best_ind