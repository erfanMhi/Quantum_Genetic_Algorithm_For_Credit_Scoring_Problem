

def new_multiprocess_main(pop_size, iter_num, n_max, m_max,
                          max_feature_num, min_feature_num,
                          pm, pc, pmm, pcc, chrom_mask, mask_best_num=1,
                          mask_evapuration_rate=.1, mask_update_rate=.5,
                          epsilon=1, multiprocessing=False, workers=20):

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

        if best_same_iter >= m_max:
            offspring[:] = toolbox.catastrophe(best_ind)
            best_same_iter = 0

    return best_fits, best_ind
