# -*- coding: utf-8 -*-
"""
Created on Thu Apr 21 10:46:15 2022

@author: erfan
"""
from pdp_utils import load_problem, feasibility_check, cost_function
from auxiliary_functions import *
from operators import one_ins_best, one_ins_first_better, multi_ins_new, multi_ins_rand, v_swap, one_ins_first_better_worst_remove,one_ins_best_worst_remove,multi_ins_rand_worst_remove
from auxiliary_functions import copy_costs, copy_features
import numpy as np
import pandas as pd
from time import time


def update_weights(weights, thetas, scores, r = 0.2):
    new_weights = []
    for w, weight in enumerate(weights):
        if thetas[w] > 0:
            new_weights.append((weight*(1-r)+r*(scores[w]/thetas[w])))
        else:
            new_weights.append(weight)
    return new_weights

def normalize_weights(weights):
    return weights/np.sum(weights)

def ALNS(init_sol, init_cost, probability, operators, prob, rng, call_costs, T_f = 0.1, warm_up = 100):
    incumbent = init_sol
    best_sol = init_sol
    cost_incumb = init_cost
    r = 0.2
    costs = [0 for i in range(prob['n_vehicles'])]+ [init_cost]
    LoadSize = [[] for i in range(prob['n_vehicles'])]
    Timewindows = [[[],[]] for i in range(prob['n_vehicles'])]
    PortIndex = [[] for i in range(prob['n_vehicles'])]
    LU_Time = [[] for i in range(prob['n_vehicles'])]
    features = [LoadSize, Timewindows, PortIndex, LU_Time]
    scores = [0 for i in range(len(operators))]
    thetas = [0 for i in range(len(operators))]
    last_improvement = 0
    best_cost = cost_incumb 
    delta = [0]
    non_imp_count = 0
    weights = probability
    sols = set([str(init_sol)[1:-1]])
    best_call_costs = copy_call_costs(call_costs)
    for itr in range(10000):
        # print(itr)
        if itr == warm_up and np.mean(delta) == 0:
            warm_up += 100
        if itr < warm_up:
            op_id = rng.choice(range(len(operators)), replace=True, p=probability )
            operator = operators[op_id]
            thetas[op_id] += 1
            new_sol, new_costs, new_features, new_call_costs = operator(incumbent, copy_costs(costs), copy_features(features), call_costs, rng, prob)
            new_sol_str = str(new_sol)[1:-1]
            if new_sol_str not in sols:
                sols.add(new_sol_str)
                scores[op_id] += 1
            new_cost = sum(new_costs)
            delta_E = new_cost - cost_incumb
            if delta_E >= 0 :
                non_imp_count+=1
            feasiblity, c = True, 'Feasible'
            if feasiblity and delta_E < 0:
                scores[op_id] += 1
                incumbent = new_sol
                cost_incumb = new_cost
                costs = copy_costs(new_costs)
                features = copy_features(new_features)
                call_costs = copy_call_costs(new_call_costs)
                if cost_incumb < best_cost:
                    scores[op_id] += 2
                    best_sol = incumbent
                    best_cost = cost_incumb
                    last_improvement = itr
                    non_imp_count = 0
                    best_call_costs = copy_call_costs(call_costs)
                    
            elif feasiblity:
                if rng.uniform() < 0.8:
                    incumbent = new_sol
                    cost_incumb = new_cost
                    costs = copy_costs(new_costs)
                    features = copy_features(new_features)
                    call_costs = copy_call_costs(new_call_costs)
                if delta_E>0:
                    delta.append(delta_E)
        else: 
            if itr == warm_up:
                delta_avg = np.mean(delta[1:])
                T_0 = -delta_avg / np.log(0.8)
                alpha = 0.9995#np.power((T_f/T_0), (1/(10000-warm_up)))
                T = T_0
                Ts = [T]
                Ps = [np.exp(-delta_avg/T)]
            if non_imp_count > 300:
                for i in range(1,21):
                    escape = operators[-1]
                    incumbent, costs, features, call_costs = escape(incumbent, copy_costs(costs), copy_features(features), copy_call_costs(call_costs), rng, prob) 
                    new_cost = sum(costs)
                    if new_cost < best_cost:
                        best_call_costs = copy_call_costs(call_costs)
                        best_sol = incumbent
                        best_cost = new_cost
                        last_improvement = itr + i
                        
                        break
                itr += i
                non_imp_count = 0
            if (itr - warm_up) % 100 == 0:
                weights = update_weights(weights, thetas, scores, r )
                
                scores = [0 for i in range(len(operators))]
                thetas = [0 for i in range(len(operators))]
                weights = normalize_weights(weights)
                # print(itr, weights)
        
            op_id = rng.choice(range(len(operators)), replace=True, p=weights )
            operator = operators[op_id]
            thetas[op_id] += 1
            new_sol, new_costs, new_features, new_call_costs = operator(incumbent, copy_costs(costs), copy_features(features), copy_call_costs(call_costs), rng, prob)
            new_sol_str = str(new_sol)[1:-1]
            if new_sol_str not in sols:
                sols.add(new_sol_str)
                scores[op_id] += 1
            new_cost = sum(new_costs)
            delta_E = new_cost - cost_incumb
            if delta_E >= 0 :
                non_imp_count+=1
            feasiblity, c = True, 'Feasible'
            
            if feasiblity and delta_E < 0:
                scores[op_id] += 1
                incumbent = new_sol
                cost_incumb = new_cost
                costs = copy_costs(new_costs)
                features = copy_features(new_features)
                call_costs = copy_call_costs(new_call_costs)
                if cost_incumb < best_cost:
                    scores[op_id] += 2
                    best_sol = incumbent
                    best_cost = cost_incumb
                    last_improvement = itr
                    best_call_costs = copy_call_costs(call_costs)
                    
            elif feasiblity:
                prbb = np.exp(-delta_E/T)
                if rng.uniform() < prbb:
                    incumbent = new_sol
                    cost_incumb = new_cost
                    costs = copy_costs(new_costs)
                    features = copy_features(new_features)
                    call_costs = copy_call_costs(new_call_costs)
                
            T *= alpha
    return best_sol, best_cost, last_improvement, sols

if __name__ == '__main__':
    problems = [
                # 'Call_7_Vehicle_3',
                # 'Call_18_Vehicle_5',
                'Call_35_Vehicle_7',
                # 'Call_80_Vehicle_20',
                # 'Call_130_Vehicle_40',
                # 'Call_300_Vehicle_90'
                ]
    operators = [
        # one_ins_best,
        # one_ins_first_better,
        # v_swap,
        # multi_ins_new,
        multi_ins_rand,
        one_ins_first_better_worst_remove,
        one_ins_best_worst_remove,
        multi_ins_rand_worst_remove
        ]
    probabilities = [
        [1/len(operators) for i in operators],
        ]
    
    repeat = 10
    for j, p in enumerate(problems):
        
        for prb in probabilities:
            start = time()
            prob = load_problem( "..//..//Data//" +p+ ".txt")
            initial_sol = [0]*prob['n_vehicles'] + [i for i in range(1,prob['n_calls']+1) for j in range(2)]
            init_cost = prob['Cargo'][:,3].sum()
            best_sol, best_cost, last_improvement = [[] for i in range(repeat)], [0 for i in range(repeat)], [0 for i in range(repeat)]
            sols = [set() for i in range(repeat)]
            call_costs = [ [prob['Cargo'][i-1,3] for i in range(1, prob['n_calls']+1)], [ i for i in range(1, prob['n_calls']+1)]]
            srt = np.argsort(call_costs[0])[::-1]
            call_costs = np.array([[call_costs[0][i] for i in srt], [call_costs[1][i] for i in srt], np.argsort([call_costs[1][i] for i in srt])])
            for i in range(repeat ):
                rng = np.random.default_rng(31+i)
                best_sol[i], best_cost[i], last_improvement[i], sols[i] = ALNS(initial_sol, init_cost, prb, operators, prob, rng, call_costs)
            running_time = (time()-start)/repeat
            minidx = np.argmin(best_cost)
            print(p,'\t', str(prb), '\t', str(np.mean(best_cost)), '\t', str(best_cost[minidx]), '\t', 100*((init_cost-best_cost[minidx])/init_cost),
                  '\t', running_time)
            print('Solution: ', str(best_sol[minidx]))