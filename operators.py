# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 14:40:15 2022

@author: erfan
"""
from auxiliary_functions import *
import numpy as np
from itertools import combinations

def one_ins_first_better(sol, costs, features, call_costs, rng, prob):
    Solution = sol + [0]
    selected_call = rng.choice(np.arange(1,prob['n_calls']+1))
    call_locs = np.where(Solution==selected_call)[0]
    call_cost_loc = int(call_costs[2, selected_call -1])
    ZeroIndexBef = np.array(np.where(Solution == np.array(0))[0], dtype=int)
    len_v = [ZeroIndexBef[0]] + [j-i-1 for i, j in zip(ZeroIndexBef[:-1], ZeroIndexBef[1:])]
    cur_v = len(ZeroIndexBef[ZeroIndexBef<call_locs[0]])
    best_sol = Solution.copy()
    best_features = copy_features(features)
    
    if cur_v == prob['n_vehicles']:
        cur_cost = prob['Cargo'][selected_call-1,3]
    else:
        sol_2 = Solution[:call_locs[0]] + Solution[call_locs[0]+1:call_locs[1]] + Solution[call_locs[1]+1:]
        cur_cost = cur_cost4V(cur_v, Solution[ZeroIndexBef[cur_v]-len_v[cur_v]:ZeroIndexBef[cur_v]],
                           sol_2[ZeroIndexBef[cur_v]-len_v[cur_v]:ZeroIndexBef[cur_v]-2],
                           *(call_locs-(sum(len_v[:cur_v])+ len(len_v[:cur_v]))),features[2],prob)
     
        features = features_delete(features, cur_v, *(call_locs-(sum(len_v[:cur_v])+ len(len_v[:cur_v]))))   
    costs[cur_v] -= cur_cost
    costs[-1] += prob['Cargo'][selected_call-1,3]
    Solution = Solution[:call_locs[0]] + Solution[call_locs[0]+1:call_locs[1]] + Solution[call_locs[1]+1:]
    ZeroIndex = np.array(np.where(Solution == np.array(0))[0], dtype=int)
    avail_vehs = [i for i in range(prob['n_vehicles']) if prob['VesselCargo'][i,selected_call-1] == 1]+ [prob['n_vehicles']]
    rng.shuffle(avail_vehs)
    len_v = [ZeroIndex[0]] + [j-i-1 for i, j in zip(ZeroIndex[:-1], ZeroIndex[1:])]
    best_v = cur_v 
    best_cost = cur_cost.copy()
    for v in avail_vehs:
        if v == prob['n_vehicles']:
            if best_cost >= prob['Cargo'][selected_call-1,3]:
                best_sol = Solution[:-1] + [selected_call,selected_call] + Solution[-1:]
                best_call_costs = copy_call_costs(call_costs)
                best_call_costs[0, call_cost_loc] = prob['Cargo'][selected_call-1,3]
                best_call_costs, _ = update_call_costs(best_call_costs, selected_call, call_cost_loc)
                break
            continue
        for pos in range(len_v[v]+1):
            cost_v = costs[v]
            new_sol = Solution[:ZeroIndex[v]+ pos - len_v[v]] +[selected_call]+ Solution[ZeroIndex[v]+ pos - len_v[v]:]
            
            del_idx, new_cost, feasibility, c, new_features = \
                findBestPosForDel(selected_call, v, new_sol[ZeroIndex[v]-len_v[v]:ZeroIndex[v]+1],
                                  pos, copy_features(features), prob)
            if not c.startswith('F'):
                if int(c[-1]) == pos or int(c[-1]) == del_idx:
                    Solution = new_sol[:ZeroIndex[v]+ pos - len_v[v]] + new_sol[ZeroIndex[v]+ pos - len_v[v]+1:]
                    if c.startswith('T'):
                        break
                    else:
                        continue
                else:
                    Solution = new_sol[:ZeroIndex[v]+ pos - len_v[v]] + new_sol[ZeroIndex[v]+ pos - len_v[v]+1:]
                    continue
            
            elif new_cost - cost_v < best_cost:
                best_sol = new_sol[:ZeroIndex[v]-len_v[v]+ del_idx] +[selected_call]+ new_sol[ZeroIndex[v]-len_v[v]+ del_idx:]
                best_call_costs = copy_call_costs(call_costs)
                best_call_costs[0, call_cost_loc] = new_cost - cost_v
                best_call_costs, _ = update_call_costs(best_call_costs, selected_call, call_cost_loc)
                best_cost = new_cost - cost_v
                best_v = v
                best_features = new_features
                break
            else:
               Solution = new_sol[:ZeroIndex[v]+ pos - len_v[v]] + new_sol[ZeroIndex[v]+ pos - len_v[v]+1:]
        if best_v != prob['n_vehicles']:
            break
    if best_cost == cur_cost:
        best_call_costs = copy_call_costs(call_costs)
        costs[best_v] += best_cost
        costs[-1] -= prob['Cargo'][selected_call-1,3]
        return sol, costs, best_features, best_call_costs
    if best_v != prob['n_vehicles']:
        costs[best_v] += best_cost
        costs[-1] -= prob['Cargo'][selected_call-1,3]
    return best_sol [:-1], costs, best_features, best_call_costs

def one_ins_best(sol, costs, features, call_costs, rng, prob):
    Solution = sol + [0]
    selected_call = rng.choice(np.arange(1,prob['n_calls']+1))
    call_locs = np.where(Solution==selected_call)[0]
    call_cost_loc = int(call_costs[2, selected_call -1])
    ZeroIndexBef = np.array(np.where(Solution == np.array(0))[0], dtype=int)
    len_v = [ZeroIndexBef[0]] + [j-i-1 for i, j in zip(ZeroIndexBef[:-1], ZeroIndexBef[1:])]
    cur_v = len(ZeroIndexBef[ZeroIndexBef<call_locs[0]])
    
    best_sol = Solution.copy()
    best_features = copy_features(features)
    if cur_v != prob['n_vehicles']:
        sol_2 = Solution[:call_locs[0]] + Solution[call_locs[0]+1:call_locs[1]] + Solution[call_locs[1]+1:]
        cur_cost =  cur_cost4V(cur_v, Solution[ZeroIndexBef[cur_v]-len_v[cur_v]:ZeroIndexBef[cur_v]],
                              sol_2[ZeroIndexBef[cur_v]-len_v[cur_v]:ZeroIndexBef[cur_v]-2],
                              *(call_locs-(sum(len_v[:cur_v])+ len(len_v[:cur_v]))),features[2],prob) 
        
        features = features_delete(features, cur_v, *(call_locs-(sum(len_v[:cur_v])+ len(len_v[:cur_v]))))
        costs[cur_v] -= cur_cost
        costs[-1] += prob['Cargo'][selected_call-1,3]

    Solution = Solution[:call_locs[0]] + Solution[call_locs[0]+1:call_locs[1]] + Solution[call_locs[1]+1:] 
    ZeroIndex = np.array(np.where(Solution == np.array(0))[0], dtype=int)
    avail_vehs = [i for i in range(prob['n_vehicles']) if prob['VesselCargo'][i,selected_call-1] == 1]+ [prob['n_vehicles']]
    len_v = [ZeroIndex[0]] + [j-i-1 for i, j in zip(ZeroIndex[:-1], ZeroIndex[1:])]
    best_v = prob['n_vehicles']
    best_cost = np.inf
    for v in avail_vehs:
        if v == prob['n_vehicles']:
            if best_cost >= prob['Cargo'][selected_call-1,3]:
                best_sol = Solution[:-1] + [selected_call,selected_call] + Solution[-1:]
                best_call_costs = copy_call_costs(call_costs)
                best_call_costs[0, call_cost_loc] = prob['Cargo'][selected_call-1,3]
                best_call_costs, _ = update_call_costs(best_call_costs, selected_call, call_cost_loc)
            continue
        for pos in range(len_v[v]+1):
            cost_v = costs[v]
            new_sol = Solution[:ZeroIndex[v]+ pos - len_v[v]] +[selected_call]+ Solution[ZeroIndex[v]+ pos - len_v[v]:]
            
            del_idx, new_cost, feasibility, c, new_features = \
                findBestPosForDel(selected_call, v, new_sol[ZeroIndex[v]-len_v[v]:ZeroIndex[v]+1],
                                  pos, copy_features(features), prob)
            if not c.startswith('F'):
                if int(c[-1]) == pos or int(c[-1]) == del_idx:
                    Solution = new_sol[:ZeroIndex[v]+ pos - len_v[v]] + new_sol[ZeroIndex[v]+ pos - len_v[v]+1:]
                    # continue
                    if c.startswith('T'):
                        break
                    else:
                        continue
                else:
                    Solution = new_sol[:ZeroIndex[v]+ pos - len_v[v]] + new_sol[ZeroIndex[v]+ pos - len_v[v]+1:]
                    continue
                
            elif new_cost - cost_v <= best_cost:
                best_sol = new_sol[:ZeroIndex[v]-len_v[v]+ del_idx] +[selected_call]+ new_sol[ZeroIndex[v]-len_v[v]+ del_idx:]
                best_cost = new_cost - cost_v
                best_call_costs = copy_call_costs(call_costs)
                best_call_costs[0, call_cost_loc] = new_cost - cost_v
                best_call_costs, _ = update_call_costs(best_call_costs, selected_call, call_cost_loc)
                best_v = v
                best_features = new_features
            else:
               Solution = new_sol[:ZeroIndex[v]+ pos - len_v[v]] + new_sol[ZeroIndex[v]+ pos - len_v[v]+1:]

    
    if best_v != prob['n_vehicles']:
        costs[best_v] += best_cost
        costs[-1] -= prob['Cargo'][selected_call-1,3]
    return best_sol [:-1], costs, best_features, best_call_costs

def multi_ins_new(sol, costs, features, call_costs, rng, prob):
    Solution = sol + [0]
    rm_size = rng.choice(np.arange(2,5))
    selected_calls = rng.choice(np.arange(1,prob['n_calls']+1),
                                     size = rm_size,
                                     replace = False
                                     )
    call_cost_loc = call_costs[2, selected_calls -1].astype(int)
    sel_loc = np.array([np.where(Solution == selected_calls[i])[0] for i in range(rm_size)])
    ZeroIndexBef = np.array(np.where(Solution == np.array(0))[0], dtype=int)
    len_v = [ZeroIndexBef[0]] + [j-i-1 for i, j in zip(ZeroIndexBef[:-1], ZeroIndexBef[1:])]
    cur_v = []
    for c_id, call in enumerate(selected_calls):
        cur_v.append(len(ZeroIndexBef[ZeroIndexBef<sel_loc[c_id][0]]))
        if cur_v[c_id] == prob['n_vehicles']:
            Solution = Solution[:sel_loc[c_id][0]] + Solution[sel_loc[c_id][0]+1:sel_loc[c_id][1]] + Solution[sel_loc[c_id][1]+1:]
            ZeroIndexBef[cur_v[c_id]:] -= 2
            len_v[cur_v[c_id]] -= 2
            sel_loc[sel_loc>sel_loc[c_id][1]]-=1
            sel_loc[sel_loc>sel_loc[c_id][0]]-=1
            sel_loc[c_id][1] += 1
        else:
            sol_2 = Solution[:sel_loc[c_id][0]] + Solution[sel_loc[c_id][0]+1:sel_loc[c_id][1]] + Solution[sel_loc[c_id][1]+1:]
            best_cost = cur_cost4V(cur_v[c_id], Solution[ZeroIndexBef[cur_v[c_id]]-len_v[cur_v[c_id]]:ZeroIndexBef[cur_v[c_id]]],
                                  sol_2[ZeroIndexBef[cur_v[c_id]]-len_v[cur_v[c_id]]:ZeroIndexBef[cur_v[c_id]]-2],
                                  *(sel_loc[c_id]-(sum(len_v[:cur_v[c_id]])+ len(len_v[:cur_v[c_id]]))),features[2],prob)
            
            Solution = sol_2
            ZeroIndexBef[cur_v[c_id]:] -= 2
            len_v[cur_v[c_id]] -= 2
            sel_loc[sel_loc>sel_loc[c_id][1]]-=1
            sel_loc[sel_loc>sel_loc[c_id][0]]-=1
            sel_loc[c_id][1] += 1
            features = features_delete(features, cur_v[c_id], *(sel_loc[c_id]-(sum(len_v[:cur_v[c_id]])+ len(len_v[:cur_v[c_id]]))))
            costs[cur_v[c_id]] -= best_cost
            costs[-1] += prob['Cargo'][call-1,3]

    for k in range(rm_size):
        best_costs = [np.inf for i in range(len(selected_calls))]
        best_sols = [[] for i in range(len(selected_calls)) ]
        best_vs = [prob['n_vehicles'] for i in range(len(selected_calls))]
        best_features = [copy_features(features) for i in range(len(selected_calls))]
        best_call_costs = [copy_call_costs(call_costs) for i in range(len(selected_calls))]
        changes = [[] for i in range(len(selected_calls))]
        ZeroIndex = np.array(np.where(Solution == np.array(0))[0], dtype=int)
        len_v = [ZeroIndex[0]] + [j-i-1 for i, j in zip(ZeroIndex[:-1], ZeroIndex[1:])]
        for s, selected_call in enumerate(selected_calls):
            avail_vehs = [i for i in range(prob['n_vehicles']) if prob['VesselCargo'][i,selected_call-1] == 1]+ [prob['n_vehicles']]
            for v in avail_vehs:
                if v == prob['n_vehicles']:
                    if best_costs[s] >= prob['Cargo'][selected_call-1,3]:
                        best_sols[s] = Solution[:-1] + [selected_call,selected_call] + Solution[-1:]
                        best_vs[s] = v
                        best_costs[s] = prob['Cargo'][selected_call-1,3]
                        best_call_costs[s] = copy_call_costs(call_costs)
                        best_call_costs[s][0, call_cost_loc[s]] = prob['Cargo'][selected_call-1,3]
                        best_call_costs[s], changes[s] = update_call_costs(best_call_costs[s], selected_call, call_cost_loc, s)
                        best_features[s] = copy_features(features)
                    continue
                for pos in range(len_v[v]+1):
                    cost_v = costs[v]
                    new_sol = Solution[:ZeroIndex[v]+ pos - len_v[v]] +[selected_call]+ Solution[ZeroIndex[v]+ pos - len_v[v]:]
                    del_idx, new_cost, feasibility, c, new_features = \
                        findBestPosForDel(selected_call, v, 
                                          new_sol[ZeroIndex[v]-len_v[v]:ZeroIndex[v]+1],
                                          pos, copy_features(features), prob)
                    if not c.startswith('F'):
                        if int(c[-1]) == pos or int(c[-1]) == del_idx:
                            Solution = new_sol[:ZeroIndex[v]+ pos - len_v[v]] + new_sol[ZeroIndex[v]+ pos - len_v[v]+1:]
                            # continue
                            if c.startswith('T'):
                                break
                            else:
                                continue
                        else:
                            Solution = new_sol[:ZeroIndex[v]+ pos - len_v[v]] + new_sol[ZeroIndex[v]+ pos - len_v[v]+1:]
                            continue
                    elif new_cost - cost_v < best_costs[s]:
                        best_sols[s] = new_sol[:ZeroIndex[v]-len_v[v]+ del_idx] +[selected_call]+ new_sol[ZeroIndex[v]-len_v[v]+ del_idx:]
                        best_costs[s] = new_cost - cost_v
                        best_vs[s] = v
                        best_features[s] = new_features
                        best_call_costs[s] = copy_call_costs(call_costs)
                        best_call_costs[s][0, call_cost_loc[s]] = new_cost - cost_v
                        best_call_costs[s], changes[s] = update_call_costs(best_call_costs[s], selected_call, call_cost_loc, s)
                    else:
                       Solution = new_sol[:ZeroIndex[v]+ pos - len_v[v]] + new_sol[ZeroIndex[v]+ pos - len_v[v]+1:]
        best_idx = np.argmin(best_costs)
        call_cost_loc = list(map(sum, zip(call_cost_loc, changes[best_idx])))
        call_costs = copy_call_costs(best_call_costs[best_idx])
        Solution = best_sols[best_idx]
        features = copy_features(best_features[best_idx])
        if best_vs[best_idx] != prob['n_vehicles']:
            costs[best_vs[best_idx]] += best_costs[best_idx]
            costs[-1] -= prob['Cargo'][selected_calls[best_idx]-1,3]
        selected_calls = np.delete(selected_calls, best_idx)
        call_cost_loc = call_cost_loc[:best_idx] + call_cost_loc[best_idx+1:]
    return Solution [:-1], costs, features, call_costs

def multi_ins_rand(sol, costs, features, call_costs, rng, prob):
    Solution = sol + [0]
    rm_size = rng.choice(np.arange(2,5))
    selected_calls = rng.choice(np.arange(1,prob['n_calls']+1),
                                     size = rm_size,
                                     replace = False)
    call_cost_loc = call_costs[2, selected_calls -1].astype(int)
    sel_loc = np.array([np.where(Solution == selected_calls[i])[0] for i in range(rm_size)])
    ZeroIndexBef = np.array(np.where(Solution == np.array(0))[0], dtype=int)
    len_v = [ZeroIndexBef[0]] + [j-i-1 for i, j in zip(ZeroIndexBef[:-1], ZeroIndexBef[1:])]
    cur_v = []
    for c_id, call in enumerate(selected_calls):
        cur_v.append(len(ZeroIndexBef[ZeroIndexBef<sel_loc[c_id][0]]))
        if cur_v[c_id] == prob['n_vehicles']:
            Solution = Solution[:sel_loc[c_id][0]] + Solution[sel_loc[c_id][0]+1:sel_loc[c_id][1]] + Solution[sel_loc[c_id][1]+1:]
            ZeroIndexBef[cur_v[c_id]:] -= 2
            len_v[cur_v[c_id]] -= 2
            sel_loc[sel_loc>sel_loc[c_id][1]]-=1
            sel_loc[sel_loc>sel_loc[c_id][0]]-=1
            sel_loc[c_id][1] += 1
        else:
            sol_2 = Solution[:sel_loc[c_id][0]] + Solution[sel_loc[c_id][0]+1:sel_loc[c_id][1]] + Solution[sel_loc[c_id][1]+1:]
            best_cost = cur_cost4V(cur_v[c_id], Solution[ZeroIndexBef[cur_v[c_id]]-len_v[cur_v[c_id]]:ZeroIndexBef[cur_v[c_id]]],
                                  sol_2[ZeroIndexBef[cur_v[c_id]]-len_v[cur_v[c_id]]:ZeroIndexBef[cur_v[c_id]]-2],
                                  *(sel_loc[c_id]-(sum(len_v[:cur_v[c_id]])+ len(len_v[:cur_v[c_id]]))),features[2],prob)
            
            Solution = sol_2
            ZeroIndexBef[cur_v[c_id]:] -= 2
            len_v[cur_v[c_id]] -= 2
            sel_loc[sel_loc>sel_loc[c_id][1]]-=1
            sel_loc[sel_loc>sel_loc[c_id][0]]-=1
            sel_loc[c_id][1] += 1
            features = features_delete(features, cur_v[c_id], *(sel_loc[c_id]-(sum(len_v[:cur_v[c_id]])+ len(len_v[:cur_v[c_id]]))))
            costs[cur_v[c_id]] -= best_cost
            costs[-1] += prob['Cargo'][call-1,3]
    temp = list(zip(selected_calls,call_cost_loc))
    rng.shuffle(temp)
    selected_calls,call_cost_loc = zip(*temp)
    call_cost_loc = list(call_cost_loc)
    for s, selected_call in enumerate(selected_calls):
        avail_vehs = [i for i in range(prob['n_vehicles']) if prob['VesselCargo'][i,selected_call-1] == 1]+ [prob['n_vehicles']]
        best_features = copy_features(features)
        
        best_v = prob['n_vehicles']
        best_cost = np.inf
        for v in avail_vehs:
            if v == prob['n_vehicles']:
                if best_cost >= prob['Cargo'][selected_call-1,3]:
                    best_sol = Solution[:-1] + [selected_call,selected_call] + Solution[-1:]
                    best_v = v
                    best_call_costs = copy_call_costs(call_costs)
                    best_cost = prob['Cargo'][selected_call-1,3]
                    best_call_costs[0, call_cost_loc[s]] = prob['Cargo'][selected_call-1,3]
                    best_call_costs, changes = update_call_costs(best_call_costs, selected_call, call_cost_loc, s)
                    best_features = copy_features(features)
                continue
            for pos in range(len_v[v]+1):
                cost_v = costs[v]
                new_sol = Solution[:ZeroIndexBef[v]+ pos - len_v[v]] +[selected_call]+ Solution[ZeroIndexBef[v]+ pos - len_v[v]:]
                
                del_idx, new_cost, feasibility, c, new_features = \
                    findBestPosForDel(selected_call, v, new_sol[ZeroIndexBef[v]-len_v[v]:ZeroIndexBef[v]+1],
                                      pos, copy_features(features), prob)
                if not c.startswith('F'):
                    if int(c[-1]) == pos or int(c[-1]) == del_idx:
                        Solution = new_sol[:ZeroIndexBef[v]+ pos - len_v[v]] + new_sol[ZeroIndexBef[v]+ pos - len_v[v]+1:]
                        if c.startswith('T'):
                            break
                        else:
                            continue
                    else:
                        Solution = new_sol[:ZeroIndexBef[v]+ pos - len_v[v]] + new_sol[ZeroIndexBef[v]+ pos - len_v[v]+1:]
                        continue
                    
                elif new_cost - cost_v <= best_cost:
                    best_sol = new_sol[:ZeroIndexBef[v]-len_v[v]+ del_idx] +[selected_call]+ new_sol[ZeroIndexBef[v]-len_v[v]+ del_idx:]
                    best_cost = new_cost - cost_v
                    best_v = v
                    best_call_costs = copy_call_costs(call_costs)
                    best_call_costs[0, call_cost_loc[s]] = new_cost - cost_v
                    best_call_costs, changes = update_call_costs(best_call_costs, selected_call, call_cost_loc, s)
                    best_features = new_features
                else:
                   Solution = new_sol[:ZeroIndexBef[v]+ pos - len_v[v]] + new_sol[ZeroIndexBef[v]+ pos - len_v[v]+1:]
        call_cost_loc = list(map(sum, zip(call_cost_loc, changes)))      
        Solution = best_sol
        features = copy_features(best_features)
        call_costs = copy_call_costs(best_call_costs)
        ZeroIndexBef[best_v:] += 2
        len_v[best_v] += 2
        if best_v != prob['n_vehicles']:
            costs[best_v] += best_cost
            costs[-1] -= prob['Cargo'][selected_call-1,3]
    
    return Solution [:-1], costs, features, best_call_costs

def v_swap(sol, costs, features, call_costs, rng, prob):
    Solution = sol + [0]
    ZeroIndex = np.array(np.where(Solution == np.array(0))[0], dtype=int)
    len_v = [ZeroIndex[0]] + [j-i-1 for i, j in zip(ZeroIndex[:-1], ZeroIndex[1:])]
    combs = list(combinations(range(prob['n_vehicles']),2))
    rng.shuffle(combs)
    possible = False
    for V1, V2 in combs:
        if V1<V2:
            v1 = V1
            v2 = V2
        else:
            v1 = V2
            v2 = V1
        v1_calls = np.unique(Solution[ZeroIndex[v1]-len_v[v1]:ZeroIndex[v1]])
        v2_calls = np.unique(Solution[ZeroIndex[v2]-len_v[v2]:ZeroIndex[v2]])
        if len(v1_calls) > 0 and len(v2_calls) >0:
            if (prob['VesselCargo'][v1,v1_calls-1] == prob['VesselCargo'][v2,v1_calls-1]).all() and \
                (prob['VesselCargo'][v1,v2_calls-1] == prob['VesselCargo'][v2,v2_calls-1]).all():
                new_sol = Solution[:ZeroIndex[v1]-len_v[v1]] + Solution[ZeroIndex[v2]-len_v[v2]:ZeroIndex[v2]] +Solution[ZeroIndex[v1]:ZeroIndex[v2]-len_v[v2]] +\
                    Solution[ZeroIndex[v1]-len_v[v1]:ZeroIndex[v1]] + Solution[ZeroIndex[v2]:]
                new_features, new_costs = swap_features_costs(copy_features(features), copy_costs(costs), v1, v2, sol[ZeroIndex[v1]-len_v[v1]:ZeroIndex[v1]],
                                         sol[ZeroIndex[v2]-len_v[v2]:ZeroIndex[v2]], prob)
                possible = v_feas_check(new_features, v1, prob) and v_feas_check(new_features, v2, prob)

                if possible:
                    return new_sol[:-1], new_costs, new_features
        elif len(v1_calls) > 0:
            if (prob['VesselCargo'][v1,v1_calls-1] == prob['VesselCargo'][v2,v1_calls-1]).all():
                new_sol = Solution[:ZeroIndex[v1]-len_v[v1]] + Solution[ZeroIndex[v2]-len_v[v2]:ZeroIndex[v2]] +Solution[ZeroIndex[v1]:ZeroIndex[v2]-len_v[v2]] +\
                    Solution[ZeroIndex[v1]-len_v[v1]:ZeroIndex[v1]] + Solution[ZeroIndex[v2]:]
                new_features, new_costs = swap_features_costs(copy_features(features), copy_costs(costs), v1, v2, sol[ZeroIndex[v1]-len_v[v1]:ZeroIndex[v1]],
                                         sol[ZeroIndex[v2]-len_v[v2]:ZeroIndex[v2]], prob)
                possible = v_feas_check(new_features, v2, prob)
                if possible:
                    return new_sol[:-1], new_costs, new_features
        elif len(v2_calls) > 0:
            if (prob['VesselCargo'][v1,v2_calls-1] == prob['VesselCargo'][v2,v2_calls-1]).all():
                new_sol = Solution[:ZeroIndex[v1]-len_v[v1]] + Solution[ZeroIndex[v2]-len_v[v2]:ZeroIndex[v2]] +Solution[ZeroIndex[v1]:ZeroIndex[v2]-len_v[v2]] +\
                    Solution[ZeroIndex[v1]-len_v[v1]:ZeroIndex[v1]] + Solution[ZeroIndex[v2]:]
                new_features, new_costs = swap_features_costs(copy_features(features), copy_costs(costs), v1, v2, sol[ZeroIndex[v1]-len_v[v1]:ZeroIndex[v1]],
                                         sol[ZeroIndex[v2]-len_v[v2]:ZeroIndex[v2]], prob)
                possible = v_feas_check(new_features, v1, prob)
                if possible:
                    return new_sol[:-1], new_costs, new_features
      
    return Solution [:-1], costs, features, best_call_costs

def one_ins_first_better_worst_remove(sol, costs, features, call_costs, rng, prob):
    Solution = sol + [0]
    selected_call = worstRemoval(call_costs, 1, rng)[0]#rng.choice(np.arange(1,prob['n_calls']+1))
    call_locs = np.where(Solution==np.int32(selected_call))[0]
    call_cost_loc = int(call_costs[2, selected_call -1])
    ZeroIndexBef = np.array(np.where(Solution == np.array(0))[0], dtype=int)
    len_v = [ZeroIndexBef[0]] + [j-i-1 for i, j in zip(ZeroIndexBef[:-1], ZeroIndexBef[1:])]
    cur_v = len(ZeroIndexBef[ZeroIndexBef<call_locs[0]])
    best_sol = Solution.copy()
    best_features = copy_features(features)
    
    if cur_v == prob['n_vehicles']:
        cur_cost = prob['Cargo'][selected_call-1,3]
    else:
        sol_2 = Solution[:call_locs[0]] + Solution[call_locs[0]+1:call_locs[1]] + Solution[call_locs[1]+1:]
        cur_cost = cur_cost4V(cur_v, Solution[ZeroIndexBef[cur_v]-len_v[cur_v]:ZeroIndexBef[cur_v]],
                           sol_2[ZeroIndexBef[cur_v]-len_v[cur_v]:ZeroIndexBef[cur_v]-2],
                           *(call_locs-(sum(len_v[:cur_v])+ len(len_v[:cur_v]))),features[2],prob)
     
        features = features_delete(features, cur_v, *(call_locs-(sum(len_v[:cur_v])+ len(len_v[:cur_v]))))   
    costs[cur_v] -= cur_cost
    costs[-1] += prob['Cargo'][selected_call-1,3]
    Solution = Solution[:call_locs[0]] + Solution[call_locs[0]+1:call_locs[1]] + Solution[call_locs[1]+1:]
    ZeroIndex = np.array(np.where(Solution == np.array(0))[0], dtype=int)
    avail_vehs = [i for i in range(prob['n_vehicles']) if prob['VesselCargo'][i,selected_call-1] == 1]+ [prob['n_vehicles']]
    rng.shuffle(avail_vehs)
    len_v = [ZeroIndex[0]] + [j-i-1 for i, j in zip(ZeroIndex[:-1], ZeroIndex[1:])]
    best_v = cur_v 
    best_cost = cur_cost.copy()
    for v in avail_vehs:
        if v == prob['n_vehicles']:
            if best_cost >= prob['Cargo'][selected_call-1,3]:
                best_sol = Solution[:-1] + [selected_call,selected_call] + Solution[-1:]
                best_call_costs = copy_call_costs(call_costs)
                best_call_costs[0, call_cost_loc] = prob['Cargo'][selected_call-1,3]
                best_call_costs, _ = update_call_costs(best_call_costs, selected_call, call_cost_loc)
                break
            continue
        for pos in range(len_v[v]+1):
            cost_v = costs[v]
            new_sol = Solution[:ZeroIndex[v]+ pos - len_v[v]] +[selected_call]+ Solution[ZeroIndex[v]+ pos - len_v[v]:]
            
            del_idx, new_cost, feasibility, c, new_features = \
                findBestPosForDel(selected_call, v, new_sol[ZeroIndex[v]-len_v[v]:ZeroIndex[v]+1],
                                  pos, copy_features(features), prob)
            if not c.startswith('F'):
                if int(c[-1]) == pos or int(c[-1]) == del_idx:
                    Solution = new_sol[:ZeroIndex[v]+ pos - len_v[v]] + new_sol[ZeroIndex[v]+ pos - len_v[v]+1:]
                    if c.startswith('T'):
                        break
                    else:
                        continue
                else:
                    Solution = new_sol[:ZeroIndex[v]+ pos - len_v[v]] + new_sol[ZeroIndex[v]+ pos - len_v[v]+1:]
                    continue
            
            elif new_cost - cost_v < best_cost:
                best_sol = new_sol[:ZeroIndex[v]-len_v[v]+ del_idx] +[selected_call]+ new_sol[ZeroIndex[v]-len_v[v]+ del_idx:]
                best_call_costs = copy_call_costs(call_costs)
                best_call_costs[0, call_cost_loc] = new_cost - cost_v
                best_call_costs, _ = update_call_costs(best_call_costs, selected_call, call_cost_loc)
                best_cost = new_cost - cost_v
                best_v = v
                best_features = new_features
                break
            else:
               Solution = new_sol[:ZeroIndex[v]+ pos - len_v[v]] + new_sol[ZeroIndex[v]+ pos - len_v[v]+1:]
        if best_v != prob['n_vehicles']:
            break
    if best_cost == cur_cost:
        best_call_costs = copy_call_costs(call_costs)
        costs[best_v] += best_cost
        costs[-1] -= prob['Cargo'][selected_call-1,3]
        return sol, costs, best_features, best_call_costs
    if best_v != prob['n_vehicles']:
        costs[best_v] += best_cost
        costs[-1] -= prob['Cargo'][selected_call-1,3]
    return best_sol [:-1], costs, best_features, best_call_costs

def one_ins_best_worst_remove(sol, costs, features, call_costs, rng, prob):
    Solution = sol + [0]
    selected_call = worstRemoval(call_costs, 1, rng)[0]#rng.choice(np.arange(1,prob['n_calls']+1))
    call_locs = np.where(Solution==np.int32(selected_call))[0]
    call_cost_loc = int(call_costs[2, selected_call -1])
    ZeroIndexBef = np.array(np.where(Solution == np.array(0))[0], dtype=int)
    len_v = [ZeroIndexBef[0]] + [j-i-1 for i, j in zip(ZeroIndexBef[:-1], ZeroIndexBef[1:])]
    cur_v = len(ZeroIndexBef[ZeroIndexBef<call_locs[0]])
    
    best_sol = Solution.copy()
    best_features = copy_features(features)
    if cur_v != prob['n_vehicles']:
        sol_2 = Solution[:call_locs[0]] + Solution[call_locs[0]+1:call_locs[1]] + Solution[call_locs[1]+1:]
        cur_cost =  cur_cost4V(cur_v, Solution[ZeroIndexBef[cur_v]-len_v[cur_v]:ZeroIndexBef[cur_v]],
                              sol_2[ZeroIndexBef[cur_v]-len_v[cur_v]:ZeroIndexBef[cur_v]-2],
                              *(call_locs-(sum(len_v[:cur_v])+ len(len_v[:cur_v]))),features[2],prob) 
        
        features = features_delete(features, cur_v, *(call_locs-(sum(len_v[:cur_v])+ len(len_v[:cur_v]))))
        costs[cur_v] -= cur_cost
        costs[-1] += prob['Cargo'][selected_call-1,3]

    Solution = Solution[:call_locs[0]] + Solution[call_locs[0]+1:call_locs[1]] + Solution[call_locs[1]+1:] 
    ZeroIndex = np.array(np.where(Solution == np.array(0))[0], dtype=int)
    avail_vehs = [i for i in range(prob['n_vehicles']) if prob['VesselCargo'][i,selected_call-1] == 1]+ [prob['n_vehicles']]
    len_v = [ZeroIndex[0]] + [j-i-1 for i, j in zip(ZeroIndex[:-1], ZeroIndex[1:])]
    best_v = prob['n_vehicles']
    best_cost = np.inf
    for v in avail_vehs:
        if v == prob['n_vehicles']:
            if best_cost >= prob['Cargo'][selected_call-1,3]:
                best_sol = Solution[:-1] + [selected_call,selected_call] + Solution[-1:]
                best_call_costs = copy_call_costs(call_costs)
                best_call_costs[0, call_cost_loc] = prob['Cargo'][selected_call-1,3]
                best_call_costs, _ = update_call_costs(best_call_costs, selected_call, call_cost_loc)
            continue
        for pos in range(len_v[v]+1):
            cost_v = costs[v]
            new_sol = Solution[:ZeroIndex[v]+ pos - len_v[v]] +[selected_call]+ Solution[ZeroIndex[v]+ pos - len_v[v]:]
            
            del_idx, new_cost, feasibility, c, new_features = \
                findBestPosForDel(selected_call, v, new_sol[ZeroIndex[v]-len_v[v]:ZeroIndex[v]+1],
                                  pos, copy_features(features), prob)
            if not c.startswith('F'):
                if int(c[-1]) == pos or int(c[-1]) == del_idx:
                    Solution = new_sol[:ZeroIndex[v]+ pos - len_v[v]] + new_sol[ZeroIndex[v]+ pos - len_v[v]+1:]
                    # continue
                    if c.startswith('T'):
                        break
                    else:
                        continue
                else:
                    Solution = new_sol[:ZeroIndex[v]+ pos - len_v[v]] + new_sol[ZeroIndex[v]+ pos - len_v[v]+1:]
                    continue
                
            elif new_cost - cost_v <= best_cost:
                best_sol = new_sol[:ZeroIndex[v]-len_v[v]+ del_idx] +[selected_call]+ new_sol[ZeroIndex[v]-len_v[v]+ del_idx:]
                best_cost = new_cost - cost_v
                best_call_costs = copy_call_costs(call_costs)
                best_call_costs[0, call_cost_loc] = new_cost - cost_v
                best_call_costs, _ = update_call_costs(best_call_costs, selected_call, call_cost_loc)
                best_v = v
                best_features = new_features
            else:
               Solution = new_sol[:ZeroIndex[v]+ pos - len_v[v]] + new_sol[ZeroIndex[v]+ pos - len_v[v]+1:]

    
    if best_v != prob['n_vehicles']:
        costs[best_v] += best_cost
        costs[-1] -= prob['Cargo'][selected_call-1,3]
    return best_sol [:-1], costs, best_features, best_call_costs

def multi_ins_rand_worst_remove(sol, costs, features, call_costs, rng, prob):
    Solution = sol + [0]
    rm_size = rng.choice(np.arange(2,5))
    selected_calls = worstRemoval(call_costs, rm_size, rng)
    # rng.choice(np.arange(1,prob['n_calls']+1),
    #                                  size = rm_size,
    #                                  replace = False)
    call_cost_loc = [call_costs[2, selected_calls[i] -1].astype(int) for i in range(rm_size)]
    sel_loc = np.array([np.where(Solution == np.int32(selected_calls[i]))[0] for i in range(rm_size)])
    ZeroIndexBef = np.array(np.where(Solution == np.array(0))[0], dtype=int)
    len_v = [ZeroIndexBef[0]] + [j-i-1 for i, j in zip(ZeroIndexBef[:-1], ZeroIndexBef[1:])]
    cur_v = []
    for c_id, call in enumerate(selected_calls):
        cur_v.append(len(ZeroIndexBef[ZeroIndexBef<sel_loc[c_id][0]]))
        if cur_v[c_id] == prob['n_vehicles']:
            Solution = Solution[:sel_loc[c_id][0]] + Solution[sel_loc[c_id][0]+1:sel_loc[c_id][1]] + Solution[sel_loc[c_id][1]+1:]
            ZeroIndexBef[cur_v[c_id]:] -= 2
            len_v[cur_v[c_id]] -= 2
            sel_loc[sel_loc>sel_loc[c_id][1]]-=1
            sel_loc[sel_loc>sel_loc[c_id][0]]-=1
            sel_loc[c_id][1] += 1
        else:
            sol_2 = Solution[:sel_loc[c_id][0]] + Solution[sel_loc[c_id][0]+1:sel_loc[c_id][1]] + Solution[sel_loc[c_id][1]+1:]
            best_cost = cur_cost4V(cur_v[c_id], Solution[ZeroIndexBef[cur_v[c_id]]-len_v[cur_v[c_id]]:ZeroIndexBef[cur_v[c_id]]],
                                  sol_2[ZeroIndexBef[cur_v[c_id]]-len_v[cur_v[c_id]]:ZeroIndexBef[cur_v[c_id]]-2],
                                  *(sel_loc[c_id]-(sum(len_v[:cur_v[c_id]])+ len(len_v[:cur_v[c_id]]))),features[2],prob)
            
            Solution = sol_2
            ZeroIndexBef[cur_v[c_id]:] -= 2
            len_v[cur_v[c_id]] -= 2
            sel_loc[sel_loc>sel_loc[c_id][1]]-=1
            sel_loc[sel_loc>sel_loc[c_id][0]]-=1
            sel_loc[c_id][1] += 1
            features = features_delete(features, cur_v[c_id], *(sel_loc[c_id]-(sum(len_v[:cur_v[c_id]])+ len(len_v[:cur_v[c_id]]))))
            costs[cur_v[c_id]] -= best_cost
            costs[-1] += prob['Cargo'][call-1,3]
    temp = list(zip(selected_calls,call_cost_loc))
    rng.shuffle(temp)
    selected_calls,call_cost_loc = zip(*temp)
    call_cost_loc = list(call_cost_loc)
    for s, selected_call in enumerate(selected_calls):
        avail_vehs = [i for i in range(prob['n_vehicles']) if prob['VesselCargo'][i,selected_call-1] == 1]+ [prob['n_vehicles']]
        best_features = copy_features(features)
        
        best_v = prob['n_vehicles']
        best_cost = np.inf
        for v in avail_vehs:
            if v == prob['n_vehicles']:
                if best_cost >= prob['Cargo'][selected_call-1,3]:
                    best_sol = Solution[:-1] + [selected_call,selected_call] + Solution[-1:]
                    best_v = v
                    best_call_costs = copy_call_costs(call_costs)
                    best_cost = prob['Cargo'][selected_call-1,3]
                    best_call_costs[0, call_cost_loc[s]] = prob['Cargo'][selected_call-1,3]
                    best_call_costs, changes = update_call_costs(best_call_costs, selected_call, call_cost_loc, s)
                    best_features = copy_features(features)
                continue
            for pos in range(len_v[v]+1):
                cost_v = costs[v]
                new_sol = Solution[:ZeroIndexBef[v]+ pos - len_v[v]] +[selected_call]+ Solution[ZeroIndexBef[v]+ pos - len_v[v]:]
                
                del_idx, new_cost, feasibility, c, new_features = \
                    findBestPosForDel(selected_call, v, new_sol[ZeroIndexBef[v]-len_v[v]:ZeroIndexBef[v]+1],
                                      pos, copy_features(features), prob)
                if not c.startswith('F'):
                    if int(c[-1]) == pos or int(c[-1]) == del_idx:
                        Solution = new_sol[:ZeroIndexBef[v]+ pos - len_v[v]] + new_sol[ZeroIndexBef[v]+ pos - len_v[v]+1:]
                        if c.startswith('T'):
                            break
                        else:
                            continue
                    else:
                        Solution = new_sol[:ZeroIndexBef[v]+ pos - len_v[v]] + new_sol[ZeroIndexBef[v]+ pos - len_v[v]+1:]
                        continue
                    
                elif new_cost - cost_v <= best_cost:
                    best_sol = new_sol[:ZeroIndexBef[v]-len_v[v]+ del_idx] +[selected_call]+ new_sol[ZeroIndexBef[v]-len_v[v]+ del_idx:]
                    best_cost = new_cost - cost_v
                    best_v = v
                    best_call_costs = copy_call_costs(call_costs)
                    best_call_costs[0, call_cost_loc[s]] = new_cost - cost_v
                    best_call_costs, changes = update_call_costs(best_call_costs, selected_call, call_cost_loc, s)
                    best_features = new_features
                else:
                   Solution = new_sol[:ZeroIndexBef[v]+ pos - len_v[v]] + new_sol[ZeroIndexBef[v]+ pos - len_v[v]+1:]
        call_cost_loc = list(map(sum, zip(call_cost_loc, changes)))      
        Solution = best_sol
        features = copy_features(best_features)
        call_costs = copy_call_costs(best_call_costs)
        ZeroIndexBef[best_v:] += 2
        len_v[best_v] += 2
        if best_v != prob['n_vehicles']:
            costs[best_v] += best_cost
            costs[-1] -= prob['Cargo'][selected_call-1,3]
    
    return Solution [:-1], costs, features, best_call_costs

def multi_ins_new_worst_remove(sol, costs, features, call_costs, rng, prob):
    Solution = sol + [0]
    rm_size = rng.choice(np.arange(2,5))
    selected_calls = worstRemoval(call_costs, rm_size, rng)
    # rng.choice(np.arange(1,prob['n_calls']+1),
    #                                  size = rm_size,
    #                                  replace = False
    #                                  )
    call_cost_loc = [call_costs[2, selected_calls[i] -1].astype(int) for i in range(rm_size)]#call_costs[2, selected_calls -1].astype(int)
    sel_loc = np.array([np.where(Solution == np.int32(selected_calls[i]))[0] for i in range(rm_size)])
    ZeroIndexBef = np.array(np.where(Solution == np.array(0))[0], dtype=int)
    len_v = [ZeroIndexBef[0]] + [j-i-1 for i, j in zip(ZeroIndexBef[:-1], ZeroIndexBef[1:])]
    cur_v = []
    for c_id, call in enumerate(selected_calls):
        cur_v.append(len(ZeroIndexBef[ZeroIndexBef<sel_loc[c_id][0]]))
        if cur_v[c_id] == prob['n_vehicles']:
            Solution = Solution[:sel_loc[c_id][0]] + Solution[sel_loc[c_id][0]+1:sel_loc[c_id][1]] + Solution[sel_loc[c_id][1]+1:]
            ZeroIndexBef[cur_v[c_id]:] -= 2
            len_v[cur_v[c_id]] -= 2
            sel_loc[sel_loc>sel_loc[c_id][1]]-=1
            sel_loc[sel_loc>sel_loc[c_id][0]]-=1
            sel_loc[c_id][1] += 1
        else:
            sol_2 = Solution[:sel_loc[c_id][0]] + Solution[sel_loc[c_id][0]+1:sel_loc[c_id][1]] + Solution[sel_loc[c_id][1]+1:]
            best_cost = cur_cost4V(cur_v[c_id], Solution[ZeroIndexBef[cur_v[c_id]]-len_v[cur_v[c_id]]:ZeroIndexBef[cur_v[c_id]]],
                                  sol_2[ZeroIndexBef[cur_v[c_id]]-len_v[cur_v[c_id]]:ZeroIndexBef[cur_v[c_id]]-2],
                                  *(sel_loc[c_id]-(sum(len_v[:cur_v[c_id]])+ len(len_v[:cur_v[c_id]]))),features[2],prob)
            
            Solution = sol_2
            ZeroIndexBef[cur_v[c_id]:] -= 2
            len_v[cur_v[c_id]] -= 2
            sel_loc[sel_loc>sel_loc[c_id][1]]-=1
            sel_loc[sel_loc>sel_loc[c_id][0]]-=1
            sel_loc[c_id][1] += 1
            features = features_delete(features, cur_v[c_id], *(sel_loc[c_id]-(sum(len_v[:cur_v[c_id]])+ len(len_v[:cur_v[c_id]]))))
            costs[cur_v[c_id]] -= best_cost
            costs[-1] += prob['Cargo'][call-1,3]

    for k in range(rm_size):
        best_costs = [np.inf for i in range(len(selected_calls))]
        best_sols = [[] for i in range(len(selected_calls)) ]
        best_vs = [prob['n_vehicles'] for i in range(len(selected_calls))]
        best_features = [copy_features(features) for i in range(len(selected_calls))]
        best_call_costs = [copy_call_costs(call_costs) for i in range(len(selected_calls))]
        changes = [[] for i in range(len(selected_calls))]
        ZeroIndex = np.array(np.where(Solution == np.array(0))[0], dtype=int)
        len_v = [ZeroIndex[0]] + [j-i-1 for i, j in zip(ZeroIndex[:-1], ZeroIndex[1:])]
        for s, selected_call in enumerate(selected_calls):
            avail_vehs = [i for i in range(prob['n_vehicles']) if prob['VesselCargo'][i,selected_call-1] == 1]+ [prob['n_vehicles']]
            for v in avail_vehs:
                if v == prob['n_vehicles']:
                    if best_costs[s] >= prob['Cargo'][selected_call-1,3]:
                        best_sols[s] = Solution[:-1] + [selected_call,selected_call] + Solution[-1:]
                        best_vs[s] = v
                        best_costs[s] = prob['Cargo'][selected_call-1,3]
                        best_call_costs[s] = copy_call_costs(call_costs)
                        best_call_costs[s][0, call_cost_loc[s]] = prob['Cargo'][selected_call-1,3]
                        best_call_costs[s], changes[s] = update_call_costs(best_call_costs[s], selected_call, call_cost_loc, s)
                        best_features[s] = copy_features(features)
                    continue
                for pos in range(len_v[v]+1):
                    cost_v = costs[v]
                    new_sol = Solution[:ZeroIndex[v]+ pos - len_v[v]] +[selected_call]+ Solution[ZeroIndex[v]+ pos - len_v[v]:]
                    del_idx, new_cost, feasibility, c, new_features = \
                        findBestPosForDel(selected_call, v, 
                                          new_sol[ZeroIndex[v]-len_v[v]:ZeroIndex[v]+1],
                                          pos, copy_features(features), prob)
                    if not c.startswith('F'):
                        if int(c[-1]) == pos or int(c[-1]) == del_idx:
                            Solution = new_sol[:ZeroIndex[v]+ pos - len_v[v]] + new_sol[ZeroIndex[v]+ pos - len_v[v]+1:]
                            # continue
                            if c.startswith('T'):
                                break
                            else:
                                continue
                        else:
                            Solution = new_sol[:ZeroIndex[v]+ pos - len_v[v]] + new_sol[ZeroIndex[v]+ pos - len_v[v]+1:]
                            continue
                    elif new_cost - cost_v < best_costs[s]:
                        best_sols[s] = new_sol[:ZeroIndex[v]-len_v[v]+ del_idx] +[selected_call]+ new_sol[ZeroIndex[v]-len_v[v]+ del_idx:]
                        best_costs[s] = new_cost - cost_v
                        best_vs[s] = v
                        best_features[s] = new_features
                        best_call_costs[s] = copy_call_costs(call_costs)
                        best_call_costs[s][0, call_cost_loc[s]] = new_cost - cost_v
                        best_call_costs[s], changes[s] = update_call_costs(best_call_costs[s], selected_call, call_cost_loc, s)
                    else:
                       Solution = new_sol[:ZeroIndex[v]+ pos - len_v[v]] + new_sol[ZeroIndex[v]+ pos - len_v[v]+1:]
        best_idx = np.argmin(best_costs)
        call_cost_loc = list(map(sum, zip(call_cost_loc, changes[best_idx])))
        call_costs = copy_call_costs(best_call_costs[best_idx])
        Solution = best_sols[best_idx]
        features = copy_features(best_features[best_idx])
        if best_vs[best_idx] != prob['n_vehicles']:
            costs[best_vs[best_idx]] += best_costs[best_idx]
            costs[-1] -= prob['Cargo'][selected_calls[best_idx]-1,3]
        selected_calls = np.delete(selected_calls, best_idx)
        call_cost_loc = call_cost_loc[:best_idx] + call_cost_loc[best_idx+1:]
    return Solution [:-1], costs, features, call_costs

def one_ins_new_v_worst_remove(sol, costs, features, call_costs, rng, prob):
    Solution = sol + [0]
    ZeroIndexBef = np.array(np.where(Solution == np.array(0))[0], dtype=int)
    len_v = [ZeroIndexBef[0]] + [j-i-1 for i, j in zip(ZeroIndexBef[:-1], ZeroIndexBef[1:])]
    cur_v = rng.choice(np.arange(prob['n_vehicles']+1),p=len_v/np.sum(len_v))
    call_list = np.unique(Solution[ZeroIndexBef[cur_v]-len_v[cur_v]:ZeroIndexBef[cur_v]])
    call_costs_temp = call_costs[:,call_costs[2, call_list -1].astype(int)]
    selected_call = worstRemoval(call_costs_temp, 1, rng)[0]
    call_locs = np.where(Solution==np.int32(selected_call))[0]
    call_cost_loc = int(call_costs[2, selected_call -1])
    
    # cur_v = len(ZeroIndexBef[ZeroIndexBef<call_locs[0]])
    best_sol = Solution.copy()
    best_features = copy_features(features)
    
    if cur_v == prob['n_vehicles']:
        cur_cost = prob['Cargo'][selected_call-1,3]
    else:
        sol_2 = Solution[:call_locs[0]] + Solution[call_locs[0]+1:call_locs[1]] + Solution[call_locs[1]+1:]
        cur_cost = cur_cost4V(cur_v, Solution[ZeroIndexBef[cur_v]-len_v[cur_v]:ZeroIndexBef[cur_v]],
                           sol_2[ZeroIndexBef[cur_v]-len_v[cur_v]:ZeroIndexBef[cur_v]-2],
                           *(call_locs-(sum(len_v[:cur_v])+ len(len_v[:cur_v]))),features[2],prob)
     
        features = features_delete(features, cur_v, *(call_locs-(sum(len_v[:cur_v])+ len(len_v[:cur_v]))))   
    costs[cur_v] -= cur_cost
    costs[-1] += prob['Cargo'][selected_call-1,3]
    Solution = Solution[:call_locs[0]] + Solution[call_locs[0]+1:call_locs[1]] + Solution[call_locs[1]+1:]
    ZeroIndex = np.array(np.where(Solution == np.array(0))[0], dtype=int)
    avail_vehs = [i for i in range(prob['n_vehicles']) if prob['VesselCargo'][i,selected_call-1] == 1]+ [prob['n_vehicles']]
    rng.shuffle(avail_vehs)
    len_v = [ZeroIndex[0]] + [j-i-1 for i, j in zip(ZeroIndex[:-1], ZeroIndex[1:])]
    best_v = cur_v 
    best_cost = cur_cost.copy()
    for v in avail_vehs:
        if v == cur_v:
            continue
        if v == prob['n_vehicles']:
            if best_cost >= prob['Cargo'][selected_call-1,3]:
                best_sol = Solution[:-1] + [selected_call,selected_call] + Solution[-1:]
                best_call_costs = copy_call_costs(call_costs)
                best_call_costs[0, call_cost_loc] = prob['Cargo'][selected_call-1,3]
                best_call_costs, _ = update_call_costs(best_call_costs, selected_call, call_cost_loc)
                break
            continue
        for pos in range(len_v[v]+1):
            cost_v = costs[v]
            new_sol = Solution[:ZeroIndex[v]+ pos - len_v[v]] +[selected_call]+ Solution[ZeroIndex[v]+ pos - len_v[v]:]
            
            del_idx, new_cost, feasibility, c, new_features = \
                findBestPosForDel(selected_call, v, new_sol[ZeroIndex[v]-len_v[v]:ZeroIndex[v]+1],
                                  pos, copy_features(features), prob)
            if not c.startswith('F'):
                if int(c[-1]) == pos or int(c[-1]) == del_idx:
                    Solution = new_sol[:ZeroIndex[v]+ pos - len_v[v]] + new_sol[ZeroIndex[v]+ pos - len_v[v]+1:]
                    if c.startswith('T'):
                        break
                    else:
                        continue
                else:
                    Solution = new_sol[:ZeroIndex[v]+ pos - len_v[v]] + new_sol[ZeroIndex[v]+ pos - len_v[v]+1:]
                    continue
            
            elif new_cost - cost_v < best_cost:
                best_sol = new_sol[:ZeroIndex[v]-len_v[v]+ del_idx] +[selected_call]+ new_sol[ZeroIndex[v]-len_v[v]+ del_idx:]
                best_call_costs = copy_call_costs(call_costs)
                best_call_costs[0, call_cost_loc] = new_cost - cost_v
                best_call_costs, _ = update_call_costs(best_call_costs, selected_call, call_cost_loc)
                best_cost = new_cost - cost_v
                best_v = v
                best_features = new_features
                break
            else:
               Solution = new_sol[:ZeroIndex[v]+ pos - len_v[v]] + new_sol[ZeroIndex[v]+ pos - len_v[v]+1:]
        if best_v != prob['n_vehicles']:
            break
    if best_cost == cur_cost:
        best_call_costs = copy_call_costs(call_costs)
        costs[best_v] += best_cost
        costs[-1] -= prob['Cargo'][selected_call-1,3]
        return sol, costs, best_features, best_call_costs
    if best_v != prob['n_vehicles']:
        costs[best_v] += best_cost
        costs[-1] -= prob['Cargo'][selected_call-1,3]
    return best_sol [:-1], costs, best_features, best_call_costs

def multi_ins_rand_new_v_worst_remove(sol, costs, features, call_costs, rng, prob):
    Solution = sol + [0]
    rm_size = rng.choice(np.arange(2,5))
    ZeroIndexBef = np.array(np.where(Solution == np.array(0))[0], dtype=int)
    len_v = [ZeroIndexBef[0]] + [j-i-1 for i, j in zip(ZeroIndexBef[:-1], ZeroIndexBef[1:])]
    cur_v = rng.choice(np.arange(prob['n_vehicles']+1),p=len_v/np.sum(len_v), size = rm_size)
    call_list = []
    added = []
    for num in range(rm_size):
        if cur_v[num] not in added:
            call_list.extend(np.unique(Solution[ZeroIndexBef[cur_v[num]]-len_v[cur_v[num]]:ZeroIndexBef[cur_v[num]]]))
            added.append(cur_v[num])
    call_costs_temp = call_costs[:,call_costs[2, np.array(call_list) -1].astype(int)]
    selected_calls = worstRemoval(call_costs, rm_size, rng)
    call_cost_loc = [call_costs[2, selected_calls[i] -1].astype(int) for i in range(rm_size)]
    sel_loc = np.array([np.where(Solution == np.int32(selected_calls[i]))[0] for i in range(rm_size)])
    
    cur_v = []
    for c_id, call in enumerate(selected_calls):
        cur_v.append(len(ZeroIndexBef[ZeroIndexBef<sel_loc[c_id][0]]))
        if cur_v[c_id] == prob['n_vehicles']:
            Solution = Solution[:sel_loc[c_id][0]] + Solution[sel_loc[c_id][0]+1:sel_loc[c_id][1]] + Solution[sel_loc[c_id][1]+1:]
            ZeroIndexBef[cur_v[c_id]:] -= 2
            len_v[cur_v[c_id]] -= 2
            sel_loc[sel_loc>sel_loc[c_id][1]]-=1
            sel_loc[sel_loc>sel_loc[c_id][0]]-=1
            sel_loc[c_id][1] += 1
        else:
            sol_2 = Solution[:sel_loc[c_id][0]] + Solution[sel_loc[c_id][0]+1:sel_loc[c_id][1]] + Solution[sel_loc[c_id][1]+1:]
            best_cost = cur_cost4V(cur_v[c_id], Solution[ZeroIndexBef[cur_v[c_id]]-len_v[cur_v[c_id]]:ZeroIndexBef[cur_v[c_id]]],
                                  sol_2[ZeroIndexBef[cur_v[c_id]]-len_v[cur_v[c_id]]:ZeroIndexBef[cur_v[c_id]]-2],
                                  *(sel_loc[c_id]-(sum(len_v[:cur_v[c_id]])+ len(len_v[:cur_v[c_id]]))),features[2],prob)
            
            Solution = sol_2
            ZeroIndexBef[cur_v[c_id]:] -= 2
            len_v[cur_v[c_id]] -= 2
            sel_loc[sel_loc>sel_loc[c_id][1]]-=1
            sel_loc[sel_loc>sel_loc[c_id][0]]-=1
            sel_loc[c_id][1] += 1
            features = features_delete(features, cur_v[c_id], *(sel_loc[c_id]-(sum(len_v[:cur_v[c_id]])+ len(len_v[:cur_v[c_id]]))))
            costs[cur_v[c_id]] -= best_cost
            costs[-1] += prob['Cargo'][call-1,3]
    temp = list(zip(selected_calls,call_cost_loc))
    rng.shuffle(temp)
    selected_calls,call_cost_loc = zip(*temp)
    call_cost_loc = list(call_cost_loc)
    for s, selected_call in enumerate(selected_calls):
        avail_vehs = [i for i in range(prob['n_vehicles']) if prob['VesselCargo'][i,selected_call-1] == 1]+ [prob['n_vehicles']]
        best_features = copy_features(features)
        
        best_v = prob['n_vehicles']
        best_cost = np.inf
        for v in avail_vehs:
            if v == prob['n_vehicles']:
                if best_cost >= prob['Cargo'][selected_call-1,3]:
                    best_sol = Solution[:-1] + [selected_call,selected_call] + Solution[-1:]
                    best_v = v
                    best_call_costs = copy_call_costs(call_costs)
                    best_cost = prob['Cargo'][selected_call-1,3]
                    best_call_costs[0, call_cost_loc[s]] = prob['Cargo'][selected_call-1,3]
                    best_call_costs, changes = update_call_costs(best_call_costs, selected_call, call_cost_loc, s)
                    best_features = copy_features(features)
                continue
            for pos in range(len_v[v]+1):
                cost_v = costs[v]
                new_sol = Solution[:ZeroIndexBef[v]+ pos - len_v[v]] +[selected_call]+ Solution[ZeroIndexBef[v]+ pos - len_v[v]:]
                
                del_idx, new_cost, feasibility, c, new_features = \
                    findBestPosForDel(selected_call, v, new_sol[ZeroIndexBef[v]-len_v[v]:ZeroIndexBef[v]+1],
                                      pos, copy_features(features), prob)
                if not c.startswith('F'):
                    if int(c[-1]) == pos or int(c[-1]) == del_idx:
                        Solution = new_sol[:ZeroIndexBef[v]+ pos - len_v[v]] + new_sol[ZeroIndexBef[v]+ pos - len_v[v]+1:]
                        if c.startswith('T'):
                            break
                        else:
                            continue
                    else:
                        Solution = new_sol[:ZeroIndexBef[v]+ pos - len_v[v]] + new_sol[ZeroIndexBef[v]+ pos - len_v[v]+1:]
                        continue
                    
                elif new_cost - cost_v <= best_cost:
                    best_sol = new_sol[:ZeroIndexBef[v]-len_v[v]+ del_idx] +[selected_call]+ new_sol[ZeroIndexBef[v]-len_v[v]+ del_idx:]
                    best_cost = new_cost - cost_v
                    best_v = v
                    best_call_costs = copy_call_costs(call_costs)
                    best_call_costs[0, call_cost_loc[s]] = new_cost - cost_v
                    best_call_costs, changes = update_call_costs(best_call_costs, selected_call, call_cost_loc, s)
                    best_features = new_features
                else:
                   Solution = new_sol[:ZeroIndexBef[v]+ pos - len_v[v]] + new_sol[ZeroIndexBef[v]+ pos - len_v[v]+1:]
        call_cost_loc = list(map(sum, zip(call_cost_loc, changes)))      
        Solution = best_sol
        features = copy_features(best_features)
        call_costs = copy_call_costs(best_call_costs)
        ZeroIndexBef[best_v:] += 2
        len_v[best_v] += 2
        if best_v != prob['n_vehicles']:
            costs[best_v] += best_cost
            costs[-1] -= prob['Cargo'][selected_call-1,3]
    
    return Solution [:-1], costs, features, best_call_costs