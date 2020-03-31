"""
# Original Work Copyright © 2018-present, Mean Field Reinforcement Learning (https://github.com/mlii/mfrl). 
# All rights reserved.
# Modifications (c) 2020-present, Royal Bank of Canada.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""


import random
import math
import numpy as np

 

def generate_map(env, map_size, handles):
    width = map_size
    height = map_size

    init_num = map_size * map_size * 0.04

    gap = 3
    # left
    n = init_num
    side = int(math.sqrt(n)) * 2
    pos = [[], []]
    ct = 0
    for x in range(width//2 - gap - side, width//2 - gap - side + side, 2):
        for y in range((height - side)//2, (height - side)//2 + side, 2):
            pos[ct % 2].append([x, y])
        ct += 1
    env.add_agents(handles[0], method="custom", pos=pos[0])
    env.add_agents(handles[1], method="custom", pos=pos[1])

    # right
    n = init_num
    side = int(math.sqrt(n)) * 2
    pos = [[], []]
    ct = 0
    for x in range(width//2 + gap, width//2 + gap + side, 2):
        for y in range((height - side)//2, (height - side)//2 + side, 2):
            pos[ct % 2].append([x, y])
        ct += 1
    env.add_agents(handles[2], method="custom", pos=pos[0])
    env.add_agents(handles[3], method="custom", pos=pos[1])


def play(env, n_round, map_size, max_steps, handles, models, print_every, eps=1.0, mtmfq_position = 0, render=False, train=False):
    """play a round and train"""
    env.reset()
    generate_map(env, map_size, handles)

    step_ct = 0
    done = False

    n_group = len(handles)
    state = [None for _ in range(n_group)]
    acts = [None for _ in range(n_group)]
    ids = [None for _ in range(n_group)]

    alives = [None for _ in range(n_group)]
    rewards = [None for _ in range(n_group)]
    nums = [env.get_num(handle) for handle in handles]
    max_nums = nums.copy()
    loss = [None for _ in range(n_group)]
    eval_q = [None for _ in range(n_group)]
    n_action = [env.get_action_space(handles[0])[0], env.get_action_space(handles[1])[0], env.get_action_space(handles[2])[0], env.get_action_space(handles[3])[0]]

    print("\n\n[*] ROUND #{0}, EPS: {1:.2f} NUMBER: {2}".format(n_round, eps, nums))
    mean_rewards = [[] for _ in range(n_group)]
    total_rewards = [[] for _ in range(n_group)]

    former_act_prob = [np.zeros((1, env.get_action_space(handles[0])[0])), np.zeros((1, env.get_action_space(handles[1])[0])), np.zeros((1, env.get_action_space(handles[2])[0])), np.zeros((1, env.get_action_space(handles[3])[0]))]

    while not done and step_ct < max_steps:
        # take actions for every model
        for i in range(n_group):
            state[i] = list(env.get_observation(handles[i])) 
            ids[i] = env.get_agent_id(handles[i]) 

        for i in range(n_group):
            former_act_prob[i] = np.tile(former_act_prob[i], (len(state[i][0]), 1))
            acts[i] = models[i].act(state=state[i], prob=former_act_prob[i], eps=eps) 

        for i in range(n_group):
            env.set_action(handles[i], acts[i])

        # simulate one step
        done = env.step()

        for i in range(n_group):
            rewards[i] = env.get_reward(handles[i])
            alives[i] = env.get_alive(handles[i])

        buffer = {
            'state': state[0], 'acts': acts[0], 'rewards': rewards[0],
            'alives': alives[0], 'ids': ids[0]
        }

        buffer['prob'] = former_act_prob[0]

        if train:
            models[0].flush_buffer(**buffer)

        buffer = {
            'state': state[1], 'acts': acts[1], 'rewards': rewards[1],
            'alives': alives[1], 'ids': ids[1]
        }

        buffer['prob'] = former_act_prob[1]

        if train:
            models[1].flush_buffer(**buffer)
        
        
        buffer = {
            'state': state[2], 'acts': acts[2], 'rewards': rewards[2],
            'alives': alives[2], 'ids': ids[2]
        }

        buffer['prob'] = former_act_prob[2]

        if train:
            models[2].flush_buffer(**buffer)
        
        
        
        buffer = {
            'state': state[3], 'acts': acts[3], 'rewards': rewards[3],
            'alives': alives[3], 'ids': ids[3]
        }

        buffer['prob'] = former_act_prob[3]

        if train:
            models[3].flush_buffer(**buffer)
        
        acts_new = []
        for i in range(n_group):
            for j in range(len(acts[i])):
                    acts_new.append(acts[i][j])
        
        
        for i in range(n_group):            
            former_act_prob[i] = np.mean(list(map(lambda x: np.eye(n_action[i])[x], acts_new)), axis=0, keepdims=True)


        # stat info
        nums = [env.get_num(handle) for handle in handles]

        for i in range(n_group):
            sum_reward = sum(rewards[i])
            rewards[i] = sum_reward / nums[i]
            mean_rewards[i].append(rewards[i])
            total_rewards[i].append(sum_reward)

        if render:
            env.render()

        # clear dead agents
        env.clear_dead()

        info = {"Ave-Reward": np.round(rewards, decimals=6), "NUM": nums}

        step_ct += 1

        if step_ct % print_every == 0:
            print("> step #{}, info: {}".format(step_ct, info))

    if train:
        models[0].train()
        models[1].train()
        models[2].train()
        models[3].train()

    for i in range(n_group):
        mean_rewards[i] = sum(mean_rewards[i]) / len(mean_rewards[i])
        total_rewards[i] = sum(total_rewards[i])

    return max_nums, nums, mean_rewards, total_rewards


def play2(env, n_round, map_size, max_steps, handles, models, print_every, mtmfq_position = 0, eps=1.0, render=False, train=False):
    """play a round and train"""
    env.reset()
    generate_map(env, map_size, handles)

    step_ct = 0
    done = False

    n_group = len(handles)
    state = [None for _ in range(n_group)]
    acts = [None for _ in range(n_group)]
    ids = [None for _ in range(n_group)]

    alives = [None for _ in range(n_group)]
    rewards = [None for _ in range(n_group)]
    nums = [env.get_num(handle) for handle in handles]
    max_nums = nums.copy()
    loss = [None for _ in range(n_group)]
    eval_q = [None for _ in range(n_group)]
    n_action = [env.get_action_space(handles[0])[0], env.get_action_space(handles[1])[0], env.get_action_space(handles[2])[0], env.get_action_space(handles[3])[0]]

    print("\n\n[*] ROUND #{0}, EPS: {1:.2f} NUMBER: {2}".format(n_round, eps, nums))
    mean_rewards = [[] for _ in range(n_group)]
    total_rewards = [[] for _ in range(n_group)]

    former_act_prob = [np.zeros((1, env.get_action_space(handles[0])[0])), np.zeros((1, env.get_action_space(handles[1])[0])), np.zeros((1, env.get_action_space(handles[2])[0])), np.zeros((1, env.get_action_space(handles[3])[0]))]

    former_act_prob0 = [np.zeros((1, env.get_action_space(handles[0])[0])), np.zeros((1, env.get_action_space(handles[1])[0])), np.zeros((1, env.get_action_space(handles[2])[0])), np.zeros((1, env.get_action_space(handles[3])[0]))]
    
    former_act_prob1 = [np.zeros((1, env.get_action_space(handles[0])[0])), np.zeros((1, env.get_action_space(handles[1])[0])), np.zeros((1, env.get_action_space(handles[2])[0])), np.zeros((1, env.get_action_space(handles[3])[0]))]
    
    former_act_prob2 = [np.zeros((1, env.get_action_space(handles[0])[0])), np.zeros((1, env.get_action_space(handles[1])[0])), np.zeros((1, env.get_action_space(handles[2])[0])), np.zeros((1, env.get_action_space(handles[3])[0]))]
     
    former_act_prob3 = [np.zeros((1, env.get_action_space(handles[0])[0])), np.zeros((1, env.get_action_space(handles[1])[0])), np.zeros((1, env.get_action_space(handles[2])[0])), np.zeros((1, env.get_action_space(handles[3])[0]))]
    
    while not done and step_ct < max_steps:
        # take actions for every model
        for i in range(n_group):
            state[i] = list(env.get_observation(handles[i]))
            ids[i] = env.get_agent_id(handles[i])

        for i in range(n_group):
            former_act_prob0[i] = np.tile(former_act_prob[0], (len(state[i][0]), 1)) 
            former_act_prob1[i] = np.tile(former_act_prob[1], (len(state[i][0]), 1)) 
            former_act_prob2[i] = np.tile(former_act_prob[2], (len(state[i][0]), 1)) 
            former_act_prob3[i] = np.tile(former_act_prob[3], (len(state[i][0]), 1)) 
            acts[i] = models[i].act(state=state[i], prob0=former_act_prob0[i], prob1=former_act_prob1[i], prob2=former_act_prob2[i], prob3=former_act_prob3[i], eps=eps) 

        for i in range(n_group):
            env.set_action(handles[i], acts[i])

        # simulate one step
        done = env.step()

        for i in range(n_group):
            rewards[i] = env.get_reward(handles[i])
            alives[i] = env.get_alive(handles[i])

        buffer = {
            'state': state[0], 'acts': acts[0], 'rewards': rewards[0],
            'alives': alives[0], 'ids': ids[0]
        }

        buffer['prob0'] = former_act_prob0[0]
        buffer['prob1'] = former_act_prob1[0]
        buffer['prob2'] = former_act_prob2[0]
        buffer['prob3'] = former_act_prob3[0]


        if train:
            models[0].flush_buffer(**buffer)

        
        buffer = {
            'state': state[1], 'acts': acts[1], 'rewards': rewards[1],
            'alives': alives[1], 'ids': ids[1]
        }

        buffer['prob0'] = former_act_prob0[1]
        buffer['prob1'] = former_act_prob1[1]
        buffer['prob2'] = former_act_prob2[1]
        buffer['prob3'] = former_act_prob3[1]


        if train:
            models[1].flush_buffer(**buffer)
        
        
        
        buffer = {
            'state': state[2], 'acts': acts[2], 'rewards': rewards[2],
            'alives': alives[2], 'ids': ids[2]
        }

        buffer['prob0'] = former_act_prob0[2]
        buffer['prob1'] = former_act_prob1[2]
        buffer['prob2'] = former_act_prob2[2]
        buffer['prob3'] = former_act_prob3[2]


        if train:
            models[2].flush_buffer(**buffer)
        
        
        
        
        buffer = {
            'state': state[3], 'acts': acts[3], 'rewards': rewards[3],
            'alives': alives[3], 'ids': ids[3]
        }

        buffer['prob0'] = former_act_prob0[3]
        buffer['prob1'] = former_act_prob1[3]
        buffer['prob2'] = former_act_prob2[3]
        buffer['prob3'] = former_act_prob3[3]


        if train:
            models[3].flush_buffer(**buffer)
        
        
        
        for i in range(n_group):
            
            former_act_prob[i] = np.mean(list(map(lambda x: np.eye(n_action[i])[x], acts[i])), axis=0, keepdims=True)
        # stat info
        nums = [env.get_num(handle) for handle in handles]

        for i in range(n_group):
            sum_reward = sum(rewards[i])
            rewards[i] = sum_reward / nums[i]
            mean_rewards[i].append(rewards[i])
            total_rewards[i].append(sum_reward)

        if render:
            env.render()

        # clear dead agents
        env.clear_dead()

        info = {"Ave-Reward": np.round(rewards, decimals=6), "NUM": nums}

        step_ct += 1

        if step_ct % print_every == 0:
            print("> step #{}, info: {}".format(step_ct, info))

    if train:
        models[0].train()
        models[1].train()
        models[2].train()
        models[3].train()

    for i in range(n_group):
        mean_rewards[i] = sum(mean_rewards[i]) / len(mean_rewards[i])
        total_rewards[i] = sum(total_rewards[i])

    return max_nums, nums, mean_rewards, total_rewards







def battle(env, n_round, map_size, max_steps, handles, models, print_every, mtmfq_position=0, eps=1.0, render=False, train=False):
    """play a round and train"""
    env.reset()
    generate_map(env, map_size, handles)

    step_ct = 0
    done = False

    n_group = len(handles)
    state = [None for _ in range(n_group)]
    acts = [None for _ in range(n_group)]
    ids = [None for _ in range(n_group)]

    alives = [None for _ in range(n_group)]
    rewards = [None for _ in range(n_group)]
    nums = [env.get_num(handle) for handle in handles]
    max_nums = nums.copy()

    n_action = [env.get_action_space(handles[0])[0], env.get_action_space(handles[1])[0], env.get_action_space(handles[2])[0], env.get_action_space(handles[3])[0]]

    print("\n\n[*] ROUND #{0}, EPS: {1:.2f} NUMBER: {2}".format(n_round, eps, nums))
    mean_rewards = [[] for _ in range(n_group)]
    total_rewards = [[] for _ in range(n_group)]

    former_act_prob = [np.zeros((1, env.get_action_space(handles[0])[0])), np.zeros((1, env.get_action_space(handles[1])[0])), np.zeros((1, env.get_action_space(handles[2])[0])), np.zeros((1, env.get_action_space(handles[3])[0]))]

    former_act_prob0 = [np.zeros((1, env.get_action_space(handles[0])[0])), np.zeros((1, env.get_action_space(handles[1])[0])), np.zeros((1, env.get_action_space(handles[2])[0])), np.zeros((1, env.get_action_space(handles[3])[0]))]
    
    former_act_prob1 = [np.zeros((1, env.get_action_space(handles[0])[0])), np.zeros((1, env.get_action_space(handles[1])[0])), np.zeros((1, env.get_action_space(handles[2])[0])), np.zeros((1, env.get_action_space(handles[3])[0]))]
    
    former_act_prob2 = [np.zeros((1, env.get_action_space(handles[0])[0])), np.zeros((1, env.get_action_space(handles[1])[0])), np.zeros((1, env.get_action_space(handles[2])[0])), np.zeros((1, env.get_action_space(handles[3])[0]))]
     
    former_act_prob3 = [np.zeros((1, env.get_action_space(handles[0])[0])), np.zeros((1, env.get_action_space(handles[1])[0])), np.zeros((1, env.get_action_space(handles[2])[0])), np.zeros((1, env.get_action_space(handles[3])[0]))]
    
    former_act_prob_new = [np.zeros((1, env.get_action_space(handles[0])[0])), np.zeros((1, env.get_action_space(handles[1])[0])), np.zeros((1, env.get_action_space(handles[2])[0])), np.zeros((1, env.get_action_space(handles[3])[0]))]

    while not done and step_ct < max_steps:
        # take actions for every model
        for i in range(n_group):
            state[i] = list(env.get_observation(handles[i]))
            ids[i] = env.get_agent_id(handles[i])

        for i in range(n_group):

            if i == mtmfq_position:
                former_act_prob0[i] = np.tile(former_act_prob[0], (len(state[i][0]), 1)) 
                former_act_prob1[i] = np.tile(former_act_prob[1], (len(state[i][0]), 1)) 
                former_act_prob2[i] = np.tile(former_act_prob[2], (len(state[i][0]), 1)) 
                former_act_prob3[i] = np.tile(former_act_prob[3], (len(state[i][0]), 1)) 
                acts[i] = models[i].act(state=state[i], prob0=former_act_prob0[i], prob1=former_act_prob1[i], prob2=former_act_prob2[i], prob3=former_act_prob3[i], eps=eps)

            else: 
                former_act_prob_new[i] = np.tile(former_act_prob_new[i], (len(state[i][0]), 1))
                acts[i] = models[i].act(state=state[i], prob=former_act_prob_new[i], eps=eps)

        for i in range(n_group):
            env.set_action(handles[i], acts[i])

        # simulate one step
        done = env.step()

        for i in range(n_group):
            rewards[i] = env.get_reward(handles[i])
            alives[i] = env.get_alive(handles[i])

        for i in range(n_group):
            former_act_prob[i] = np.mean(list(map(lambda x: np.eye(n_action[i])[x], acts[i])), axis=0, keepdims=True)

        acts_new = []
        for i in range(n_group):
            for j in range(len(acts[i])):
                    acts_new.append(acts[i][j])
        
        
        for i in range(n_group):    
            former_act_prob_new[i] = np.mean(list(map(lambda x: np.eye(n_action[i])[x], acts_new)), axis=0, keepdims=True)
        
        
        # stat info
        nums = [env.get_num(handle) for handle in handles]

        for i in range(n_group):
            sum_reward = sum(rewards[i])
            rewards[i] = sum_reward / nums[i]
            mean_rewards[i].append(rewards[i])
            total_rewards[i].append(sum_reward)

        if render:
            env.render()

        # clear dead agents
        env.clear_dead()

        info = {"Ave-Reward": np.round(rewards, decimals=6), "NUM": nums}

        step_ct += 1

        if step_ct % print_every == 0:
            print("> step #{}, info: {}".format(step_ct, info))

    for i in range(n_group):
        mean_rewards[i] = sum(mean_rewards[i]) / len(mean_rewards[i])
        total_rewards[i] = sum(total_rewards[i])

    return max_nums, nums, mean_rewards, total_rewards

