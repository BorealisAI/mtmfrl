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
import argparse
import logging as log
import time

def generate_map(env, map_size, food_handle, handles):
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

    center_x, center_y = map_size // 2, map_size // 2

    def add_square(pos, side, gap):
        side = int(side)
        for x in range(center_x - side//2, center_x + side//2 + 1, gap):
            pos.append([x, center_y - side//2])
            pos.append([x, center_y + side//2])
        for y in range(center_y - side//2, center_y + side//2 + 1, gap):
            pos.append([center_x - side//2, y])
            pos.append([center_x + side//2, y])


       # food
    pos = []
    add_square(pos, map_size * 0.65, 10)
    add_square(pos, map_size * 0.6,  10)
    add_square(pos, map_size * 0.55, 10)
    add_square(pos, map_size * 0.5,  4)
    add_square(pos, map_size * 0.45, 3)
    add_square(pos, map_size * 0.4, 1)
    add_square(pos, map_size * 0.3, 1)
    add_square(pos, map_size * 0.3 - 2, 1)
    add_square(pos, map_size * 0.3 - 4, 1)
    add_square(pos, map_size * 0.3 - 6, 1)
    env.add_agents(food_handle, method="custom", pos=pos)

        # legend
    legend = [
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,],
        [1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0,],
        [1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0,],
        [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0,],
        [1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0,],
        [1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0,],
        [1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0,],
        [1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0,],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,],
    ]

    org = [
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0,],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0,],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0,],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0,],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0,],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,],
    ]

    def draw(base_x, base_y, scale, data):
        w, h = len(data), len(data[0])
        pos = []
        for i in range(w):
            for j in range(h):
                if data[i][j] == 1:
                    start_x = i * scale + base_x
                    start_y = j * scale + base_y
                    for x in range(start_x, start_x + scale):
                        for y in range(start_y, start_y + scale):
                            pos.append([y, x])

        env.add_agents(food_handle, method="custom", pos=pos)

    scale = 1
    w, h = len(legend), len(legend[0])
    offset = -3
    draw(offset + map_size // 2 - w // 2 * scale, map_size // 2 - h // 2 * scale, scale, legend)
    draw(offset + map_size // 2 - w // 2 * scale + len(legend), map_size // 2 - h // 2 * scale, scale, org)


def play(env, n_round, map_size, max_steps, handles, models, print_every, mtmfq_position = 0, eps=1.0, render=False, train=False):
    """play a round and train"""
    env.reset()
    food_handle = handles[0] 
    handles = handles[1:]
    generate_map(env, map_size, food_handle, handles)

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
        food_num = env.get_num(food_handle)
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
            print("Remaining food is:", food_num)

    if train:
        models[0].train()
        models[1].train()
        models[2].train()
        models[3].train()

    for i in range(n_group):
        mean_rewards[i] = sum(mean_rewards[i]) / len(mean_rewards[i])
        total_rewards[i] = sum(total_rewards[i])

    return max_nums, nums, mean_rewards, total_rewards




def play2(env, n_round, map_size, max_steps, handles, models, print_every, eps=1.0, mtmfq_position = 0, render=False, train=False):
    """play a round and train"""
    env.reset()
    food_handle = handles[0] 
    handles = handles[1:]
    generate_map(env, map_size, food_handle, handles)

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
        food_num = env.get_num(food_handle)

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
            print("Remaining food is", food_num)

    if train:
        models[0].train()
        models[1].train()
        models[2].train()
        models[3].train()

    for i in range(n_group):
        mean_rewards[i] = sum(mean_rewards[i]) / len(mean_rewards[i])
        total_rewards[i] = sum(total_rewards[i])

    return max_nums, nums, mean_rewards, total_rewards


def battle(env, n_round, map_size, max_steps, handles, models, print_every, eps=1.0, mtmfq_position=0, render=False, train=False):
    """play a round and train"""

    env.reset()
    food_handle = handles[0] 
    handles = handles[1:]
    generate_map(env, map_size, food_handle, handles)

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

