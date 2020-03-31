"""
# Original Work Copyright © 2018-present, Mean Field Reinforcement Learning (https://github.com/mlii/mfrl). 
# All rights reserved.
# Modifications (c) 2020-present, Royal Bank of Canada.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""



import magent

def get_config(map_size):
    gw = magent.gridworld
    cfg = gw.Config()

    cfg.set({"map_width": map_size, "map_height": map_size})

    predator = cfg.register_agent_type(
        "predator",
        {
            'width': 2, 'length': 2, 'hp': 10, 'speed': 2,
            'view_range': gw.CircleRange(7), 'attack_range': gw.CircleRange(2), 'damage': 1, 'step_recover': 0.1, 'kill_reward': 5, 'dead_penalty': -0.1, 'attack_penalty': -0.2
        })

    prey = cfg.register_agent_type(
        "prey",
        {
            'width': 1, 'length': 1, 'hp': 10, 'speed': 2.5,
            'view_range': gw.CircleRange(6), 'attack_range': gw.CircleRange(0), 'damage': 2, 'step_recover': 0.1, 'kill_reward':5, 'dead_penalty': -0.1 
        })

    predator_group  = cfg.add_group(predator)
    predator_group2  = cfg.add_group(predator)
    prey_group = cfg.add_group(prey)
    prey_group2 = cfg.add_group(prey)

    a = gw.AgentSymbol(predator_group, index='any')
    c = gw.AgentSymbol(predator_group2, index='any')
    b = gw.AgentSymbol(prey_group, index='any')
    d = gw.AgentSymbol(prey_group2, index = 'any')

    cfg.add_reward_rule(gw.Event(a, 'attack', b), receiver=[a, b], value=[1, -1])
    cfg.add_reward_rule(gw.Event(a, 'attack', d), receiver=[a, d], value=[3, -3])
    cfg.add_reward_rule(gw.Event(c, 'attack', d), receiver=[c, d], value=[1, -1])
    cfg.add_reward_rule(gw.Event(c, 'attack', b), receiver=[c, b], value=[3, -3])
    cfg.add_reward_rule(gw.Event(a, 'attack', c), receiver=[a, c], value=[0.5, -0.5])
    cfg.add_reward_rule(gw.Event(c, 'attack', a), receiver=[c, a], value=[0.5, -0.5])


    return cfg



