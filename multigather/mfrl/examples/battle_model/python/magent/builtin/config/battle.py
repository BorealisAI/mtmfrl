"""
# Original Work Copyright © 2018-present, Mean Field Reinforcement Learning (https://github.com/mlii/mfrl). 
# All rights reserved.
# Modifications (c) 2020-present, Royal Bank of Canada.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""



""" battle of two armies """

import magent


def get_config(map_size):
    gw = magent.gridworld
    cfg = gw.Config()

    cfg.set({"map_width": map_size, "map_height": map_size})
    cfg.set({"minimap_mode": True})
    cfg.set({"embedding_size": 10})

    agent = cfg.register_agent_type(
        name="agent",
        attr={'width': 1, 'length': 1, 'hp': 3, 'speed': 3,
              'view_range': gw.CircleRange(7), 'attack_range': gw.CircleRange(1),
              'damage': 6, 'step_recover': 0,
              'step_reward': -0.01,  'dead_penalty': -1, 'attack_penalty': -0.1,
              'attack_in_group': 1})

    food = cfg.register_agent_type(
        name='food',
        attr={'width': 1, 'length': 1, 'hp': 25, 'speed': 0,
              'view_range': gw.CircleRange(1), 'attack_range': gw.CircleRange(0),
              'kill_reward': 80})

    g_f = cfg.add_group(food)
    g_s = cfg.add_group(agent)
    g_s2 = cfg.add_group(agent)
    g_s3 = cfg.add_group(agent)
    g_s4 = cfg.add_group(agent)

    a = gw.AgentSymbol(g_s, index='any')
    b = gw.AgentSymbol(g_s2, index='any')
    c = gw.AgentSymbol(g_s3, index='any')
    d = gw.AgentSymbol(g_s4, index='any')
    e = gw.AgentSymbol(g_f, index='any')
    # reward shaping to encourage attack

    cfg.add_reward_rule(gw.Event(a, 'attack', e), receiver=a, value=0.5)
    cfg.add_reward_rule(gw.Event(b, 'attack', e), receiver=b, value=0.5)
    cfg.add_reward_rule(gw.Event(c, 'attack', e), receiver=c, value=0.5)
    cfg.add_reward_rule(gw.Event(d, 'attack', e), receiver=d, value=0.5)
    cfg.add_reward_rule(gw.Event(a, 'attack', b), receiver=a, value=0.2)
    cfg.add_reward_rule(gw.Event(a, 'attack', c), receiver=a, value=0.3)
    cfg.add_reward_rule(gw.Event(a, 'attack', d), receiver=a, value=0.4)
    cfg.add_reward_rule(gw.Event(b, 'attack', a), receiver=b, value=0.4)
    cfg.add_reward_rule(gw.Event(b, 'attack', c), receiver=b, value=0.2)
    cfg.add_reward_rule(gw.Event(b, 'attack', d), receiver=b, value=0.3)
    cfg.add_reward_rule(gw.Event(c, 'attack', d), receiver=c, value=0.2)
    cfg.add_reward_rule(gw.Event(c, 'attack', a), receiver=c, value=0.3)
    cfg.add_reward_rule(gw.Event(c, 'attack', b), receiver=c, value=0.4)
    cfg.add_reward_rule(gw.Event(d, 'attack', a), receiver=d, value=0.2)
    cfg.add_reward_rule(gw.Event(d, 'attack', b), receiver=d, value=0.3)
    cfg.add_reward_rule(gw.Event(d, 'attack', c), receiver=d, value=0.4)
    cfg.add_reward_rule(gw.Event(a, 'kill', b), receiver=a, value=80)
    cfg.add_reward_rule(gw.Event(a, 'kill', c), receiver=a, value=90)
    cfg.add_reward_rule(gw.Event(a, 'kill', d), receiver=a, value=100)
    cfg.add_reward_rule(gw.Event(b, 'kill', a), receiver=b, value=100)
    cfg.add_reward_rule(gw.Event(b, 'kill', c), receiver=b, value=80)
    cfg.add_reward_rule(gw.Event(b, 'kill', d), receiver=b, value=90)
    cfg.add_reward_rule(gw.Event(c, 'kill', d), receiver=c, value=80)
    cfg.add_reward_rule(gw.Event(c, 'kill', a), receiver=c, value=90)
    cfg.add_reward_rule(gw.Event(c, 'kill', b), receiver=c, value=100)
    cfg.add_reward_rule(gw.Event(d, 'kill', a), receiver=d, value=80)
    cfg.add_reward_rule(gw.Event(d, 'kill', b), receiver=d, value=90)
    cfg.add_reward_rule(gw.Event(d, 'kill', c), receiver=d, value=100)

    return cfg
