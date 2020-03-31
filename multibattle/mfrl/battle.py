"""
# Original Work Copyright © 2018-present, Mean Field Reinforcement Learning (https://github.com/mlii/mfrl). 
# All rights reserved.
# Modifications (c) 2020-present, Royal Bank of Canada.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""


"""Battle
"""

import argparse
import os
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
import magent
import csv 
from examples.battle_model.algo import spawn_ai
from examples.battle_model.algo import tools
from examples.battle_model.senario_battle import battle

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--algo', type=str, choices={'ac', 'mfac', 'mfq', 'il', 'mtmfq'}, help='choose an algorithm from the preset', required=True)
    parser.add_argument('--oppo1', type=str, choices={'ac', 'mfac', 'mfq', 'il', 'mtmfq'}, help='indicate the first opponent model')
    parser.add_argument('--oppo2', type=str, choices={'ac', 'mfac', 'mfq', 'il', 'mtmfq'}, help='indicate the second opponent model')
    parser.add_argument('--oppo3', type=str, choices={'ac', 'mfac', 'mfq', 'il', 'mtmfq'}, help='indicate the third opponent model')
    parser.add_argument('--n_round', type=int, default=50, help='set the total number of games')
    parser.add_argument('--render', action='store_true', help='render or not (if true, will render every save)')
    parser.add_argument('--map_size', type=int, default=60, help='set the size of map')  # then the amount of agents is 72
    parser.add_argument('--max_steps', type=int, default=500, help='set the max steps')
    parser.add_argument('--idx', nargs='*', required=True)
    parser.add_argument('--mtmfqp', type=int, choices={0,1,2,3}, default=0, help='set the position of mtmfq')
    
    
    args = parser.parse_args()

    # Initialize the environment
    env = magent.GridWorld('battle', map_size=args.map_size)
    env.set_render_dir(os.path.join(BASE_DIR, 'examples/battle_model', 'build/render'))
    handles = env.get_handles()
    mtmfq_position = args.mtmfqp
    tf_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    tf_config.gpu_options.allow_growth = True

    main_model_dir = os.path.join(BASE_DIR, 'data/models/{}-0'.format(args.algo))
    oppo_model_dir1 = os.path.join(BASE_DIR, 'data/models/{}-1'.format(args.oppo1))
    oppo_model_dir2 = os.path.join(BASE_DIR, 'data/models/{}-2'.format(args.oppo2))
    oppo_model_dir3 = os.path.join(BASE_DIR, 'data/models/{}-3'.format(args.oppo3))

    sess = tf.Session(config=tf_config)
    models = [spawn_ai(args.algo, sess, env, handles[0], args.algo + '-me', args.max_steps), spawn_ai(args.oppo1, sess, env, handles[1], args.oppo1 + '-opponent1', args.max_steps), spawn_ai(args.oppo2, sess, env, handles[2], args.oppo2 + '-opponent2', args.max_steps), spawn_ai(args.oppo3, sess, env, handles[3], args.oppo3 + '-opponent3', args.max_steps)]
    sess.run(tf.global_variables_initializer())

    models[0].load(main_model_dir, step=args.idx[0])
    models[1].load(oppo_model_dir1, step=args.idx[1])
    models[2].load(oppo_model_dir2, step=args.idx[2])
    models[3].load(oppo_model_dir3, step=args.idx[3])

    runner = tools.Runner(sess, env, handles, args.map_size, args.max_steps, models, battle, mtmfq_position, render_every=0)
    win_cnt = {'main': 0, 'opponent1': 0, 'opponent2': 0, 'opponent3': 0}
    total_rewards = []
    with open('storepoints_multibattle.csv', 'w+') as myfile:
        myfile.write('{0},{1},{2},{3},{4}\n'.format("Game", "Reward 1", "Reward 2","Reward 3", "Reward 4"))
    for k in range(0, args.n_round):
        total_rewards = runner.run(0.0, k, win_cnt=win_cnt)
        with open('storepoints_multibattle.csv', 'a') as myfile:
            myfile.write('{0},{1},{2},{3},{4}\n'.format(k, total_rewards[0], total_rewards[1],total_rewards[2], total_rewards[3]))

    print('\n[*] >>> WIN_RATE: [{0}] {1} / [{2}] {3} / [{4}] {5} / [{6}] {7} '.format(args.algo, win_cnt['main'] / args.n_round, args.oppo1, win_cnt['opponent1'] / args.n_round, args.oppo2, win_cnt['opponent2'] / args.n_round, args.oppo3, win_cnt['opponent3'] / args.n_round))





