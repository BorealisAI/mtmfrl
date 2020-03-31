"""
# Original Work Copyright © 2018-present, Mean Field Reinforcement Learning (https://github.com/mlii/mfrl). 
# All rights reserved.
# Modifications (c) 2020-present, Royal Bank of Canada.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""



"""Self Play
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
from examples.battle_model.senario_battle import play
from examples.battle_model.senario_battle import play2


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def linear_decay(epoch, x, y):
    min_v, max_v = y[0], y[-1]
    start, end = x[0], x[-1]

    if epoch == start:
        return min_v

    eps = min_v

    for i, x_i in enumerate(x):
        if epoch <= x_i:
            interval = (y[i] - y[i - 1]) / (x_i - x[i - 1])
            eps = interval * (epoch - x[i - 1]) + y[i - 1]
            break

    return eps

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--algo', type=str, choices={'ac', 'mfac', 'mfq', 'il','mtmfq'}, help='choose an algorithm from the preset', required=True)
    parser.add_argument('--save_every', type=int, default=10, help='decide the self-play update interval')
    parser.add_argument('--update_every', type=int, default=10, help='decide the udpate interval for q-learning, optional')
    parser.add_argument('--n_round', type=int, default=10, help='set the trainning round')
    parser.add_argument('--render', action='store_true', help='render or not (if true, will render every save)')
    parser.add_argument('--map_size', type=int, default=60, help='set the size of map')  # then the amount of predators is 45 and prey is 90
    parser.add_argument('--max_steps', type=int, default=500, help='set the max steps')

    args = parser.parse_args()

    # Initialize the environment
    env = magent.GridWorld('battle', map_size=args.map_size)
    env.set_render_dir(os.path.join(BASE_DIR, 'examples/battle_model', 'build/render'))
    handles = env.get_handles()

    tf_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    tf_config.gpu_options.allow_growth = True

    log_dir = os.path.join(BASE_DIR,'data/tmp'.format(args.algo))
    model_dir = os.path.join(BASE_DIR, 'data/models/{}'.format(args.algo))

    with open('predator.csv', 'w+') as myfile:
        myfile.write('{0},{1}\n'.format("Episode", "Reward"))
    
    if args.algo in ['mfq', 'mfac', 'mtmfq']:
        use_mf = True
    else:
        use_mf = False

    start_from = 0
    total_reward = []
    meanerrortotal = []
    sess = tf.Session(config=tf_config)
    models = [spawn_ai(args.algo, sess, env, handles[0], args.algo + '-me', args.max_steps), spawn_ai(args.algo, sess, env, handles[1], args.algo + '-opponent1', args.max_steps), spawn_ai(args.algo, sess, env, handles[2], args.algo + '-opponent2', args.max_steps), spawn_ai(args.algo, sess, env, handles[3], args.algo + '-opponent3', args.max_steps)]
    sess.run(tf.global_variables_initializer())
    if args.algo == 'mtmfq':
        runner = tools.Runner(sess, env, handles, args.map_size, args.max_steps, models, play2,
                            render_every=args.save_every if args.render else 0, save_every=args.save_every, tau=0.01, log_name=args.algo,
                            log_dir=log_dir, model_dir=model_dir, train=True)

    else:
        runner = tools.Runner(sess, env, handles, args.map_size, args.max_steps, models, play,
                            render_every=args.save_every if args.render else 0, save_every=args.save_every, tau=0.01, log_name=args.algo,
                            log_dir=log_dir, model_dir=model_dir, train=True)
    
        
    for k in range(start_from, start_from + args.n_round):
        eps = linear_decay(k, [0, int(args.n_round * 0.8), args.n_round], [1, 0.2, 0.1])
        total_reward = runner.run(eps, k)
        with open('predator.csv', 'a') as myfile:
            myfile.write('{0},{1}\n'.format(k, total_reward[0]))
        print("Writen to file")

