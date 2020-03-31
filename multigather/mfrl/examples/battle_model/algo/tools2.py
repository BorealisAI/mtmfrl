"""
# Copyright (c) 2018-present, Royal Bank of Canada.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""



import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import os


class Color:
    INFO = '\033[1;34m{}\033[0m'
    WARNING = '\033[1;33m{}\033[0m'
    ERROR = '\033[1;31m{}\033[0m'


class Buffer:
    def __init__(self):
        pass

    def push(self, **kwargs):
        raise NotImplementedError


class MetaBuffer(object):
    def __init__(self, shape, max_len, dtype='float32'):
        self.max_len = max_len
        self.data = np.zeros((max_len,) + shape).astype(dtype)
        self.start = 0
        self.length = 0
        self._flag = 0

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if idx < 0 or idx >= self.length:
            raise KeyError()
        return self.data[idx]

    def sample(self, idx):
        return self.data[idx % self.length]

    def pull(self):
        return self.data[:self.length]

    def append(self, value):
        start = 0
        num = len(value)

        if self._flag + num > self.max_len:
            tail = self.max_len - self._flag
            self.data[self._flag:] = value[:tail]
            num -= tail
            start = tail
            self._flag = 0

        self.data[self._flag:self._flag + num] = value[start:]
        self._flag += num
        self.length = min(self.length + len(value), self.max_len)

    def reset_new(self, start, value):
        self.data[start:] = value


class EpisodesBufferEntry:
    """Entry for episode buffer"""
    def __init__(self):
        self.views = []
        self.features = []
        self.actions = []
        self.rewards = []
        self.prob0 = []
        self.prob1 = []
        self.prob2 = []
        self.prob3 = []
        self.terminal = False


    def append(self, view, feature, action, reward, alive, prob0=None,  prob1=None,  prob2=None,  prob3=None):
        self.views.append(view.copy())
        self.features.append(feature.copy())
        self.actions.append(action)
        self.rewards.append(reward)
        if prob0 is not None:
            self.prob0.append(prob0)
            self.prob1.append(prob1)
            self.prob2.append(prob2)
            self.prob3.append(prob3)
        if not alive:
            self.terminal = True


class EpisodesBuffer(Buffer):
    """Replay buffer to store a whole episode for all agents
       one entry for one agent
    """
    def __init__(self, use_mean=False):
        super().__init__()
        self.buffer = {}
        self.use_mean = use_mean

    def push(self, **kwargs):
        view, feature = kwargs['state']
        acts = kwargs['acts']
        rewards = kwargs['rewards']
        alives = kwargs['alives']
        ids = kwargs['ids']

        if self.use_mean:
            prob0 = kwargs['prob0']
            prob1 = kwargs['prob1']
            prob2 = kwargs['prob2']
            prob3 = kwargs['prob3']

        buffer = self.buffer
        index = np.random.permutation(len(view))

        for i in range(len(ids)):
            i = index[i]
            entry = buffer.get(ids[i])
            if entry is None:
                entry = EpisodesBufferEntry()
                buffer[ids[i]] = entry

            if self.use_mean:
                entry.append(view[i], feature[i], acts[i], rewards[i], alives[i], prob0=prob0[i], prob1=prob1[i], prob2=prob2[i], prob3=prob3[i])
            else:
                entry.append(view[i], feature[i], acts[i], rewards[i], alives[i])

    def reset(self):
        """ clear replay buffer """
        self.buffer = {}

    def episodes(self):
        """ get episodes """
        return self.buffer.values()


class AgentMemory(object):
    def __init__(self, obs_shape, feat_shape, act_n, max_len, use_mean=False):
        self.obs0 = MetaBuffer(obs_shape, max_len)
        self.feat0 = MetaBuffer(feat_shape, max_len)
        self.actions = MetaBuffer((), max_len, dtype='int32')
        self.rewards = MetaBuffer((), max_len)
        self.terminals = MetaBuffer((), max_len, dtype='bool')
        self.use_mean = use_mean

        if self.use_mean:
            self.prob0 = MetaBuffer((act_n,), max_len)
            self.prob1 = MetaBuffer((act_n,), max_len)
            self.prob2 = MetaBuffer((act_n,), max_len)
            self.prob3 = MetaBuffer((act_n,), max_len)

    def append(self, obs0, feat0, act, reward, alive, prob0=None,  prob1=None,  prob2=None,  prob3=None):
        self.obs0.append(np.array([obs0]))
        self.feat0.append(np.array([feat0]))
        self.actions.append(np.array([act], dtype=np.int32))
        self.rewards.append(np.array([reward]))
        self.terminals.append(np.array([not alive], dtype=np.bool))

        if self.use_mean:
            self.prob0.append(np.array([prob0]))
            self.prob1.append(np.array([prob1]))
            self.prob2.append(np.array([prob2]))
            self.prob3.append(np.array([prob3]))


    def pull(self):
        res = {
            'obs0': self.obs0.pull(),
            'feat0': self.feat0.pull(),
            'act': self.actions.pull(),
            'rewards': self.rewards.pull(),
            'terminals': self.terminals.pull(),
            'prob0': None if not self.use_mean else self.prob0.pull(),
            'prob1': None if not self.use_mean else self.prob1.pull(),
            'prob2': None if not self.use_mean else self.prob2.pull(),
            'prob3': None if not self.use_mean else self.prob3.pull()
        }

        return res


class MemoryGroup(object):
    def __init__(self, obs_shape, feat_shape, act_n, max_len, batch_size, sub_len, use_mean=False):
        self.agent = dict()
        self.max_len = max_len
        self.batch_size = batch_size
        self.obs_shape = obs_shape
        self.feat_shape = feat_shape
        self.sub_len = sub_len
        self.use_mean = use_mean
        self.act_n = act_n

        self.obs0 = MetaBuffer(obs_shape, max_len)
        self.feat0 = MetaBuffer(feat_shape, max_len)
        self.actions = MetaBuffer((), max_len, dtype='int32')
        self.rewards = MetaBuffer((), max_len)
        self.terminals = MetaBuffer((), max_len, dtype='bool')
        self.masks = MetaBuffer((), max_len, dtype='bool')
        if use_mean:
            self.prob0 = MetaBuffer((act_n,), max_len)
            self.prob1 = MetaBuffer((act_n,), max_len)
            self.prob2 = MetaBuffer((act_n,), max_len)
            self.prob3 = MetaBuffer((act_n,), max_len)
        self._new_add = 0

    def _flush(self, **kwargs):
        self.obs0.append(kwargs['obs0'])
        self.feat0.append(kwargs['feat0'])
        self.actions.append(kwargs['act'])
        self.rewards.append(kwargs['rewards'])
        self.terminals.append(kwargs['terminals'])

        if self.use_mean:
            self.prob0.append(kwargs['prob0'])
            self.prob1.append(kwargs['prob1'])
            self.prob2.append(kwargs['prob2'])
            self.prob3.append(kwargs['prob3'])

        mask = np.where(kwargs['terminals'] == True, False, True)
        mask[-1] = False
        self.masks.append(mask)

    def push(self, **kwargs):
        for i, _id in enumerate(kwargs['ids']):
            if self.agent.get(_id) is None:
                self.agent[_id] = AgentMemory(self.obs_shape, self.feat_shape, self.act_n, self.sub_len, use_mean=self.use_mean)
            if self.use_mean:
                self.agent[_id].append(obs0=kwargs['state'][0][i], feat0=kwargs['state'][1][i], act=kwargs['acts'][i], reward=kwargs['rewards'][i], alive=kwargs['alives'][i], prob0=kwargs['prob0'][i], prob1=kwargs['prob1'][i], prob2=kwargs['prob2'][i], prob3=kwargs['prob3'][i])
            else:
                self.agent[_id].append(obs0=kwargs['state'][0][i], feat0=kwargs['state'][1][i], act=kwargs['acts'][i], reward=kwargs['rewards'][i], alive=kwargs['alives'][i])

    def tight(self):
        ids = list(self.agent.keys())
        np.random.shuffle(ids)
        for ele in ids:
            tmp = self.agent[ele].pull()
            self._new_add += len(tmp['obs0'])
            self._flush(**tmp)
        self.agent = dict()  # clear

    def sample(self):
        idx = np.random.choice(self.nb_entries, size=self.batch_size)
        next_idx = (idx + 1) % self.nb_entries

        obs = self.obs0.sample(idx)
        obs_next = self.obs0.sample(next_idx)
        feature = self.feat0.sample(idx)
        feature_next = self.feat0.sample(next_idx)
        actions = self.actions.sample(idx)
        rewards = self.rewards.sample(idx)
        dones = self.terminals.sample(idx)
        masks = self.masks.sample(idx)

        if self.use_mean:
            act_prob0 = self.prob0.sample(idx)
            act_prob1 = self.prob1.sample(idx)
            act_prob2 = self.prob2.sample(idx)
            act_prob3 = self.prob3.sample(idx)
            act_next_prob0 = self.prob0.sample(next_idx)
            act_next_prob1 = self.prob1.sample(next_idx)
            act_next_prob2 = self.prob2.sample(next_idx)
            act_next_prob3 = self.prob3.sample(next_idx)
            return obs, feature, actions, act_prob0, act_prob1, act_prob2, act_prob3, obs_next, feature_next, act_next_prob0, act_next_prob1, act_next_prob2, act_next_prob3, rewards, dones, masks
        else:
            return obs, feature, obs_next, feature_next, dones, rewards, actions, masks

    def get_batch_num(self):
        print('\n[INFO] Length of buffer and new add:', len(self.obs0), self._new_add)
        res = self._new_add * 2 // self.batch_size
        self._new_add = 0
        return res

    @property
    def nb_entries(self):
        return len(self.obs0)


class SummaryObj:
    """
    Define a summary holder
    """
    def __init__(self, log_dir, log_name, n_group=1):
        self.name_set = set()
        self.gra = tf.Graph()
        self.n_group = n_group

        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        sess_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        sess_config.gpu_options.allow_growth = True

        with self.gra.as_default():
            self.sess = tf.Session(graph=self.gra, config=sess_config)
            self.train_writer = tf.summary.FileWriter(log_dir + "/" + log_name, graph=tf.get_default_graph())
            self.sess.run(tf.global_variables_initializer())

    def register(self, name_list):
        """Register summary operations with a list contains names for these operations

        Parameters
        ----------
        name_list: list, contains name whose type is str
        """

        with self.gra.as_default():
            for name in name_list:
                if name in self.name_set:
                    raise Exception("You cannot define different operations with same name: `{}`".format(name))
                self.name_set.add(name)
                setattr(self, name, [tf.placeholder(tf.float32, shape=None, name='Agent_{}_{}'.format(i, name))
                                     for i in range(self.n_group)])
                setattr(self, name + "_op", [tf.summary.scalar('Agent_{}_{}_op'.format(i, name), getattr(self, name)[i])
                                             for i in range(self.n_group)])

    def write(self, summary_dict, step):
        """Write summary related to a certain step

        Parameters
        ----------
        summary_dict: dict, summary value dict
        step: int, global step
        """

        assert isinstance(summary_dict, dict)

        for key, value in summary_dict.items():
            if key not in self.name_set:
                raise Exception("Undefined operation: `{}`".format(key))
            if isinstance(value, list):
                for i in range(self.n_group):
                    self.train_writer.add_summary(self.sess.run(getattr(self, key + "_op")[i], feed_dict={
                        getattr(self, key)[i]: value[i]}), global_step=step)
            else:
                self.train_writer.add_summary(self.sess.run(getattr(self, key + "_op")[0], feed_dict={
                        getattr(self, key)[0]: value}), global_step=step)


class Runner(object):
    def __init__(self, sess, env, handles, map_size, max_steps, models,
                play_handle, mtmfq_position=0, render_every=None, save_every=None, tau=None, log_name=None, log_dir=None, model_dir=None, train=False):
        """Initialize runner

        Parameters
        ----------
        sess: tf.Session
            session
        env: magent.GridWorld
            environment handle
        handles: list
            group handles
        map_size: int
            map size of grid world
        max_steps: int
            the maximum of stages in a episode
        render_every: int
            render environment interval
        save_every: int
            states the interval of evaluation for self-play update
        models: list
            contains models
        play_handle: method like
            run game
        tau: float
            tau index for self-play update
        log_name: str
            define the name of log dir
        log_dir: str
            donates the directory of logs
        model_dir: str
            donates the dircetory of models
        """
        self.env = env
        self.models = models
        self.max_steps = max_steps
        self.handles = handles
        self.map_size = map_size
        self.render_every = render_every
        self.save_every = save_every
        self.play = play_handle
        self.model_dir = model_dir
        self.train = train
        self.mtmfq_position = mtmfq_position

        if self.train:
            self.summary = SummaryObj(log_name=log_name, log_dir=log_dir)

            summary_items = ['ave_agent_reward', 'total_reward', 'alive', "Sum_Reward", "Kill_Sum"]
            self.summary.register(summary_items)  # summary register
            self.summary_items = summary_items

            assert isinstance(sess, tf.Session)
            assert self.models[0].name_scope != self.models[1].name_scope
            self.sess = sess

            if not os.path.exists(self.model_dir):
                os.makedirs(self.model_dir)



    def run(self, variant_eps, iteration, win_cnt=None):
        info = {'main': None, 'opponent1': None, 'opponent2': None, 'opponent3': None}

        total_reward_main = []

        # pass
        info['main'] = {'ave_agent_reward': 0., 'total_reward': 0., 'alive': 0.}
        info['opponent1'] = {'ave_agent_reward': 0., 'total_reward': 0., 'alive': 0.}
        info['opponent2'] = {'ave_agent_reward': 0., 'total_reward': 0., 'alive': 0.}
        info['opponent3'] = {'ave_agent_reward': 0., 'total_reward': 0., 'alive': 0.}
        max_nums, nums, agent_r_records, total_rewards = self.play(env=self.env, n_round=iteration, map_size=self.map_size, max_steps=self.max_steps, handles=self.handles,
                    models=self.models, mtmfq_position = self.mtmfq_position, print_every=50, eps=variant_eps, render=(iteration + 1) % self.render_every if self.render_every > 0 else False, train=self.train)

        for i, tag in enumerate(['main', 'opponent1', 'opponent2', 'opponent3']):
            info[tag]['total_reward'] = total_rewards[i]
            info[tag]['alive'] = nums[i]
            info[tag]['ave_agent_reward'] = agent_r_records[i]

        if self.train:
            print('\n[INFO] {}'.format(info['main']))

            if self.save_every and (iteration + 1) % self.save_every == 0:

                self.summary.write(info['main'], iteration)
                print(Color.INFO.format('[INFO] Saving model ...'))
                self.models[0].save(self.model_dir + '-0', iteration)
                self.models[1].save(self.model_dir + '-1', iteration)
                self.models[2].save(self.model_dir + '-2', iteration)
                self.models[3].save(self.model_dir + '-3', iteration)


        else:
            print('\n[INFO] {0} \n {1} \n {2} \n {3}'''.format(info['main'], info['opponent1'], info['opponent2'], info['opponent3']))
            if info['main']['alive'] >= info['opponent1']['alive'] and info['main']['alive'] >= info['opponent2']['alive'] and info['main']['alive'] >= info['opponent3']['alive']:
                win_cnt['main'] += 1
            if info['opponent1']['alive'] >= info['main']['alive'] and info['opponent1']['alive'] >= info['opponent2']['alive'] and info['opponent1']['alive'] >= info['opponent3']['alive']:
                win_cnt['opponent1'] += 1

            if info['opponent2']['alive'] >= info['main']['alive'] and info['opponent2']['alive'] >= info['opponent1']['alive'] and info['opponent2']['alive'] >= info['opponent3']['alive']:
                win_cnt['opponent2'] += 1

            if info['opponent3']['alive'] >= info['main']['alive'] and info['opponent3']['alive'] >= info['opponent2']['alive'] and info['opponent3']['alive'] >= info['opponent1']['alive']:
                win_cnt['opponent3'] += 1


        return total_rewards



