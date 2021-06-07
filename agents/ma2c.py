import numpy as np
import tensorflow as tf

from agents.agent import Agent, IndependentAgent
from signal_config import signal_configs


class MA2C(IndependentAgent):
    def __init__(self, config, obs_act, map_name, thread_number, sess=None):
        super().__init__(config, obs_act, map_name, thread_number)

        self.signal_config = signal_configs[map_name]

        if sess is None:
            tf.reset_default_graph()
            cfg_proto = tf.ConfigProto(allow_soft_placement=True)
            self.sess = tf.Session(config=cfg_proto)

        for key in obs_act:
            obs_space = obs_act[key][0]
            act_space = obs_act[key][1]

            # Get waiting size
            lane_sets = self.signal_config[key]['lane_sets']
            lanes = []
            for direction in lane_sets:
                for lane in lane_sets[direction]:
                    if lane not in lanes: lanes.append(lane)
            waits_len = len(lanes)

            # Get fingerprint size
            downstream = self.signal_config[key]['downstream']
            neighbors = [downstream[direction] for direction in downstream]
            fp_size = 0
            for neighbor in neighbors:
                if neighbor is not None:
                    fp_size += obs_act[neighbor][1]     # neighbor's action size

            self.agents[key] = MA2CAgent(config, obs_space, act_space, fp_size, waits_len, 'ma2c'+key + str(thread_number), self.sess)

        if sess is None:
            self.saver = tf.train.Saver(max_to_keep=1)
            self.sess.run(tf.global_variables_initializer())
        else:
            self.saver = None

    def fingerprints(self, observation):
        agent_fingerprint = {}
        for agent_id in observation.keys():
            downstream = self.signal_config[agent_id]['downstream']
            neighbors = [downstream[direction] for direction in downstream]
            fingerprints = []
            for neighbor in neighbors:
                if neighbor is not None:
                    neighbor_fp = self.agents[neighbor].fingerprint
                    fingerprints.append(neighbor_fp)
            agent_fingerprint[agent_id] = np.concatenate(fingerprints)
        return agent_fingerprint

    def act(self, observation):
        acts = {}
        fingerprints = self.fingerprints(observation)
        for agent_id in observation.keys():
            env_obs = observation[agent_id]
            neighbor_fingerprints = fingerprints[agent_id]
            combine = np.concatenate([env_obs, neighbor_fingerprints])

            acts[agent_id] = self.agents[agent_id].act(combine)
        return acts

    def observe(self, observation, reward, done, info):
        fingerprints = self.fingerprints(observation)
        for agent_id in observation.keys():
            env_obs = observation[agent_id]
            neighbor_fingerprints = fingerprints[agent_id]
            combine = np.concatenate([env_obs, neighbor_fingerprints])

            agent = self.agents[agent_id]
            agent.observe(combine, reward[agent_id], done, info)

        if done:
            if info['eps'] % 100 == 0:
                if self.saver is not None:
                    self.saver.save(self.sess, self.config['log_dir']+'agent_' + 'checkpoint', global_step=info['eps'])


class MA2CAgent(Agent):
    def __init__(self, config, observation_shape, num_actions, fingerprint_size, waits_len, name, sess):
        super().__init__()
        self.config = config
        self.num_actions = num_actions
        self.sess = sess

        self.steps_done = 0
        self.state = None
        self.value = None
        self.action = None
        self.fingerprint = np.zeros(num_actions)

        n_s = observation_shape[0] + fingerprint_size
        n_a = num_actions
        n_w = waits_len
        n_f = fingerprint_size
        total_step = config['steps']
        model_config = config
        print(name, n_s, n_a, n_w, n_f)

        self.model = MA2CImplementation(n_s, n_a, n_w, n_f, total_step, model_config, name, sess)

    def act(self, observation):
        self.state = observation

        policy, self.value = self.model.forward(observation, False)
        self.action = np.random.choice(np.arange(len(policy)), p=policy)
        self.fingerprint = np.array(policy)

        return self.action

    def observe(self, observation, reward, done, info):
        self.model.add_transition(self.state, self.action, reward, self.value, done)
        self.steps_done += 1

        if self.steps_done % self.config['batch_size'] == 0 or done:
            if done:
                R = 0
            else:
                R = self.model.forward(observation, False, 'v')
            self.model.backward(R)

        if done:
            self.steps_done = 0
            self.model.reset()


# https://github.com/cts198859/deeprl_signal_control
class MA2CImplementation:
    def __init__(self, n_s, n_a, n_w, n_f, total_step, model_config, name, sess):
        self.name = name
        self.sess = sess
        self.reward_clip = model_config['reward_clip']
        self.reward_norm = model_config['reward_norm']
        self.n_s = n_s
        self.n_a = n_a
        self.n_f = n_f
        self.n_w = n_w
        self.n_step = model_config['batch_size']

        # agent_name is needed to differentiate multi-agents
        self.policy = self._init_policy(n_s - n_f - n_w, n_a, n_w, n_f, model_config, agent_name=name)

        if total_step:
            # training
            self.total_step = total_step
            self._init_scheduler(model_config)
            self._init_train(model_config)

    def _init_policy(self, n_s, n_a, n_w, n_f, model_config, agent_name=None):
        n_fw = model_config['num_fw']
        n_ft = model_config['num_ft']
        n_lstm = model_config['num_lstm']
        n_fp = model_config['num_fp']
        policy = FPLstmACPolicy(n_s, n_a, n_w, n_f, self.n_step, n_fc_wave=n_fw,
                                n_fc_wait=n_ft, n_fc_fp=n_fp, n_lstm=n_lstm, name=agent_name)
        return policy

    def _init_scheduler(self, model_config):
        lr_init = model_config['lr_init']
        lr_decay = model_config['lr_decay']
        beta_init = model_config['entropy_coef_init']
        beta_decay = model_config['entropy_decay']
        if lr_decay == 'constant':
            self.lr_scheduler = Scheduler(lr_init, decay=lr_decay)
        else:
            lr_min = model_config['LR_MIN']
            self.lr_scheduler = Scheduler(lr_init, lr_min, self.total_step, decay=lr_decay)
        if beta_decay == 'constant':
            self.beta_scheduler = Scheduler(beta_init, decay=beta_decay)
        else:
            beta_min = model_config['ENTROPY_COEF_MIN']
            beta_ratio = model_config['ENTROPY_RATIO']
            self.beta_scheduler = Scheduler(beta_init, beta_min, self.total_step * beta_ratio,
                                            decay=beta_decay)

    def _init_train(self, model_config):
        # init loss
        v_coef = model_config['value_coef']
        max_grad_norm = model_config['max_grad_norm']
        alpha = model_config['rmsp_alpha']
        epsilon = model_config['rmsp_epsilon']
        gamma = model_config['gamma']
        self.policy.prepare_loss(v_coef, max_grad_norm, alpha, epsilon)
        self.trans_buffer = OnPolicyBuffer(gamma)

    def backward(self, R, summary_writer=None, global_step=None):
        cur_lr = self.lr_scheduler.get(self.n_step)
        cur_beta = self.beta_scheduler.get(self.n_step)
        obs, acts, dones, Rs, Advs = self.trans_buffer.sample_transition(R)
        self.policy.backward(self.sess, obs, acts, dones, Rs, Advs, cur_lr, cur_beta)

    def forward(self, obs, done, out_type='pv'):
        return self.policy.forward(self.sess, obs, done, out_type)

    def reset(self):
        self.policy._reset()

    def add_transition(self, obs, actions, rewards, values, done):
        if (self.reward_norm):
            rewards = rewards / self.reward_norm
        if self.reward_clip:
            rewards = np.clip(rewards, -self.reward_clip, self.reward_clip)
        self.trans_buffer.add_transition(obs, actions, rewards, values, done)

    """def load(self, model_dir, checkpoint=None):
        save_file = None
        save_step = 0
        if os.path.exists(model_dir):
            if checkpoint is None:
                for file in os.listdir(model_dir):
                    if file.startswith('checkpoint'):
                        prefix = file.split('.')[0]
                        tokens = prefix.split('-')
                        if len(tokens) != 2:
                            continue
                        cur_step = int(tokens[1])
                        if cur_step > save_step:
                            save_file = prefix
                            save_step = cur_step
            else:
                save_file = 'checkpoint-' + str(int(checkpoint))
        if save_file is not None:
            self.saver.restore(self.sess, model_dir + save_file)
            logging.info('Checkpoint loaded: %s' % save_file)
            return True
        logging.error('Can not find old checkpoint for %s' % model_dir)
        return False"""


class ACPolicy:
    def __init__(self, n_a, n_s, n_step, policy_name, agent_name):
        self.name = policy_name
        if agent_name is not None:
            # for multi-agent system
            self.name += '_' + str(agent_name)
        self.n_a = n_a
        self.n_s = n_s
        self.n_step = n_step

    def forward(self, ob, *_args, **_kwargs):
        raise NotImplementedError()

    def _build_out_net(self, h, out_type):
        if out_type == 'pi':
            pi = fc(h, out_type, self.n_a, act=tf.nn.softmax)
            return tf.squeeze(pi)
        else:
            v = fc(h, out_type, 1, act=lambda x: x)
            return tf.squeeze(v)

    def _get_forward_outs(self, out_type):
        outs = []
        if 'p' in out_type:
            outs.append(self.pi)
        if 'v' in out_type:
            outs.append(self.v)
        return outs

    def _return_forward_outs(self, out_values):
        if len(out_values) == 1:
            return out_values[0]
        return out_values

    def prepare_loss(self, v_coef, max_grad_norm, alpha, epsilon):
        self.A = tf.placeholder(tf.int32, [self.n_step])
        self.ADV = tf.placeholder(tf.float32, [self.n_step])
        self.R = tf.placeholder(tf.float32, [self.n_step])
        self.entropy_coef = tf.placeholder(tf.float32, [])
        A_sparse = tf.one_hot(self.A, self.n_a)
        log_pi = tf.log(tf.clip_by_value(self.pi, 1e-10, 1.0))
        entropy = -tf.reduce_sum(self.pi * log_pi, axis=1)
        entropy_loss = -tf.reduce_mean(entropy) * self.entropy_coef
        policy_loss = -tf.reduce_mean(tf.reduce_sum(log_pi * A_sparse, axis=1) * self.ADV)
        value_loss = tf.reduce_mean(tf.square(self.R - self.v)) * 0.5 * v_coef
        self.loss = policy_loss + value_loss + entropy_loss

        wts = tf.trainable_variables(scope=self.name)
        grads = tf.gradients(self.loss, wts)
        if max_grad_norm > 0:
            grads, self.grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)
        self.lr = tf.placeholder(tf.float32, [])
        self.optimizer = tf.train.RMSPropOptimizer(learning_rate=self.lr, decay=alpha,
                                                   epsilon=epsilon)
        self._train = self.optimizer.apply_gradients(list(zip(grads, wts)))
        # monitor training
        if self.name.endswith('_0a'):
            summaries = []
            # summaries.append(tf.summary.scalar('loss/%s_entropy_loss' % self.name, entropy_loss))
            summaries.append(tf.summary.scalar('loss/%s_policy_loss' % self.name, policy_loss))
            summaries.append(tf.summary.scalar('loss/%s_value_loss' % self.name, value_loss))
            summaries.append(tf.summary.scalar('loss/%s_total_loss' % self.name, self.loss))
            # summaries.append(tf.summary.scalar('train/%s_lr' % self.name, self.lr))
            # summaries.append(tf.summary.scalar('train/%s_entropy_beta' % self.name, self.entropy_coef))
            summaries.append(tf.summary.scalar('train/%s_gradnorm' % self.name, self.grad_norm))
            self.summary = tf.summary.merge(summaries)


class LstmACPolicy(ACPolicy):
    def __init__(self, n_s, n_a, n_w, n_step, n_fc_wave=128, n_fc_wait=32, n_lstm=64, name=None):
        super().__init__(n_a, n_s, n_step, 'lstm', name)
        self.n_lstm = n_lstm
        self.n_fc_wait = n_fc_wait
        self.n_fc_wave = n_fc_wave
        self.n_w = n_w
        self.ob_fw = tf.placeholder(tf.float32, [1, n_s + n_w])     # forward 1-step
        self.done_fw = tf.placeholder(tf.float32, [1])
        self.ob_bw = tf.placeholder(tf.float32, [n_step, n_s + n_w])     # backward n-step
        self.done_bw = tf.placeholder(tf.float32, [n_step])
        self.states = tf.placeholder(tf.float32, [2, n_lstm * 2])
        with tf.variable_scope(self.name):
            # pi and v use separate nets
            self.pi_fw, pi_state = self._build_net('forward', 'pi')
            self.v_fw, v_state = self._build_net('forward', 'v')
            pi_state = tf.expand_dims(pi_state, 0)
            v_state = tf.expand_dims(v_state, 0)
            self.new_states = tf.concat([pi_state, v_state], 0)
        with tf.variable_scope(self.name, reuse=True):
            self.pi, _ = self._build_net('backward', 'pi')
            self.v, _ = self._build_net('backward', 'v')
        self._reset()

    def _build_net(self, in_type, out_type):
        if in_type == 'forward':
            ob = self.ob_fw
            done = self.done_fw
        else:
            ob = self.ob_bw
            done = self.done_bw
        if out_type == 'pi':
            states = self.states[0]
        else:
            states = self.states[1]
        if self.n_w == 0:
            h = fc(ob, out_type + '_fcw', self.n_fc_wave)
        else:
            h0 = fc(ob[:, :self.n_s], out_type + '_fcw', self.n_fc_wave)
            h1 = fc(ob[:, self.n_s:], out_type + '_fct', self.n_fc_wait)
            h = tf.concat([h0, h1], 1)
        h, new_states = lstm(h, done, states, out_type + '_lstm')
        out_val = self._build_out_net(h, out_type)
        return out_val, new_states

    def _reset(self):
        # forget the cumulative states every cum_step
        self.states_fw = np.zeros((2, self.n_lstm * 2), dtype=np.float32)
        self.states_bw = np.zeros((2, self.n_lstm * 2), dtype=np.float32)

    def forward(self, sess, ob, done, out_type='pv'):
        outs = self._get_forward_outs(out_type)
        # update state only when p is called
        if 'p' in out_type:
            outs.append(self.new_states)
        out_values = sess.run(outs, {self.ob_fw: np.array([ob]),
                                     self.done_fw: np.array([done]),
                                     self.states: self.states_fw})
        if 'p' in out_type:
            self.states_fw = out_values[-1]
            out_values = out_values[:-1]
        return self._return_forward_outs(out_values)

    def backward(self, sess, obs, acts, dones, Rs, Advs, cur_lr, cur_beta,
                 summary_writer=None, global_step=None):
        if summary_writer is None:
            ops = self._train
        else:
            ops = [self.summary, self._train]
        outs = sess.run(ops,
                        {self.ob_bw: obs,
                         self.done_bw: dones,
                         self.states: self.states_bw,
                         self.A: acts,
                         self.ADV: Advs,
                         self.R: Rs,
                         self.lr: cur_lr,
                         self.entropy_coef: cur_beta})
        self.states_bw = np.copy(self.states_fw)
        if summary_writer is not None:
            summary_writer.add_summary(outs[0], global_step=global_step)

    def _get_forward_outs(self, out_type):
        outs = []
        if 'p' in out_type:
            outs.append(self.pi_fw)
        if 'v' in out_type:
            outs.append(self.v_fw)
        return outs


class FPLstmACPolicy(LstmACPolicy):
    def __init__(self, n_s, n_a, n_w, n_f, n_step, n_fc_wave=128, n_fc_wait=32, n_fc_fp=32, n_lstm=64, name=None):
        ACPolicy.__init__(self, n_a, n_s, n_step, 'fplstm', name)
        self.n_lstm = n_lstm
        self.n_fc_wave = n_fc_wave
        self.n_fc_wait = n_fc_wait
        self.n_fc_fp = n_fc_fp
        self.n_w = n_w
        self.ob_fw = tf.placeholder(tf.float32, [1, n_s + n_w + n_f])   # forward 1-step
        self.done_fw = tf.placeholder(tf.float32, [1])
        self.ob_bw = tf.placeholder(tf.float32, [n_step, n_s + n_w + n_f])  # backward n-step
        self.done_bw = tf.placeholder(tf.float32, [n_step])
        self.states = tf.placeholder(tf.float32, [2, n_lstm * 2])
        with tf.variable_scope(self.name):
            # pi and v use separate nets
            self.pi_fw, pi_state = self._build_net('forward', 'pi')
            self.v_fw, v_state = self._build_net('forward', 'v')
            pi_state = tf.expand_dims(pi_state, 0)
            v_state = tf.expand_dims(v_state, 0)
            self.new_states = tf.concat([pi_state, v_state], 0)
        with tf.variable_scope(self.name, reuse=True):
            self.pi, _ = self._build_net('backward', 'pi')
            self.v, _ = self._build_net('backward', 'v')
        self._reset()

    def _build_net(self, in_type, out_type):
        if in_type == 'forward':
            ob = self.ob_fw
            done = self.done_fw
        else:
            ob = self.ob_bw
            done = self.done_bw
        if out_type == 'pi':
            states = self.states[0]
        else:
            states = self.states[1]
        h0 = fc(ob[:, :self.n_s], out_type + '_fcw', self.n_fc_wave)
        h1 = fc(ob[:, (self.n_s + self.n_w):], out_type + '_fcf', self.n_fc_fp)
        if self.n_w == 0:
            h = tf.concat([h0, h1], 1)
        else:
            h2 = fc(ob[:, self.n_s: (self.n_s + self.n_w)], out_type + '_fct', self.n_fc_wait)
            h = tf.concat([h0, h1, h2], 1)
        h, new_states = lstm(h, done, states, out_type + '_lstm')
        out_val = self._build_out_net(h, out_type)
        return out_val, new_states


DEFAULT_SCALE = np.sqrt(2)
DEFAULT_MODE = 'fan_in'


def ortho_init(scale=DEFAULT_SCALE, mode=None):
    def _ortho_init(shape, dtype, partition_info=None):
        # lasagne ortho init for tf
        shape = tuple(shape)
        if len(shape) == 2:     # fc: in, out
            flat_shape = shape
        elif (len(shape) == 3) or (len(shape) == 4):    # 1d/2dcnn: (in_h), in_w, in_c, out
            flat_shape = (np.prod(shape[:-1]), shape[-1])
        a = np.random.standard_normal(flat_shape)
        u, _, v = np.linalg.svd(a, full_matrices=False)
        q = u if u.shape == flat_shape else v   # pick the one with the correct shape
        q = q.reshape(shape)
        return (scale * q).astype(np.float32)
    return _ortho_init


DEFAULT_METHOD = ortho_init


def fc(x, scope, n_out, act=tf.nn.relu, init_scale=DEFAULT_SCALE,
       init_mode=DEFAULT_MODE, init_method=DEFAULT_METHOD):
    with tf.variable_scope(scope):
        n_in = x.shape[1].value
        w = tf.get_variable("w", [n_in, n_out],
                            initializer=init_method(init_scale, init_mode))
        b = tf.get_variable("b", [n_out], initializer=tf.constant_initializer(0.0))
        z = tf.matmul(x, w) + b
        return act(z)


def batch_to_seq(x):
    n_step = x.shape[0].value
    if len(x.shape) == 1:
        x = tf.expand_dims(x, -1)
    return tf.split(axis=0, num_or_size_splits=n_step, value=x)


def seq_to_batch(x):
    return tf.concat(axis=0, values=x)


def lstm(xs, dones, s, scope, init_scale=DEFAULT_SCALE, init_mode=DEFAULT_MODE,
         init_method=DEFAULT_METHOD):
    xs = batch_to_seq(xs)
    # need dones to reset states
    dones = batch_to_seq(dones)
    n_in = xs[0].shape[1].value
    n_out = s.shape[0] // 2
    with tf.variable_scope(scope):
        wx = tf.get_variable("wx", [n_in, n_out*4],
                             initializer=init_method(init_scale, init_mode))
        wh = tf.get_variable("wh", [n_out, n_out*4],
                             initializer=init_method(init_scale, init_mode))
        b = tf.get_variable("b", [n_out*4], initializer=tf.constant_initializer(0.0))
    s = tf.expand_dims(s, 0)
    c, h = tf.split(axis=1, num_or_size_splits=2, value=s)
    for ind, (x, done) in enumerate(zip(xs, dones)):
        c = c * (1-done)
        h = h * (1-done)
        z = tf.matmul(x, wx) + tf.matmul(h, wh) + b
        i, f, o, u = tf.split(axis=1, num_or_size_splits=4, value=z)
        i = tf.nn.sigmoid(i)
        f = tf.nn.sigmoid(f)
        o = tf.nn.sigmoid(o)
        u = tf.tanh(u)
        c = f*c + i*u
        h = o*tf.tanh(c)
        xs[ind] = h
    s = tf.concat(axis=1, values=[c, h])
    return seq_to_batch(xs), tf.squeeze(s)


class Scheduler:
    def __init__(self, val_init, val_min=0, total_step=0, decay='linear'):
        self.val = val_init
        self.N = float(total_step)
        self.val_min = val_min
        self.decay = decay
        self.n = 0

    def get(self, n_step):
        self.n += n_step
        if self.decay == 'linear':
            return max(self.val_min, self.val * (1 - self.n / self.N))
        else:
            return self.val


class TransBuffer:
    def reset(self):
        self.buffer = []

    @property
    def size(self):
        return len(self.buffer)

    def add_transition(self, ob, a, r, *_args, **_kwargs):
        raise NotImplementedError()

    def sample_transition(self, *_args, **_kwargs):
        raise NotImplementedError()


class OnPolicyBuffer(TransBuffer):
    def __init__(self, gamma):
        self.gamma = gamma
        self.reset()

    def reset(self, done=False):
        # the done before each step is required
        self.obs = []
        self.acts = []
        self.rs = []
        self.vs = []
        self.dones = [done]

    def add_transition(self, ob, a, r, v, done):
        self.obs.append(ob)
        self.acts.append(a)
        self.rs.append(r)
        self.vs.append(v)
        self.dones.append(done)

    def _add_R_Adv(self, R):
        Rs = []
        Advs = []
        # use post-step dones here
        for r, v, done in zip(self.rs[::-1], self.vs[::-1], self.dones[:0:-1]):
            R = r + self.gamma * R * (1.-done)
            Adv = R - v
            Rs.append(R)
            Advs.append(Adv)
        Rs.reverse()
        Advs.reverse()
        self.Rs = Rs
        self.Advs = Advs

    def sample_transition(self, R, discrete=True):
        self._add_R_Adv(R)
        obs = np.array(self.obs, dtype=np.float32)
        if discrete:
            acts = np.array(self.acts, dtype=np.int32)
        else:
            acts = np.array(self.acts, dtype=np.float32)
        Rs = np.array(self.Rs, dtype=np.float32)
        Advs = np.array(self.Advs, dtype=np.float32)
        # use pre-step dones here
        dones = np.array(self.dones[:-1], dtype=np.bool)
        self.reset(self.dones[-1])
        return obs, acts, dones, Rs, Advs
