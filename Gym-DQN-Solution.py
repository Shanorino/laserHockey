import gym
from gym import spaces
import numpy as np
import laser_hockey_env as lh
from importlib import reload
import time

import matplotlib.pyplot as plt
import tensorflow as tf


reload(lh)
env = lh.LaserHockeyEnv(mode=1)
#env = gym.make(env_name)
ac_space = env.action_space
o_space = env.observation_space
print(ac_space)
print(o_space)
print(zip(env.observation_space.low, env.observation_space.high))


import numpy as np

class Memory():
    # class to store x/u trajectory
    def __init__(self, buffer_shapes, buffer_size=int(1e5)):
        self._buffer_size = buffer_size
        self._buffer_shapes = buffer_shapes
        self._data = {key: np.empty((self._buffer_size, value) if value is not None else (self._buffer_size,)) 
                      for key, value in self._buffer_shapes.items()}
        self._current_size = 0
        self._t = 0

    def add_item(self, new_data): 
        for key in self._data.keys():
            self._data[key][self._t%self._buffer_size] = new_data[key]
        self._t += 1
        self._current_size = min(self._t, self._buffer_size)
      
    def sample(self, batch_size=1):
        if batch_size > self._current_size:
            batch_size = self._current_size
        inds = np.random.choice(range(self._current_size), size=batch_size, replace=False)
        batch = {key: value[inds] for key, value in self._data.items()}
        return batch




# Q network (also the critic network in DDPG)
class QFunction:
    def __init__(self, o_space, a_space, gamma=0.99, scope='Q'):
        self._scope = scope
        self._o_space = o_space
        self._a_space = a_space
        self._action_n = 7 #self._a_space.n
        self._gamma = gamma
        self._sess = tf.get_default_session() or tf.InteractiveSession()
    
        with tf.variable_scope(self._scope, reuse=tf.AUTO_REUSE):
            self._build_graph()
    
    def _build_graph(self):
        self.state = tf.placeholder(dtype=tf.float32, shape=(None, self._o_space.shape[0]), name="state")
        
        dense1 = tf.layers.dense(inputs=self.state, units=100, activation=tf.nn.leaky_relu, name='dense1')
        dense2 = tf.layers.dense(inputs=dense1, units=100, activation=tf.nn.leaky_relu, name='dense2')
        self.h = tf.layers.dense(inputs=dense2, units=7, activation=None, name='h') #nx7
        self.output = tf.squeeze(self.h, name='pred')
        
        #tf.add_to_collection('pred_Qs', self.output, scope=self._scope)
        #tf.add_to_collection('state', self.state, scope=scope) #however not used in the documents' examples
        
    
    def Qs(self, state):
        _state = np.asarray(state).reshape(-1, self._o_space.shape[0])
        inp = {self.state: _state}
        return self._sess.run(self.output, feed_dict=inp).reshape(-1, self._action_n)
    
    def Q(self, state, action):
        _action = (action,) if type(action) == int else action
        return self.Qs(state)[(tuple(range(len(_action))), _action)]
    
    def V(self, state):
        V = np.max(self.Qs(state), axis=1)
        return np.squeeze(V)
    
    def save(self, path='./DQN_model.ckpt'):
        saver = tf.train.Saver()
        
        saver.save(self._sess, path)
        print ('model saved under the path: ', path)

# In[2]: define Agents 


class DQNAgent(object):
    """
    Agent implementing Q-learning with NN function approximation.
    """
    def __init__(self, o_space, a_space, scope='DQNAgent', **userconfig):
        
        self._o_space = o_space
        self._a_space = a_space
        self._action_n = 7 #a_space.n
        self._config = {
            "eps_begin": 0.3,            # Epsilon in epsilon greedy policies
            "eps_end": 0.05,
            "eps_decay": 0.99,
            "discount": 0.95,
            "buffer_size": int(5e5),
            "batch_size": 32,
            "learning_rate": 1e-4,
            "theta": 0.05,
            "use_target_net": True,}
        self._config.update(userconfig)
        self._scope = scope
        self._eps = self._config['eps_begin']
        self._buffer_shapes = {
            's': self._o_space.shape[0],
            'a': None,
            'r': None,
            's_prime': self._o_space.shape[0],
            'd': None,
        }
        self._buffer = Memory(buffer_shapes=self._buffer_shapes, buffer_size=self._config['buffer_size'])
        self._sess = tf.get_default_session() or tf.InteractiveSession()
        
        # Create Q Networks
        with tf.variable_scope(self._scope, reuse=tf.AUTO_REUSE):
            self._Q = QFunction(scope='Q', o_space=self._o_space, a_space=self._a_space, 
                                         gamma=self._config['discount'])
            if self._config['use_target_net']:
                self._Q_target = QFunction(scope='Q_target', o_space=self._o_space, a_space=self._a_space, 
                                             gamma=self._config['discount'])
                
            self._prep_train()
            
        self._sess.run(tf.global_variables_initializer())
        
            
    def _vars(self, scope=''):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self._scope + '/' + scope)
    
    def _global_vars(self, scope=''):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self._scope + '/' + scope)
            
    def _prep_train(self):
        self._action = tf.placeholder(dtype=tf.int32, shape=(None,), name="action") #??
        self._action_onehot = tf.one_hot(self._action, self._action_n, dtype=tf.float32)
        self._Qval = tf.reduce_sum(tf.multiply(self._Q.output, self._action_onehot), axis=1)
        self._target = tf.placeholder(dtype=tf.float32, shape=(None,), name="target")
        self._loss = tf.reduce_mean(tf.square(self._Qval - self._target))
        self._optim = tf.train.AdamOptimizer(learning_rate=self._config['learning_rate'])
        self._train_op = self._optim.minimize(self._loss)
        
        if self._config['use_target_net']:
            self._update_target_op = [tf.assign(
                target_var, 
                (1-self._config['theta'])*target_var+self._config['theta']*Q_var) 
                                     for target_var, Q_var in zip(self._vars('Q_target'), self._vars('Q'))]
        
    def _update_target_net(self):
        self._sess.run(self._update_target_op)

    def act(self, observation, eps=None):
        if eps is None:
            eps = self._eps
        # epsilon greedy.
        if np.random.random() > eps:
            action = np.argmax(self._Q.Qs(observation)) 
            #action = env.discrete_to_continous_action(action)
            #action = np.hstack([action, [0,0.,0]])
        else: 
            action = np.random.randint(0, 6)
            #action = self._a_space.sample()        
        return action
    
    def store_transition(self, transition):
        self._buffer.add_item(transition)
        
    def _eps_scheduler(self, writer=None, ep=0):
        self._eps = max(self._config['eps_end'], self._eps*self._config['eps_decay'])
        if writer: writer.add_scalar('policy/eps', self._eps, ep)
    
    def train(self, iter_fit=4, writer=None, ep=0):
        losses = []
        for i in range(iter_fit):

            # sample from the replay buffer
            data = self._buffer.sample(batch_size=self._config['batch_size'])
            
            s = data['s'] # s_t
            a = data['a'] # a_t
            r = data['r'] # rew
            s_prime = data['s_prime'] # s_t+1
            d = data['d'] # done
            if self._config['use_target_net']:
                v_prime = self._Q_target.V(s_prime)
            else:
                v_prime = self._Q.V(s_prime)    

            # target
            td_target = r + self._config['discount'] * v_prime * (1-d)

            # optimize the lsq objective
            inp = {self._Q.state: s, self._action: a, self._target: td_target}
            fit_loss = self._sess.run([self._train_op, self._loss], feed_dict=inp)[1]
            losses.append(fit_loss)
            
        if self._config['use_target_net']:
            self._update_target_net()
            
        return losses
    
    


# In[4]: Initializing training parameters


fps = 100 # env.metadata.get('video.frames_per_second')
max_steps = 80 #env.spec.tags['wrapper_config.TimeLimit.max_episode_steps']
#max_steps


# In[4]: Start training


q_agent = DQNAgent(o_space, ac_space, discount=0.99, eps_begin=0.3)




stats = []
losses = []


writer=None

max_episodes=2000
#mode="random"
show=False
mode="Q"
for i in range(max_episodes):
    # print("Starting a new episode")    
    total_reward = 0
    ob = env.reset()
    max_height = -np.inf
    for t in range(max_steps):
        done = False
        if mode == "random":
            action = np.random.randint(0, 6)
            a = env.discrete_to_continous_action(action)
            a = np.hstack([a, [0,0.,0]])
            #print ('action vector(random): ', a)
            #a = ac_space.sample()                        
        elif mode == "Q":
            action = q_agent.act(ob)
            a = env.discrete_to_continous_action(action)
            a = np.hstack([a, [0,0.,0]])
            #print ('action vector(Q): ', a)
        else:
            raise ValueError("no implemented")
        (ob_new, reward, done, _info) = env.step(a)
        total_reward+= reward
        if mode == "Q":
            #print ({'s': ob, 'a': a, 'r': reward, 's_prime': ob_new, 'd': done})
            q_agent.store_transition({'s': ob, 'a': action, 'r': reward, 's_prime': ob_new, 'd': done})            
        ob=ob_new        
        if show:
            time.sleep(1.0/fps)
            env.render(mode='human')
        loss = q_agent.train(1, writer=writer, ep=i)
        losses.extend(loss)
        #max_height = max(max_height, ob[0])
        if done: break    
    stats.append([i,total_reward,t+1])
    q_agent._eps_scheduler(writer=writer, ep=i)

    print ('episode ', i, 'reward: ', total_reward, 'loss: ', loss)
    #if ((i-1)%50==0):
    #    print("Done after {} episodes. Reward: {}".format(i, total_reward))

# save the model
q_agent._Q_target.save()
# save plot
plt.plot(np.asarray(stats)[:,0], np.asarray(stats)[:,1])
plt.savefig("DQN_loss.png")
plt.close()
env.close()

