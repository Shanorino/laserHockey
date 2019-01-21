import gym
from gym import spaces
import numpy as np
import laser_hockey_env as lh
from importlib import reload
import time

import matplotlib.pyplot as plt
import tensorflow as tf

#tf.reset_default_graph()
reload(lh)
env = lh.LaserHockeyEnv(mode=1)
ac_space = env.action_space
o_space = env.observation_space
#print(ac_space)
#print(o_space)
#print(zip(env.observation_space.low, env.observation_space.high))


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


class DDPGFunction:
    def __init__(self, o_space, a_space, gamma=0.99, scope='', mode='Q'):
        self._scope = scope
        self._o_space = o_space
        self._a_space = a_space
        self._action_n = 3 #self._a_space.n
        self._gamma = gamma
        self._sess = tf.get_default_session() or tf.InteractiveSession()
        
        if mode=='Q':
            with tf.variable_scope(self._scope, reuse=tf.AUTO_REUSE): #for the 2 critic networks
                self._build_graph_q()
        else:
            with tf.variable_scope(self._scope, reuse=tf.AUTO_REUSE): #for the 2 actor networks
                self._build_graph_mu()
                
        
        
    def _build_graph_q(self):  
        self.state = tf.placeholder(dtype=tf.float32, shape=(None, self._o_space.shape[0]), name="state") #nx16
        self.action = tf.placeholder(dtype=tf.float32, shape=(None, self._action_n), name="action") #nx3
#        dense1 = tf.layers.dense(inputs=self.state, units=100, activation=tf.nn.relu, name='l_state')  #nx100
#        dense2 = tf.layers.dense(inputs=self.action, units=100, activation=tf.nn.relu, name='l_action') #nx100
#        
#        sa = tf.add(dense1,dense2) #nx100
#        self.h = tf.layers.dense(inputs=sa, units=1, activation=None, name='h') #nx1
#        self.output = tf.squeeze(self.h, name='predQ')
        
        init_w = tf.random_normal_initializer(0., 0.1)
        init_b = tf.constant_initializer(0.1)
        w1_s = tf.get_variable('w1_s', [self._o_space.shape[0], 100], trainable=1)
        w1_a = tf.get_variable('w1_a', [self._action_n, 100], trainable=1)
        b1 = tf.get_variable('b1', [1, 100], trainable=1)
        net = tf.nn.leaky_relu(tf.matmul(self.state, w1_s) + tf.matmul(self.action, w1_a) + b1)
        self.output = tf.layers.dense(net, 1, activation=None, trainable=1, name='predQ')
        #self.output = tf.squeeze(self.output, name='predQ')
        
        
        
    def _build_graph_mu(self):
        self.state = tf.placeholder(dtype=tf.float32, shape=(None, self._o_space.shape[0]), name="state")
        dense1 = tf.layers.dense(inputs=self.state, units=100, activation=tf.nn.leaky_relu, name='l1') #nx100
        #dense2 = tf.layers.dense(inputs=dense1, units=100, activation=tf.nn.leaky_relu, name='l2')  #nx100
        self.output = tf.layers.dense(inputs=dense1, units=self._action_n, activation=tf.nn.tanh, name='predMu') #nx3
        #self.output = tf.squeeze(self.output, name='predMu') # nx3
        
    # for actor's policy gradient
    def add_grads(self, e_params, a_grads): 
        # define d_mu for the critic loss
        self.policy_grads = tf.gradients(ys=self.output, xs=e_params, grad_ys=a_grads)  #ys=mu, xs=theta_mu, grad_ys=dQ/da
        
    def As(self, state):  # only for actors
        _state = np.asarray(state).reshape(-1, self._o_space.shape[0])
        inp = {self.state: _state}
        return self._sess.run(self.output, feed_dict=inp).reshape(-1, self._action_n)
    
    def Qs(self, state, action): # only for critics
        _state = np.asarray(state).reshape(-1, self._o_space.shape[0])
        _action = np.asarray(action).reshape(-1, self._action_n)
        inp = {self.state: _state, self.action: _action}
        return self._sess.run(self.output, feed_dict=inp).reshape(-1, 1)
        
    def save(self, path='./DDPG_model.ckpt'):
        saver = tf.train.Saver()
        saver.save(self._sess, path)
        print ('model saved under the path: ', path)
            
            


# In[3] DDPG-Agent
class DDPGAgent(object):
    """
    Agent implementing Q-learning with NN function approximation.
    """
    def __init__(self, o_space, a_space, scope='DDPGAgent', **userconfig):
        
        self._o_space = o_space
        self._a_space = a_space
        self._action_n = 3 #a_space.n
        self._config = {
            #"eps_begin": 0.3,            # Epsilon in epsilon greedy policies
            #"eps_end": 0.05,
            #"eps_decay": 0.99,
            "discount": 0.95,
            "buffer_size": int(5e5),
            "batch_size": 32,
            "learning_rate_a": 1e-4,
            "learning_rate_c": 1e-4,
            "theta": 0.05, #soft update rate
            "use_target_net": True,}
        self._config.update(userconfig)
        self._scope = scope
        self._buffer_shapes = {
            's': self._o_space.shape[0],
            'a': self._action_n,
            'r': None,
            's_prime': self._o_space.shape[0],
            'd': None,
        }
        self._buffer = Memory(buffer_shapes=self._buffer_shapes, buffer_size=self._config['buffer_size'])
        self._sess = tf.get_default_session() or tf.InteractiveSession()
        
        # Create Q Networks and Mu Networks
        with tf.variable_scope(self._scope, reuse=tf.AUTO_REUSE):
            self._Q = DDPGFunction(scope='Q_eval', o_space=self._o_space, a_space=self._a_space, 
                                         gamma=self._config['discount'], mode='Q')
            self._Q_target = DDPGFunction(scope='Q_target', o_space=self._o_space, a_space=self._a_space, 
                                             gamma=self._config['discount'], mode='Q')
            self._Mu = DDPGFunction(scope='Mu_eval', o_space=self._o_space, a_space=self._a_space, 
                                         gamma=self._config['discount'], mode='Mu')
            self._Mu_target = DDPGFunction(scope='Mu_target', o_space=self._o_space, a_space=self._a_space, 
                                         gamma=self._config['discount'], mode='Mu')
            
            self._prep_train()
        
        self._sess.run(tf.global_variables_initializer())
        
        self.summary_writer = tf.summary.FileWriter("log", self._sess.graph)
        
    def _vars(self, scope=''):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self._scope + '/' + scope)
    
    def _global_vars(self, scope=''):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self._scope + '/' + scope)
    
    def store_transition(self, transition):
        self._buffer.add_item(transition)
        
    # define the loss functions here
    def _prep_train(self):
        #self._Qval = self._Q.output #nx1
        
        # CRITIC optimizer
        self._target = tf.placeholder(dtype=tf.float32, shape=(None, 1), name="target") #td target
        self._c_loss = tf.reduce_mean(tf.squared_difference(self._target, self._Q.output))  # dimension correct?
        self._c_optim = tf.train.AdamOptimizer(learning_rate=self._config['learning_rate_c'])
        self._c_train_op = self._c_optim.minimize(self._c_loss)
        
        # ACTOR policy gradient for actor Mu
        a_grads = tf.gradients(self._Q.output, self._Q.action)[0]  #dQ/da; nx3
        self._Mu.add_grads(self._vars('Mu_eval'), a_grads)
        self._a_optim = tf.train.AdamOptimizer(learning_rate=-self._config['learning_rate_a']) 
        self._a_train_op = self._a_optim.apply_gradients(zip(self._Mu.policy_grads, self._vars('Mu_eval')))
        
        # update operations
        self._q_update_target_op = [tf.assign(
                target_var, 
                (1-self._config['theta'])*target_var + self._config['theta']*eval_var) 
                                     for target_var, eval_var in zip(self._vars('Q_target'), self._vars('Q_eval'))]
        self._mu_update_target_op = [tf.assign(
                target_var, 
                (1-self._config['theta'])*target_var + self._config['theta']*eval_var) 
                                     for target_var, eval_var in zip(self._vars('Mu_target'), self._vars('Mu_eval'))]
            
    def train(self, iter_fit=2, writer=None, ep=0):
        losses = []
        for i in range(iter_fit):

            # sample from the replay buffer
            data = self._buffer.sample(batch_size=self._config['batch_size'])
            
            s = data['s'] # s_t
            a = data['a'] # a_t  ###############################ATTENTION: NOW nx3 instead of nx1
            r = data['r'] # rew
            s_prime = data['s_prime'] # s_t+1
            d = data['d'] # done
            
            As = self._Mu_target.As(s_prime) # + N ?
            y_i = r.reshape(-1, 1) + self._config['discount'] * self._Q_target.Qs(s_prime, As) # * d
            

            # optimize the lsq objective
            # critic
            inp = {self._Q.state: s, self._Q.action: a, self._target: y_i}
            c_loss = self._sess.run([self._c_train_op, self._c_loss], feed_dict=inp)[1]
            
            # actor
            inp = {self._Q.state: s, self._Q.action: a, self._Mu.state: s}
            self._sess.run(self._a_train_op, feed_dict=inp)
#            a_loss = self._sess.run([self._a_train_op, self._a_loss], feed_dict=inp)[1]
#            losses.extend([c_loss, a_loss])
            losses.append(c_loss)
            
            # update target nets
        if self._config['use_target_net']:
            self._sess.run(self._q_update_target_op)
            self._sess.run(self._mu_update_target_op)
            
        return losses
    
    def act(self, observation, eps=None):
#        if eps is None:
#            eps = self._eps
#        # epsilon greedy.
#        if np.random.random() > eps:
#            action = np.argmax(self._Q.Qs(observation)) 
#            #action = env.discrete_to_continous_action(action)
#            #action = np.hstack([action, [0,0.,0]])
#        else: 
#            action = np.random.randint(0, 6)
#            #action = self._a_space.sample()        
        return np.squeeze(self._Mu.As(observation))
    
# In[4]: Initializing training parameters


fps = 100 # env.metadata.get('video.frames_per_second')
max_steps = 80 #env.spec.tags['wrapper_config.TimeLimit.max_episode_steps']
n=10 #training frequency (train once in n episodes)
update_f=1 #update frequency (update once in f trainings)
max_episodes=5000

# In[4]: Start training
ddpg_agent = DDPGAgent(o_space, ac_space, discount=0.99)

## test the outputs
#ob = env.reset()
#ac_output = ddpg_agent._Mu.As(ob)
#q_output = ddpg_agent._Q.Qs(ob, ac_output)

# start training
stats = []
losses = []
rewards = []

writer=None

show=False
mode="DDPG"

for i in range(max_episodes):
    # print("Starting a new episode")    
    total_reward = 0
    ob = env.reset()
    for t in range(max_steps):
        done = False
        action = ddpg_agent.act(ob)
        # adding noise to action
        a_t = np.clip(np.random.normal(action, 0.2), -1, 1)
        # opponent does total random actions
        a_opp = np.clip(np.random.normal([0, 0, 0], 0.5), -1, 1)
        
        a = np.hstack([a_t, a_opp])
        (ob_new, reward, done, _info) = env.step(a)
        total_reward+= reward

        ddpg_agent.store_transition({'s': ob, 'a': a_t, 'r': reward, 's_prime': ob_new, 'd': done})            
        ob=ob_new        
        if show:
            time.sleep(1.0/fps)
            env.render(mode='human')
        # training frequency n; update frequency update_f;
        if t%n==0:
            loss = ddpg_agent.train(update_f, writer=writer, ep=i)
        losses.extend(loss)
        if done: break    
    stats.append([i,total_reward,t+1])
    #q_agent._eps_scheduler(writer=writer, ep=i)
    if i%1==0: #print frequency
        print ('episode ', i, 'reward: ', total_reward, 'critic loss: ', loss)

# save the model
ddpg_agent._Mu_target.save()
# save plot
plt.plot(np.asarray(stats)[:,0], np.asarray(stats)[:,1])
plt.savefig("DDPG_loss.png")
plt.close()
env.close()

