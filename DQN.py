import numpy as np
import pandas as pd
import tensorflow as tf


MAZE_H = 4  # grid height
MAZE_W = 4  # grid width

class DeepQNetwork:
    def __init__(
            self,
            n_actions,
            n_features,
            learning_rate=0.001,
            reward_decay=0.9,
            exploration = 1,
            min_exploration = 0.01,
            replace_target_iter=300,
            memory_size=500,
            batch_size=128,
            exploration_decay = 0.99,
            
    ):
        self.n_actions = n_actions
        self.n_features = n_features
        self.learning_rate = learning_rate
        self.gamma = reward_decay
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.exploration_decay = exploration_decay
        self.exploration = exploration 
        self.min_exploration = min_exploration 
        self.memory_counter = 0
        self.memory_full = False

        
        # total learning step
        self.learn_step_counter = 0

        # initialize zero memory [s, a, r, s_, terminal]
        self.memory = np.zeros((self.memory_size, n_features * 2 + 3))

        # consist of [target_net, evaluate_net]
        self._build_net()
        t_params = tf.get_collection('target_net_params')
        e_params = tf.get_collection('Q_net_params')
        self.replace_target_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]

        self.sess = tf.Session()

        self.sess.run(tf.global_variables_initializer())
        self.cost_hist = []

    def _build_net(self):
        # ------------------ build Q_net ------------------
        # input
        self.state = tf.placeholder(tf.float32, [None, self.n_features], name='state') 
        
        # for calculating loss
        self.q_target = tf.placeholder(tf.float32, [None, self.n_actions], name='Q_target') 
        
        
        with tf.variable_scope('Q_net'):
            # weight initialize
            c_names, n_l1, w_initializer, b_initializer = \
                ['Q_net_params', tf.GraphKeys.GLOBAL_VARIABLES], 32, \
                tf.random_normal_initializer(0, 0.9), tf.constant_initializer(0.1)  

            # first layer
            with tf.variable_scope('l1'):
                w1 = tf.get_variable('w1', [self.n_features, n_l1], initializer=w_initializer, collections=c_names)
                b1 = tf.get_variable('b1', [1, n_l1], initializer=b_initializer, collections=c_names)
                l1 = tf.nn.relu(tf.matmul(self.state, w1) + b1)

            # second layer
            with tf.variable_scope('l2'):
                w2 = tf.get_variable('w2', [n_l1, self.n_actions], initializer=w_initializer, collections=c_names)
                b2 = tf.get_variable('b2', [1, self.n_actions], initializer=b_initializer, collections=c_names)
                self.q_state = tf.matmul(l1, w2) + b2

        # loss
        with tf.variable_scope('loss'):
            self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_state))
        
        # train
        with tf.variable_scope('train'):
            self._train_op = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.loss)

            
        # ------------------ build target_net (should be same as Q net) ------------------
        # input
        self.next_state = tf.placeholder(tf.float32, [None, self.n_features], name='next_state')    
        with tf.variable_scope('target_net'):
            c_names = ['target_net_params', tf.GraphKeys.GLOBAL_VARIABLES]

            # first layer
            with tf.variable_scope('l1'):
                w1 = tf.get_variable('w1', [self.n_features, n_l1], initializer=w_initializer, collections=c_names)
                b1 = tf.get_variable('b1', [1, n_l1], initializer=b_initializer, collections=c_names)
                l1 = tf.nn.relu(tf.matmul(self.next_state, w1) + b1)

            # second layer
            with tf.variable_scope('l2'):
                w2 = tf.get_variable('w2', [n_l1, self.n_actions], initializer=w_initializer, collections=c_names)
                b2 = tf.get_variable('b2', [1, self.n_actions], initializer=b_initializer, collections=c_names)
                self.q_next_state = tf.matmul(l1, w2) + b2

    # Store transition
    def store_transition(self, state, action, reward, next_state , terminal):

        transition = np.hstack((state, [[action, reward , terminal]], next_state,))

        # replace the old memory with new memory in circular way (queue)
        if self.memory_counter == self.memory_size-1 and not self.memory_full:
            self.memory_full = True
        self.memory_counter = (self.memory_counter+1) % self.memory_size
        self.memory[self.memory_counter, :] = transition

    # choose action given observation
    def choose_action(self, observation):
        
        # follow the policy
        if np.random.uniform() > self.exploration:
            #  get q value for every actions
            actions_value = self.sess.run(self.q_state, feed_dict={self.state: observation})
            # choose the action with maximum q value
            action = np.argmax(actions_value)  
        else: # explore the world
            action = np.random.randint(0, self.n_actions)
        return action

    # update parameter in q network
    def learn(self):
        
        # sample batch memory from all memory
        if self.memory_full:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
            
        
        batch_memory = self.memory[sample_index, :]

        # get the q value of current state and next state
        q_next_state, q_state = self.sess.run(
            [self.q_next_state, self.q_state],
            feed_dict={
                self.next_state: batch_memory[:, -self.n_features:],  # fixed params
                self.state: batch_memory[:, :self.n_features],  # newest params
            })

        # [state, action, reward , terminal , next_state] 
        batch_index = np.arange(self.batch_size, dtype=np.int32)
        action_index = batch_memory[:, self.n_features].astype(int)
        reward = batch_memory[:, self.n_features + 1]
        terminal = batch_memory[:, self.n_features + 2]
        
        # define the target Q value = reward + gamma*max(Q(next state))
        q_target = q_state.copy()
        for i in range(self.batch_size):
            if terminal[i]:
                q_target[i, action_index] = reward
            else:
                q_target[i, action_index] = reward + self.gamma * np.max(q_next_state, axis=1)
                
        # train Q network to minimize the difference Q target and estimated Q value 
        _, self.cost = self.sess.run([self._train_op, self.loss],
                                     feed_dict={self.state: batch_memory[:, :self.n_features],
                                                self.q_target: q_target})
        
        self.cost_hist.append(self.cost)
        
        
        # replace the target network
        if self.learn_step_counter == self.replace_target_iter:
            # decreasing exploration rate
            self.learn_step_counter = 0
            if self.exploration*self.exploration_decay > self.min_exploration :
                self.exploration = self.exploration*self.exploration_decay
            self.sess.run(self.replace_target_op)
            print('\nReplace target net parameter\n')
        
        self.learn_step_counter += 1

        
    # test the preffered action given the state
    def test(self):
        file = open("Policy.txt","w")
        state = np.zeros([1, MAZE_H*MAZE_W])
        for i in range(16):
                state[0,i] = 1
                actions_value = self.sess.run(self.q_next_state, feed_dict={self.next_state: state})
                #print(actions_value)
                action = np.argmax(actions_value)
                file.write(str(action)) 
                state[0,i] = 0
                if (i+1)%4 == 0:
                    file.write('\r\n')
        file.close() 
        