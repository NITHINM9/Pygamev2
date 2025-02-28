from keras.layers import Dense, Activation, Dropout
from keras.models import Sequential, load_model
from keras.optimizers import Adam 
from keras.losses import Huber
import numpy as np
import tensorflow as tf
import os
import pickle

class ReplayBuffer(object):
    def __init__(self, max_size, input_shape, n_actions, discrete=False):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.discrete = discrete
        self.state_memory = np.zeros((self.mem_size, input_shape))
        self.new_state_memory = np.zeros((self.mem_size, input_shape))
        dtype = np.int8 if self.discrete else np.float32
        self.action_memory = np.zeros((self.mem_size, n_actions), dtype=dtype)
        self.reward_memory = np.zeros(self.mem_size)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.float32)

    def store_transition(self, state, action, reward, state_, done):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        if self.discrete:
            actions = np.zeros(self.action_memory.shape[1])
            actions[action] = 1.0
            self.action_memory[index] = actions
        else:
            self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = 1 - done
        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size)

        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        states_ = self.new_state_memory[batch]
        terminal = self.terminal_memory[batch]

        return states, actions, rewards, states_, terminal

class DDQNAgent(object):
    def __init__(self, alpha, gamma, n_actions, epsilon, batch_size,
                 input_dims, epsilon_dec=0.999,  epsilon_end=0.10,
                 mem_size=50000, fname='ddqn_model.h5', replace_target=500):
        self.action_space = [i for i in range(n_actions)]
        self.n_actions = n_actions
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_dec = epsilon_dec
        self.epsilon_min = epsilon_end
        self.batch_size = batch_size
        self.model_file = fname
        self.replace_target = replace_target
        self.memory = ReplayBuffer(mem_size, input_dims, n_actions, discrete=True)
        
        self.brain_eval = Brain(input_dims, n_actions, batch_size)
        self.brain_target = Brain(input_dims, n_actions, batch_size)
        
        self.episode = 0

    def remember(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

    def choose_action(self, state):
        state = np.array(state)
        state = state[np.newaxis, :]

        rand = np.random.random()
        if rand < self.epsilon:
            action = np.random.choice(self.action_space)
        else:
            actions = self.brain_eval.predict(state)
            action = np.argmax(actions)

        return action

    def learn(self):
        if self.memory.mem_cntr > self.batch_size:
            state, action, reward, new_state, done = self.memory.sample_buffer(self.batch_size)

            action_values = np.array(self.action_space, dtype=np.int8)
            action_indices = np.dot(action, action_values)

            q_next = self.brain_target.predict(new_state)
            q_eval = self.brain_eval.predict(new_state)
            q_pred = self.brain_eval.predict(state)

            max_actions = np.argmax(q_eval, axis=1)

            q_target = q_pred

            batch_index = np.arange(self.batch_size, dtype=np.int32)

            q_target[batch_index, action_indices] = reward + self.gamma*q_next[batch_index, max_actions.astype(int)]*done

            _ = self.brain_eval.train(state, q_target)

            self.epsilon = self.epsilon*self.epsilon_dec if self.epsilon > self.epsilon_min else self.epsilon_min

    def update_network_parameters(self):
        self.brain_target.copy_weights(self.brain_eval)

    def save_model(self):
        if not hasattr(self.brain_eval, 'model'):
            print("Model not initialized. Creating model before saving...")
            self.brain_eval.model = self.brain_eval.createModel()

        self.brain_eval.model.save(self.model_file)
        print(f"Model saved to {self.model_file}")

        with open('training_state.pkl', 'wb') as f:
            pickle.dump({
                'epsilon': self.epsilon,
                'memory': self.memory,
                'episode': self.episode
            }, f)
        print("Training state saved to 'training_state.pkl'")

    def load_model(self):
        if os.path.exists(self.model_file):
            self.brain_eval.model = load_model(self.model_file)
            print(f"Model loaded from {self.model_file}")

            if os.path.exists('training_state.pkl'):
                with open('training_state.pkl', 'rb') as f:
                    state = pickle.load(f)
                    self.epsilon = state['epsilon']
                    self.memory = state['memory']
                    self.episode = state['episode']
                print("Training state loaded from 'training_state.pkl'")
            else:
                print("No training state found. Starting with default values.")
        else:
            print("No existing model found. Creating a new one...")
            self.brain_eval.model = self.brain_eval.createModel()

class Brain:
    def __init__(self, NbrStates, NbrActions, batch_size=256):
        self.NbrStates = NbrStates
        self.NbrActions = NbrActions
        self.batch_size = batch_size
        self.learning_rate = 0.0005
        self.state_size = NbrStates
        self.action_size = NbrActions
        self.model = self.createModel()

    def createModel(self):
        model = Sequential([
            Dense(256, input_shape=(self.state_size,), activation='relu'),
            Dropout(0.2),
            Dense(256, activation='relu'),
            Dropout(0.2),
            Dense(128, activation='relu'),
            Dense(self.action_size, activation='linear')
        ])

        model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss=Huber()
        )

        return model

    def train(self, x, y, epoch = 1, verbose = 0):
        self.model.fit(x, y, batch_size = self.batch_size , verbose = verbose)

    def predict(self, s):
        return self.model.predict(s)

    def predictOne(self, s):
        return self.model.predict(tf.reshape(s, [1, self.NbrStates])).flatten()

    def copy_weights(self, TrainNet):
        variables1 = self.model.trainable_variables
        variables2 = TrainNet.model.trainable_variables
        for v1, v2 in zip(variables1, variables2):
            v1.assign(v2.numpy())
