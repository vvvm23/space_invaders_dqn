import random
from collections import deque

import cv2
import numpy as np
import matplotlib.pyplot as plt

import gym
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense, Conv2D, Dropout, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

class DQN:
    def __init__(self, state_size, action_size, model_depth=4):
        self.state_size = state_size
        self.action_size = action_size
        self.model_depth = model_depth
        self.memory = deque(maxlen=4000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.batch_size = 64
        self.model = self._create_model()

    def _create_model(self):
        model = Sequential()
        model.add(Conv2D(16, (3,3), input_shape=(self.state_size[0], self.state_size[1], self.model_depth), activation='relu'))
        model.add(Dropout(0.25))

        model.add(Conv2D(32, (3,3), activation='relu'))
        model.add(Dropout(0.25))

        model.add(Flatten())
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    def add_memory(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        pred = self.model.predict(np.expand_dims(state, axis=0))
        return np.argmax(pred[0])

    def replay(self):
        batch = random.sample(self.memory, self.batch_size)
        states, targets_f = [], []
        for state, action, reward, next_state, done in batch: # This unpacking is clever!
            target = reward
            if not done:
                # TODO: can we predict in batch too? <19-03-20, alex> #
                target += self.gamma * np.amax(self.model.predict(np.expand_dims(next_state, axis=0)))

            target_f = self.model.predict(np.expand_dims(state, axis=0))
            target_f[0][action] = target
            states.append(state)
            targets_f.append(target_f[0])

        # verbose = 0, as we call this a lot. Don't want loading bar spam
        history = self.model.fit(np.array(states), np.array(targets_f), epochs=1, verbose = 0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return history

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        # TODO: save model instead? <19-03-20, alex> #
        self.model.save_weights(name)

def preprocess(obv):
    obv = cv2.cvtColor(cv2.resize(obv, (84, 110)), cv2.COLOR_BGR2GRAY)
    obv = obv[26:, :]
    ret, obv = cv2.threshold(obv, 1, 255, cv2.THRESH_BINARY)
    return np.reshape(obv, (84, 84, 1))

def short_to_state(short):
    return np.reshape(np.array(short), (84, 84, 4))

if __name__ == '__main__':
    NB_SHORT_MEM = 4
    NB_GAMES = 1000
    NB_STEPS = 500

    env = gym.make('SpaceInvaders-v0')
    # state_size = env.observation_space.shape[0:2]
    state_size = (84, 84)
    action_size = env.action_space.n
    agent = DQN(state_size, action_size, NB_SHORT_MEM)

    for g in range(NB_GAMES):
        obv = preprocess(env.reset())
        short_mem = deque([obv]*NB_SHORT_MEM, maxlen=NB_SHORT_MEM)
        state = short_to_state(short_mem)

        for t in range(NB_STEPS):
            if t % 10 == 0:
                print(f"Game {g}: Time - {t}")
            action = agent.act(state)
            obv, reward, done, _ = env.step(action)
            obv = preprocess(obv)
            reward = reward if not done else -10

            short_mem.append(obv)
            next_state = short_to_state(short_mem)
            agent.add_memory(state, action, reward, next_state, done)
            state = next_state
            if done:
                print(f"Game {g}: Time - {t}\tReward: {reward}")
                break
            # TODO: record memory size <19-03-20, alex> #
            if len(agent.memory) > agent.batch_size:
                history = agent.replay()

        if g % 10 == 0:
            plt.plot(history.history['loss'])
            plt.title("SpaceInvaders DQN Loss")
            plt.ylabel("Loss")
            plt.xlabel("Game")
            plt.savefig(f"./space-invaders-loss-graph-{g}.png")
            agent.save(f"./space-invaders-dqn-{g}.h5")
    
    agent.save(f"./space-invaders-dqn-final.h5")
