import numpy as np
import time
import os

class LavaPitRoom:
    def __init__(self):
        self.map = [
            ['o', 'l', 'o', 'l', 'l'],
            ['o', 'o', 'o', 'o', 'o'],
            ['l', 'l', 'l', 'o', 'l'],
            ['o', 'l', 'l', 'o', 'o'],
            ['o', 'o', 'o', 'o', 'g']
            ]
        self.state = [0, 0]
        self.size = (len(self.map), len(self.map[0]))
        self.goal = [4, 4]
        self.actions = {0:"Up", 1:"Down", 2:"Left", 3:"Right"}
        self.qtable = np.zeros((5, 5, 4))
        self.learning_rate = 0.1
        self.discount_factor = 0.9
        self.exploration_rate = 1
        self.min_exploration_rate = 0.01
        self.decay_rate = 0.990

    def render(self):
        X, Y = self.size
        os.system('cls')
        for row in range(X):
            for col in range(Y):
                if [row, col] == self.state: print('A', end=' ')
                else: print(self.map[row][col], end=' ')
            print()

    def getState(self, action):
        row, col = self.state
        X, Y = self.size
        if action == "Up" and row > 0:
            row -= 1
        elif action == "Down" and row < X-1:
            row += 1
        elif action == "Left" and col > 0:
            col -= 1
        elif action == "Right" and col < Y-1:
            col += 1
        return [row, col]
    
    def getReward(self):
        if self.state == self.goal:
            return 1000
        elif self.map[self.state[0]][self.state[1]] == 'l':
            return -100
        else:
            return -1
    
    def train(self, epochs):
        maxSteps = self.size[0]*self.size[1]
        for epoch in range(epochs):
            self.state = [0, 0]
            done = False
            steps = 0
            while not done:
                row, col = self.state
                if np.random.rand() < self.exploration_rate:
                    action = np.random.choice([0, 1, 2, 3])
                else:
                    action = np.argmax(self.qtable[row, col, :])
                
                newState = self.getState(self.actions[action])
                reward = self.getReward()

                self.qtable[row, col, action] = (
    (1 - self.learning_rate) * self.qtable[row, col, action] +
    self.learning_rate * (reward + self.discount_factor * np.max(self.qtable[newState[0], newState[1], :]))
)

                self.state = newState
                steps += 1

                if(self.state == self.goal)  or steps >= maxSteps:
                    done = True
                
            self.exploration_rate = max(self.min_exploration_rate, self.exploration_rate*self.decay_rate)

    def aiPlay(self, maxSteps):
        done = False
        steps = 0
        self.state = [0, 0]
        while not done:
            time.sleep(1)
            state = self.state
            bestAction = np.argmax(self.qtable[state[0], state[1], :])
            self.render()
            print("*********************")
            if state == self.goal:
                print("Agent reached the goal!")
                done = True
                continue
            elif self.map[state[0]][state[1]] == 'l':
                print("Agent fell in lava")
                done = True
                continue
            elif steps >= maxSteps:
                print("Max Steps exceeded!")
                done = True
                continue
            print("Step:", steps+1)
            print("Best action seems to be....", self.actions[bestAction])
            self.state = self.getState(self.actions[bestAction])
            steps += 1