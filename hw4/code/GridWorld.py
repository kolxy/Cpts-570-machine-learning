import enum
import numpy as np
import random
import math

class TYPE(enum.Enum):
    cell = 0
    wall = 1

class ACTION(enum.Enum):
    up = 0
    down = 1
    left = 2
    right = 3

    @staticmethod
    def list():
        return list(map(lambda c: c, ACTION))

class Grid:
    def __init__(self, x, y, type, reward):
        self.x = x
        self.y = y
        self.type = type
        self.reward = reward
        self.Q = dict()
        self.init_Q()

    def init_Q(self):
        self.Q = {ACTION.up: 0,
                  ACTION.down: 0,
                  ACTION.left: 0,
                  ACTION.right: 0}

    def is_terminal(self) -> bool:
        return self.reward != 0

    def is_wall(self) -> bool:
        return self.type == TYPE.wall

    def get_action_epsilon(self, epsilon):
        if np.random.random() < epsilon:
            maxValue = max(self.Q.values())
            keys = [key for key, value in self.Q.items() if value == maxValue]
            action = random.choice(keys)
            return action
        else:
            action = random.choice(ACTION.list())
            return action

    def get_action_boltzmann(self, temperature):
        action_list = ACTION.list()
        weights = []
        # Denomeninator removed since it's constant for all actions
        for action in action_list:
            weights.append(int(math.exp(self.Q[action]/temperature) * 100))
        selected = random.choices(action_list, weights, k=1)
        return selected[0]

    # Get max Q-value or reward
    def max_Q(self):
        return max(max(self.Q.values()), self.reward)

    def __str__(self):
        Q = dict([(k.name, v) for k, v in self.Q.items()])
        return "[{0}, {1}], {2}, reward:{3},  Q-values: {4}".format(self.x,
                                             self.y,
                                             self.type.name.upper(),
                                             self.reward,
                                             Q)
                                                                
class GridWorld:
    def __init__(self):
        self.width = 10
        self.height = 10
        self.default_type = TYPE.cell
        self.default_reward = 0
        self.matrix = [[ None for i in range(self.height)] for j in range(self.width)]
        for x in range(self.width):
            for y in range(self.height):
                self.matrix[x][y] = Grid(x, y, self.default_type, self.default_reward)
        
        self.load_walls()
        self.load_rewards()

    def load_walls(self):
        walls = [(1,2),(2,2),(3,2),(4,2),
                 (4,3),(4,4),(4,5),(4,6),(4,7),
                 (6,2),(7,2),(8,2)]
        for wall in walls:
            self.matrix[wall[0]][wall[1]].type = TYPE.wall
    
    def load_rewards(self):
        rewards = [(3,3,-1),(5,4,-1),(6,4,-1),(5,5,1),(6,5,-1),
                   (8,5,-1),(8,6,-1),(3,7,-1),(5,7,-1),(6,7,-1)]
        for reward in rewards:
            self.matrix[reward[0]][reward[1]].reward = reward[2]

    def get(self, x, y) -> Grid:
        return self.matrix[x][y]
    
    def move(self, state, action) -> Grid:
        x, y = state.x, state.y
        if (action == ACTION.up) & (y > 0):
            y -= 1
        if (action == ACTION.down) & (y < self.height - 1):
            y += 1
        if (action == ACTION.left) & (x > 0):
            x -= 1
        if (action == ACTION.right) & (x < self.width - 1):
            x += 1
        
        new_state = self.get(x, y)
        if not new_state.is_wall():
            return new_state
        
        return state

    def print_shape(self, empty = True):
        res = ""
        for x in range(self.width):
            for y in range(self.height):
                val = " "
                if self.matrix[y][x].type == TYPE.wall:
                    val = "X"
                elif self.matrix[y][x].reward < 0:
                    val = "-"
                elif self.matrix[y][x].reward > 0:
                    val = "G"
                elif not empty:
                    action = self.matrix[y][x].get_action_epsilon(1)
                    if action == ACTION.up:
                        val = "^"
                    elif action == ACTION.down:
                        val = "v"
                    elif action == ACTION.left:
                        val = "<"
                    elif action == ACTION.right:
                        val = ">"

                res = res + " | " + val
            res = res + " |\n"
        print(res)

    def __str__(self):
        res = ""
        for x in range(self.width):
            for y in range(self.height):
                res = res + self.matrix[y][x].reward
        
        return res