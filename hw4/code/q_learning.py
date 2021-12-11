from GridWorld import Grid, GridWorld, TYPE, ACTION
import numpy as np
import copy

ALPHA = 0.01 # learning rate
BETA = 0.9 # discount factor
def main():
    world = GridWorld()
    world.print_shape(empty = True)

    # command
    print("Choose policy (0: epsilon, 1: Boltzmann):")
    policy = input()
    print("Input parameter for chosen policy (epsilon or temperature):")
    param = float(input())
    if policy == "0":
        run_epsilon_policy(world, param)
    elif policy == "1":
        run_boltzmann_policy(world, param)
    else:
        print("Bad input!")
        exit()
    
    print("========================= All Q-values =========================\n")
    for x in range(len(world.matrix)):
        for y in range(len(world.matrix[x])):
            print(world.matrix[y][x])
    
    print("\n============================ Path ==============================\n")
    path = get_path(world)
    for s in path:
        print(s)

    world.print_shape(empty = False)

def get_path(world: GridWorld):
    path = []
    state = world.get(0, 0)
    path.append(state)
    while not state.is_terminal():
        action = state.get_action_epsilon(1)
        state = world.move(state, action)
        path.append(state)
    
    return path

def run_epsilon_policy(world: GridWorld, epsilon):
    iteration = 0
    count = 0
    max_iter = 10000
    # if Q hasn't changed for max_iter times, it's convergence
    while count < max_iter: 
        x, y = 0, 0
        state = world.get(x, y)
        static_flag = True # monitor any Q value change in one iteration
        while not state.is_terminal():
            action = state.get_action_epsilon(epsilon)
            new_state = world.move(state, action)
            # walking into wall will not update anything
            if (state == new_state):
                continue

            oldQ = copy.deepcopy(state.Q)
            # the money
            state.Q[action] = state.Q[action] + ALPHA*(state.reward + BETA*new_state.max_Q() - state.Q[action])

            static_flag = True if oldQ == state.Q else False
            state = new_state
        
        count = count + 1 if static_flag else 0
        iteration += 1
        if iteration % 10000 == 0:
            print(iteration)
    
    print(f"Epsilon: {epsilon}, Iteration: {iteration}")
    return iteration

def run_boltzmann_policy(world: GridWorld, temperature):
    iteration = 0
    count = 0
    max_iter = 10000
    temp = temperature
    while count < max_iter:
        x, y = 0, 0
        state = world.get(x, y)
        static_flag = True # monitor any Q value update in one iteration
        while not state.is_terminal():
            action = state.get_action_boltzmann(temperature)
            new_state = world.move(state, action)
            # walking into wall will not update anything
            if (state == new_state):
                continue

            oldQ = copy.deepcopy(state.Q)
            # the money
            state.Q[action] = state.Q[action] + ALPHA*(state.reward + BETA*new_state.max_Q() - state.Q[action])

            static_flag = True if oldQ == state.Q else False
            state = new_state
        
        count = count + 1 if static_flag else 0
        iteration += 1
        if iteration % 10000 == 0:
            print(iteration)
        
        temp += 0.9
    
    print(f"Temperature: {temperature}, Iteration: {iteration}")
    return

if __name__ == "__main__":
    main()