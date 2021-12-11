from GridWorld import Grid, GridWorld, TYPE, ACTION
import numpy as np

ALPHA = 0.01 # learning rate
BETA = 0.9 # discount factor
def main():
    world = GridWorld()
    epsilon = 0.2
    world.print_shape(empty = True)

    for t in range(10000):
        x, y = 0, 0
        state = world.get(x, y)

        while not state.is_terminal():
            action = state.get_action_epsilon(epsilon)
            new_state = world.move(state, action)
            # walking into wall will not update anything
            if (state == new_state):
                continue
            state.Q[action] = state.Q[action] + ALPHA*(state.reward + BETA*new_state.max_Q() - state.Q[action])
            state = new_state
    print("========================= All Q-values =========================\n")

    for x in range(len(world.matrix)):
        for y in range(len(world.matrix[x])):
            print(world.matrix[y][x])

    path = get_path(world)
    
    print("\n============================ Path ==============================\n")
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

if __name__ == "__main__":
    main()