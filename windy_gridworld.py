import numpy as np
import time
import matplotlib.pyplot as plt
import copy
import os
import pandas as pd

GRID = [[0, 0, 0, 1, 1, 1, 2, 2, 1, 0],
        [0, 0, 0, 1, 1, 1, 2, 2, 1, 0],
        [0, 0, 0, 1, 1, 1, 2, 2, 1, 0],
        ['S', 0, 0, 1, 1, 1, 2, 'G', 1, 0],
        [0, 0, 0, 1, 1, 1, 2, 2, 1, 0],
        [0, 0, 0, 1, 1, 1, 2, 2, 1, 0],
        [0, 0, 0, 1, 1, 1, 2, 2, 1, 0]]

random_wind = True
epsilon = 0.1
alpha = 0.5
gamma = 1
height = len(GRID)
weight = len(GRID[0])
actions = [-1, 0, 1]

for row in range(len(GRID)):
    for col in range(len(GRID[row])):
        if GRID[row][col] == 'S':
            start = (row, col)
            break

for row in range(len(GRID)):
    for col in range(len(GRID[row])):
        if GRID[row][col] == 'G':
            goal = (row, col)
            break

def print_grid(iteration, position):
    if iteration > 7800:
        os.system('cls')
        print('Step', iteration)
        grid = copy.deepcopy(GRID)
        grid[position[0]][position[1]] = 'o'
        for row in range(len(grid)):
            for col in range(len(grid[row])):            
                try:
                    grid[row][col] += 1                
                    grid[row][col] = '.'                
                except TypeError:                
                    continue
        
        for line in grid:        
            print(' '.join(line))
        time.sleep(0.1)

def choose_move(position, directions=4):    
    moves = []
    for i in actions:
        for j in actions:
            moves.append((i, j))

    if directions == 9:
        pass
    else:
        moves = [m for m in moves if m[0] != 0 or m[1] != 0]
        if directions == 4:
            moves = [m for m in moves if 0 in m]
    
    moves = [m for m in moves if (0 <= position[0] + m[0] < height) and (0 <= position[1] + m[1] < weight)]

    for m in moves:
        if (position, m) not in q_eval:
            q_eval[(position, m)] = 0

    if np.random.random_sample() < epsilon:
        move_ind = np.random.choice(range(len(moves)))
        return moves[move_ind]
    else:
        current_evals = {}
        for pos, mov in q_eval:
            if pos == position:
                current_evals[pos, mov] = q_eval[pos, mov]
        
        return max(current_evals, key=current_evals.get)[1]

q_eval = {}
episodes = 0
iteration = 0
result_pd = pd.Series()

while iteration in range(8000):
    
    current_position = start
    print_grid(iteration, current_position)
    
    move = choose_move(current_position, 8)
    
    steps = 0
    while True:
        if current_position != goal:
            
            next_position = (current_position[0] + move[0], current_position[1] + move[1])
            try:
                wind = int(GRID[current_position[0]][current_position[1]])
            except (TypeError, ValueError):
                wind = 0
            if random_wind == True:
                if wind > 0:
                    wind += np.random.choice([-1, 0, 1])
            while wind:
                next_position = (next_position[0] - 1, next_position[1])                
                next_position = (max(0, next_position[0]), next_position[1])            
                if next_position == goal:
                    break
                
                wind -= 1
        
            reward = -1

            next_move = choose_move(next_position, 8)
            
            q_eval[(current_position, move)] += alpha * (reward + gamma * q_eval[(next_position, next_move)] - q_eval[(current_position, move)])

            current_position = next_position            
            move = next_move
            
            print_grid(iteration, current_position)            
            
            result_pd = result_pd.append(pd.Series([episodes], index=[iteration]), verify_integrity=True)
            iteration += 1

            steps += 1
            if iteration == 8000:
                break
        else:
            q_eval[(current_position, move)] = (1 - alpha) * q_eval[(current_position, move)]
            episodes += 1           
            
            result_pd = result_pd.append(pd.Series([episodes], index=[iteration]), verify_integrity=True)
            iteration += 1
            
            break

print('\nEpisodes', episodes)

best_evals = {}
best_moves = {}
for pos, mov in q_eval:    
    if pos not in best_evals:
        best_evals[pos] = -1000000
        best_moves[pos] = (0, 0)
    if q_eval[(pos, mov)] > best_evals[pos]:
        best_evals[pos] = q_eval[(pos, mov)]
        best_moves[pos] = mov
        
dir_grid = copy.deepcopy(GRID)
direction = {(-1, 0): '^', (1, 0): 'v', (0, -1): '<', (0, 1): '>', (-1, -1): '┌', (-1, 1): '┐', (1, -1): '└', (1, 1): '┘', (0, 0): 'x'}

for pos in best_moves:    
    dir_grid[pos[0]][pos[1]] = direction[best_moves[pos]]

for row in range(len(dir_grid)):
    for col in range(len(dir_grid[row])):            
        try:
            dir_grid[row][col] += 1                
            dir_grid[row][col] = '.'                
        except TypeError:                
            continue
        
for line in dir_grid:        
    print(' '.join(line))

plt.yticks([0, 50, 100, 150, 170, episodes])
plt.plot(result_pd)
plt.show()
