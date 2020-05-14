from track import TRACK_1, TRACK_2
import numpy as np
import copy
import time
import os

epsilon = 0.01
shift = False

def print_track(track):    
    code = {0: '⬆ ', 1: 'X', 2: '□ ', 3: '. ', 4: '■ '}
    for line in track:
        changed_line = ''
        for l in line:
            changed_line += code[l]
        print(changed_line)

def possible_actions(speed):
    actions = [-1, 0, 1]
    pos_act = []
    for a in actions:
        for b in actions:
            pos_act.append((a+speed[0], b+speed[1]))

    return [a for a in pos_act if (a[0] != 0 or a[1] != 0) and (abs(a[0]) <= 5 and abs(a[1]) <= 5)]


clear = lambda: os.system('cls')
def animation(iteration, track, path, best_episode=False):
    temp_track = copy.deepcopy(track)
    if (iteration % 5000 == 0) or (iteration in range(100)) or (iteration in range(100, 1000, 100)) or (iteration in range(1000, 5000, 1000)) or best_episode:
        for (x, y) in path:
            temp_track[x][y] = 4
            clear()
            print('\nIteration', iteration)
            print_track(list(reversed(temp_track)))
            time.sleep(0.1)
        
        if track[path[-1][0]][path[-1][1]] == 1:
            print('Steps =', len(path))
            if best_episode:
                print('Best episode')
                time.sleep(5)
            else:
                time.sleep(1)
        return True
    else:
        return False

def evaluation(states, rewards):
    already_eval_states = []
    count = len(states)
    for i in range(count):
        if states[i] not in already_eval_states:
            q_eval[states[i]] += sum(rewards[i:]) / (count - i)
            already_eval_states.append(states[i])
        else: continue

for track_number in range(2):

    track = copy.deepcopy([TRACK_1, TRACK_2][track_number])
    rev_track = copy.deepcopy(track)[::-1]

    start = []
    for i in range(len(rev_track)):
        for j in range(len(rev_track[i])):
            if rev_track[i][j] == 0:
                start.append((i, j))

    for st in start:
        q_eval = {}
        visit = {}
        best_steps = 1000000
        success = 0
        last_iter = 0

        for iteration in range(1, 250001):
            
            rev_track = copy.deepcopy(track)[::-1]
            
            cur_pos = st

            path = []
            path.append(st)
            
            speed = (0, 0)

            states_per_iter = []
            rewards_per_iter = []
            best_episode = False
            
            while True:
                
                pos_spd = possible_actions(speed)
                
                for spd in pos_spd:        
                    if (cur_pos, spd) not in q_eval:
                        q_eval[(cur_pos, spd)] = 0
                        visit[(cur_pos, spd)] = 0

                temp_dict = {}        
                for s in pos_spd:
                    try:
                        temp_dict[(cur_pos, s)] = q_eval[(cur_pos, s)] / visit[(cur_pos, s)]
                    except ZeroDivisionError:
                        temp_dict[(cur_pos, s)] = q_eval[(cur_pos, s)]

                argmax = max(temp_dict, key=temp_dict.get)
                
                prob_for_action = []
                for a in pos_spd:
                    if a == argmax[1]:
                        prob_for_action.append(1 - epsilon + epsilon / len(pos_spd))
                    else:
                        prob_for_action.append(epsilon / len(pos_spd))

                next_spd = np.random.choice(len(pos_spd), p=prob_for_action)
                speed = pos_spd[next_spd]

                next_pos = tuple(sum(pair) for pair in zip(cur_pos, speed))
                
                state = (cur_pos, speed)
                states_per_iter.append(state)
                
                try:
                    pos_type = rev_track[next_pos[0]][next_pos[1]]
                except:
                    reward = -5
                    rewards_per_iter.append(reward)

                    evaluation(states_per_iter, rewards_per_iter)
                    
                    visit[(cur_pos, speed)] += 1
                    
                    break

                if next_pos[0] < 0 or next_pos[1] < 0:
                    pos_type = 3
                    next_pos = cur_pos
                
                if pos_type == 3:   #out of track                    
                    reward = -5
                    rewards_per_iter.append(reward)

                    evaluation(states_per_iter, rewards_per_iter)
                    
                    visit[(cur_pos, speed)] += 1
                    path.append(next_pos)
                    break
                elif pos_type == 1:     #episode successfully finished
                    evaluation(states_per_iter, rewards_per_iter)
                    
                    path.append(next_pos)

                    success += 1
                    
                    if best_steps > len(states_per_iter):
                        best_episode = True
                        best_pi = states_per_iter
                        best_steps = len(states_per_iter)
                        with open('track{}_best_pi_{}{}.txt'.format(track_number+1, st, ['', '_with_shift'][shift]), 'a') as f:
                            f.write('Steps = {}\n'.format(best_steps))
                            f.write('{}\n\n'.format(best_pi))
                        
                    break
                else:       #still in track                    
                    reward = -1
                    rewards_per_iter.append(reward)
                    visit[(cur_pos, speed)] += 1

                cur_pos = next_pos

                path.append(cur_pos)

                if shift:
                    if np.random.choice([True, False]):
                        shift_direction = np.random.randint(2)
                        cur_pos = (cur_pos[0] + shift_direction, cur_pos[1] + 1 - shift_direction)

            if animation(iteration, rev_track, path, best_episode):
                print('Successful episodes {}/{} - {}%'.format(success, iteration - last_iter, np.around(success / (iteration - last_iter) * 100, decimals=1)))
                if success > 0:
                    time.sleep(5)
                    success = 0
                    last_iter = iteration
                else:
                    time.sleep(0.5)
