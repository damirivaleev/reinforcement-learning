import numpy as np
import matplotlib.pyplot as plt

gamma = 1
e = 10** -10
states = range(1, 100)

for p in [0.25, 0.4, 0.55]:
    print('Prob_win =', p)
    prob_win = p
    prob_lose = 1 - prob_win
    run = 0

    v_s = np.zeros(len(states)+2)
    v_s[0] = 0
    v_s[100] = 1

    plt.figure()
    #plt.xlim(1, 99)
    plt.xticks([1, 25, 50, 75, 99])

    def eval_actions_for_state(state):    
        return [eval_reward(action, state) for action in range(1, np.minimum(state, (100 - state))+1)]

    def eval_reward(action, state):
        return (gamma * v_s[state + action] * prob_win +
                gamma * v_s[state - action] * prob_lose)
            
    while True:    
        delta = 0
        for state in states:
            v = v_s[state]
            v_s[state] = np.max(eval_actions_for_state(state))        
            delta = np.maximum(delta, abs(v - v_s[state]))

        if run < 3:
            plt.plot(states, v_s[1:100], label='Run {}'.format(run+1))
        run += 1
            
        if delta < e:
            break

    print('Runs =', run+1)
    #нарисовать финальные ценности
    plt.plot(states, v_s[1:100], label='Final ({}) run'.format(run+1))
    plt.legend()
    plt.show()

    strategy = [range(1, np.minimum(state, (100 - state))+1)[np.argmax(eval_actions_for_state(state))] for state in states]

    plt.figure()
    plt.xticks([1, 25, 50, 75, 99])
    plt.yticks([1, 10, 20, 30, 40, 50])
    plt.step(states, strategy)
    plt.show()
