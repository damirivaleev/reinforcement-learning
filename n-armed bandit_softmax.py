import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline

n = 2
k = 2000
r = 1000
q_true = np.random.normal(0, 1, (k,n))

for t in [0.5, 0.1, 0.05]:
    print('\ntemperature {}'.format(t))
    average_reward = 0
    rewards_per_run = np.zeros(r)
    optimum_per_run = np.zeros(r)

    for run in range(k):
        q_estimated = np.zeros(n)
        rewards = 0
        arm_used = np.zeros(n)
        optimum_choice = 0

        for i in range(r):
            softmax_prob = np.exp(q_estimated / t) / np.sum(np.exp(q_estimated / t))
            arm = np.random.choice(range(n), p=softmax_prob)
            
            arm_used[arm] += 1

            if arm == np.argmax(q_true[run]):
                optimum_per_run[i] += 1
            
            current_reward = np.random.normal(q_true[run][arm], 1)
            rewards += current_reward
            rewards_per_run[i] += current_reward
            
            q_estimated[arm] += (current_reward - q_estimated[arm]) / arm_used[arm]
        
        average_reward += rewards / r
        if run % 100 == 0:
            print('\nRun', run)
            #print('Average reward', average_reward / (run+1))
            #print('reward this run', rewards / r)
            #print('true', q_true)
            #print('estimated', q_estimated)

    #print(average_reward / k)

    rewards_per_run = rewards_per_run / k
    optimum_per_run = optimum_per_run / k

    plt.plot(range(r), rewards_per_run, label='t = {}'.format(t))
    #plt.plot(range(r), optimum_per_run*100, label='t = {}'.format(t))

#plt.ylim(0, 100)
plt.legend()
plt.show()
