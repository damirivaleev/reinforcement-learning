import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline

n = 10
k = 2000
r = 1000
e = 0.1
beta = 0.1
alpha = 0.1
q_true = np.random.normal(0, 1, (k,n))

for mode in [0, 1, 2]:
    print('\nРежим {}, {}'.format(mode, ['е-жадный (e=0.1)',
                                         'сравнение с эталоном',
                                         'сравнение с эталоном + коэффициент'][mode]))
    average_reward = 0
    #rewards_per_run = np.zeros(r)
    optimum_per_run = np.zeros(r)

    for run in range(k):
        q_estimated = np.zeros(n)
        predpoch = np.zeros(n)
        rewards = 0
        arm_used = np.zeros(n)
        optimum_choice = 0

        for i in range(r):
            if mode == 0:
                if np.random.random() < e:
                    arm = np.random.choice(n)
                else:
                    arm = np.argmax(q_estimated)                    
            else:
                softmax_prob = np.exp(predpoch) / np.sum(np.exp(predpoch))
                arm = np.random.choice(range(n), p=softmax_prob)
            
            arm_used[arm] += 1

            if arm == np.argmax(q_true[run]):
                optimum_per_run[i] += 1
            
            current_reward = np.random.normal(q_true[run][arm], 1)
            rewards += current_reward
            #rewards_per_run[i] += current_reward

            if mode == 0:
                q_estimated[arm] += (current_reward - q_estimated[arm]) / arm_used[arm]
            if mode == 1:
                predpoch[arm] += beta * (current_reward - q_estimated[arm] / arm_used[arm])
                #print(arm, predpoch)
                q_estimated[arm] += alpha * (current_reward - q_estimated[arm] / arm_used[arm])
            if mode == 2:
                predpoch[arm] += (1 - softmax_prob[arm]) * beta * (current_reward - q_estimated[arm] / arm_used[arm])
                q_estimated[arm] += alpha * (current_reward - q_estimated[arm] / arm_used[arm])
        
        average_reward += rewards / r
        if run % 100 == 0:
            print('\nRun', run)            

    #rewards_per_run = rewards_per_run / k
    optimum_per_run = optimum_per_run / k

    #plt.plot(range(r), rewards_per_run, label='t = {}'.format(t))
    plt.plot(range(r), optimum_per_run*100, label='Режим {}'.format(mode))

plt.ylim(0, 100)
plt.legend()
plt.show()
