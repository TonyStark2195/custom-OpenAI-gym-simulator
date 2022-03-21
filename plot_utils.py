import matplotlib.pyplot as plt
import numpy as np
import os

# plotting method that takes reward history of a policy repeated over 10 trials and compares them
def plot_results(reward_history, policy_type, filepath):
    plt.figure(figsize=(15,20))
    sum_result = 0
    for i in range(len(reward_history)):
        sum_result+= np.array(reward_history[i])
        plt.plot(range(len(reward_history[i])), reward_history[i], linestyle=':', linewidth=2)
    
    plt.plot(range(len(sum_result)), sum_result/len(reward_history), c='black', linewidth=2)
    plt.xlabel('Steps')
    plt.ylabel('Cumulative Reward')
    plt.title('Reward pattern using ' + policy_type + ' Policy')
    os.makedirs(filepath, exist_ok=True)
    plt.savefig(filepath + '/Reward pattern using ' + policy_type + ' Policy.jpeg', dpi=400, bbox_inches = 'tight')
    plt.show()

# plotting method that takes reward history of all the policies executed with a goal location
# plots the average performance of each policy over 10 trials
def plot_summary(reward_history, filepath):
    plt.figure(figsize=(15,20))
    color = {
                'better': 'blue',
                'best': 'yellow',
                'better_learning': 'black', 
                'true_random': 'red', 
                'worse': 'green',
                'worser': 'orange'
            }
    for k in reward_history.keys():    
        sum_result = 0
        for i in range(len(reward_history[k])):
            sum_result+= np.array(reward_history[k][i])
            plt.plot(range(len(reward_history[k][i])), reward_history[k][i], linestyle=':', linewidth=2)

        plt.plot(range(len(sum_result)), sum_result/len(reward_history[k]), c=color[k], linewidth=2, label=k)
    plt.legend(loc='best')
    plt.xlabel('Steps')
    plt.ylabel('Cumulative Reward')
    plt.title('Reward pattern using different Policies')
    os.makedirs(filepath, exist_ok=True)
    plt.savefig(filepath + '/Reward pattern using different Policies.jpeg', dpi=400, bbox_inches = 'tight')
    plt.show()