# coding: utf-8
import numpy as np
from myplot.lines import plot_lines
runs=2000
steps=1000
eps=0.0
k=10
qstar=[]
for _ in range(runs):
    qstar.append(np.random.normal(0,1,k))


def run_eps_greedy(qstar, runs=2000, steps=1000, eps=0.0, k=10):
    rewardlist = []
    for run in range(runs):
        l = []
        qest = np.zeros(k)
        count = np.zeros(k)
        rewardsum = np.zeros(k)
        action = np.random.randint(0, k)
        count[action] += 1
        qvalues = qstar[run]
        reward = np.random.normal(qvalues[action], 1)
        rewardsum[action] += reward
        l.append((action, reward))
        qest[action] = rewardsum[action] * 1.0 / count[action]
        prob = np.random.uniform(0, 1, steps - 1)
        for step in range(steps - 1):
            action_max = np.argmax(qest)
            action_rand = np.random.randint(0, k)
            action = action_rand
            if prob[step] > eps:
                action = action_max
            count[action] += 1
            reward = np.random.normal(qvalues[action], 1)
            rewardsum[action] += reward
            l.append((action, reward))
            qest[action] = rewardsum[action] * 1.0 / count[action]
        rewardlist.append(l)
    return rewardlist


def compute_average_reward(rewardlist, runs=2000, steps=1000):
    rewardarray = np.zeros((runs, steps), dtype=np.float)
    for run in range(runs):
        for step in range(steps):
            rewardarray[run][step] = rewardlist[run][step][1]
    avgreward = np.average(rewardarray, axis=0)
    return avgreward

def count_optimal_action_selection(rewardlist, runs=2000, steps=1000):
    countarray = np.zeros((runs, steps))
    for run in range(runs):
        opt_action = np.argmax(qstar[run])
        for step in range(steps):
            countarray[run][step] = int(opt_action == rewardlist[run][step][0])
    opt_action_pc = np.sum(countarray, axis=0) / len(rewardlist)
    return opt_action_pc

def iterate_eps_greedy(qstar, epslist, runs=2000, steps=1000, k=10, plot=0):
    eps_run_list = []
    for eps in epslist:
        r = run_eps_greedy(qstar, runs, steps, eps, k)
        a = compute_average_reward(r, runs, steps)
        o = count_optimal_action_selection(r, runs, steps)
        eps_run_list.append((eps,r,a,o))
    return eps_run_list


""" EPS lists
epslist = [0, 0.01, 0.1]
epslist = [0.1]
epslist = [0, 0.01, 0.05, 0.075, 0.1, 0.125, 0.15, 0.2, 0.3, 0.4, 0.5]
"""

# ALL
epslist = [0, 0.01, 0.1]
eps_run_list=iterate_eps_greedy(qstar,epslist)

x = np.linspace(0, steps, steps)
t=[]
for e,r,a,o in eps_run_list:
    t.append((x,a, '%f greedy'%e))
plot_lines(t, 'Average Rewards', 'x', 'y')

t=[]
for e,r,a,o in eps_run_list:
    t.append((x,o, '%f greedy'%e))
plot_lines(t, 'Optimal Action %', 'x', 'y')
