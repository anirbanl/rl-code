# coding: utf-8
import numpy as np
from myplot.lines import plot_lines
runs=2000
steps=10000
k=10

qstar=[]
for _ in range(runs):
    l=np.zeros((steps,k),dtype=np.float)
    l[0]=np.random.normal(0,1,1)*np.ones(k)
    for step in range(steps-1):
        l[step+1]=l[step]+np.random.normal(0, 0.01, k)
    qstar.append(l)

def run_eps_greedy_nonstationary(qstar, runs=2000, steps=1000, eps=0.0, k=10, alpha='avg'):
    rewardlist = []
    for run in range(runs):
        l = []
        qest = np.zeros(k)
        count = np.zeros(k)
        rewardsum = np.zeros(k)
        action = np.random.randint(0, k)
        count[action] += 1
        qvalues = qstar[run][0]
        reward = np.random.normal(qvalues[action], 1)
        rewardsum[action] += reward
        l.append((action, reward))
        if alpha=='avg':
            qest[action] = rewardsum[action] * 1.0 / count[action]
        else:
            qest[action] = qest[action] + alpha * (reward - qest[action])
        prob = np.random.uniform(0, 1, steps - 1)
        for step in range(steps - 1):
            qvalues = qstar[run][step+1]
            action_max = np.argmax(qest)
            action_rand = np.random.randint(0, k)
            action = action_rand
            if prob[step] > eps:
                action = action_max
            count[action] += 1
            reward = np.random.normal(qvalues[action], 1)
            rewardsum[action] += reward
            l.append((action, reward))
            if alpha == 'avg':
                qest[action] = rewardsum[action] * 1.0 / count[action]
            else:
                qest[action] = qest[action] + alpha * (reward - qest[action])
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
        for step in range(steps):
            opt_action = np.argmax(qstar[run][step])
            countarray[run][step] = int(opt_action == rewardlist[run][step][0])
    opt_action_pc = np.sum(countarray, axis=0) / len(rewardlist)
    return opt_action_pc

def iterate_eps_greedy(qstar, epslist, runs=2000, steps=1000, k=10, alpha='avg'):
    eps_run_list = []
    for eps in epslist:
        r = run_eps_greedy_nonstationary(qstar, runs, steps, eps, k, alpha)
        a = compute_average_reward(r, runs, steps)
        o = count_optimal_action_selection(r, runs, steps)
        eps_run_list.append((eps,r,a,o))
    return eps_run_list

epslist = [0.0, 0.01, 0.1]
eps_run_list_avg=iterate_eps_greedy(qstar, epslist, runs, steps, k, alpha='avg')
eps_run_list_alpha=iterate_eps_greedy(qstar, epslist, runs, steps, k, alpha=0.1)


x = np.linspace(0, steps, steps)
t=[]
for e,r,a,o in eps_run_list_avg:
    t.append((x,a, '%f greedy'%e))
plot_lines(t, 'Average Rewards', 'x', 'y')
t=[]
for e,r,a,o in eps_run_list_avg:
    t.append((x,o, '%f greedy'%e))
plot_lines(t, 'Optimal Action %', 'x', 'y')

t=[]
for e,r,a,o in eps_run_list_alpha:
    t.append((x,a, '%f greedy'%e))
plot_lines(t, 'Average Rewards alpha=0.1', 'x', 'y')
t=[]
for e,r,a,o in eps_run_list_alpha:
    t.append((x,o, '%f greedy'%e))
plot_lines(t, 'Optimal Action % alpha=0.1', 'x', 'y')
