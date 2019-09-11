# coding: utf-8
import numpy as np
from myplot.lines import plot_lines

def run_eps_greedy_alpha(qstar, initval=0, runs=2000, steps=1000, eps=0.0, k=10, alpha='avg'):
    rewardlist = []
    for run in range(runs):
        l = []
        qest = np.zeros(k)+initval
        count = np.zeros(k)
        rewardsum = np.zeros(k)
        action = np.random.randint(0, k)
        count[action] += 1
        qvalues = qstar[run]
        reward = np.random.normal(qvalues[action], 1)
        rewardsum[action] += reward
        l.append((action, reward))
        if alpha=='avg':
            qest[action] = rewardsum[action] * 1.0 / count[action]
        else:
            qest[action] = qest[action] + alpha * (reward - qest[action])
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
            if alpha == 'avg':
                qest[action] = rewardsum[action] * 1.0 / count[action]
            else:
                qest[action] = qest[action] + alpha * (reward - qest[action])
        rewardlist.append(l)
    return rewardlist

def run_upper_confidence_bound(qstar, initval=0, runs=2000, steps=1000, eps=0.0, k=10, alpha='avg', c=2):
    rewardlist = []
    for run in range(runs):
        l = []
        qest = np.zeros(k)+initval
        count = np.zeros(k)
        rewardsum = np.zeros(k)
        action = np.random.randint(0, k)
        count[action] += 1
        qvalues = qstar[run]
        reward = np.random.normal(qvalues[action], 1)
        rewardsum[action] += reward
        l.append((action, reward))
        if alpha=='avg':
            qest[action] = rewardsum[action] * 1.0 / count[action]
        else:
            qest[action] = qest[action] + alpha * (reward - qest[action])
        prob = np.random.uniform(0, 1, steps - 1)
        for step in range(steps - 1):
            min_count_indices = np.where(count == count.min())[0]
            num_min = min_count_indices.shape[0]
            random_min_index = min_count_indices[np.random.randint(0, num_min)]
            if count[random_min_index] == 0:
                action_max = random_min_index
            else:
                action_max = np.argmax(qest + c * np.sqrt(np.log(step+1)/count))
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

def count_optimal_action_selection(qstar, rewardlist, runs=2000, steps=1000):
    countarray = np.zeros((runs, steps))
    for run in range(runs):
        opt_action = np.argmax(qstar[run])
        for step in range(steps):
            countarray[run][step] = int(opt_action == rewardlist[run][step][0])
    opt_action_pc = np.sum(countarray, axis=0) / len(rewardlist)
    return opt_action_pc

def define_qstar(runs, k):
    qstar = []
    for _ in range(runs):
        qstar.append(np.random.normal(0, 1, k))
    return qstar

if __name__== "__main__":
    runs_list = ['greedy', '0.01greedy', '0.1greedy', 'opt_greedy', 'opt_greedy_0.1alpha', '0.1greedy_0.1alpha', 'UCB']
    plots = []
    global_steps = 1000

    ##### Run ONE - Greedy policy and sample average movement
    runs, steps, eps, k, alpha = 2000, global_steps, 0.0, 10, 'avg'
    qstar, initval = define_qstar(runs, k), 0
    r = run_eps_greedy_alpha(qstar, initval, runs, steps, eps, k, alpha)
    a = compute_average_reward(r, runs, steps)
    o = count_optimal_action_selection(qstar, r, runs, steps)
    plots.append(([runs, steps, eps, k, alpha], [a,o], runs_list[0]))

    ##### Run TWO - Greedy with eps 0.01 and sample average
    runs, steps, eps, k, alpha = 2000, global_steps, 0.01, 10, 'avg'
    initval = 0
    r = run_eps_greedy_alpha(qstar, initval, runs, steps, eps, k, alpha)
    a = compute_average_reward(r, runs, steps)
    o = count_optimal_action_selection(qstar, r, runs, steps)
    plots.append(([runs, steps, eps, k, alpha], [a,o], runs_list[1]))

    ##### Run THREE - Greedy with eps 0.1 and sample average
    runs, steps, eps, k, alpha = 2000, global_steps, 0.1, 10, 'avg'
    initval = 0
    r = run_eps_greedy_alpha(qstar, initval, runs, steps, eps, k, alpha)
    a = compute_average_reward(r, runs, steps)
    o = count_optimal_action_selection(qstar, r, runs, steps)
    plots.append(([runs, steps, eps, k, alpha], [a,o], runs_list[2]))

    ##### Run FOUR - Optimistic Greedy and sample average
    runs, steps, eps, k, alpha = 2000, global_steps, 0.0, 10, 'avg'
    initval = 5
    r = run_eps_greedy_alpha(qstar, initval, runs, steps, eps, k, alpha)
    a = compute_average_reward(r, runs, steps)
    o = count_optimal_action_selection(qstar, r, runs, steps)
    plots.append(([runs, steps, eps, k, alpha], [a,o], runs_list[3]))

    ##### Run FIVE - Optimistic Greedy and alpha 0.1
    runs, steps, eps, k, alpha = 2000, global_steps, 0.0, 10, 0.1
    initval = 5
    r = run_eps_greedy_alpha(qstar, initval, runs, steps, eps, k, alpha)
    a = compute_average_reward(r, runs, steps)
    o = count_optimal_action_selection(qstar, r, runs, steps)
    plots.append(([runs, steps, eps, k, alpha], [a,o], runs_list[4]))

    ##### Run SIX - 0.1 Greedy and alpha 0.1
    runs, steps, eps, k, alpha = 2000, global_steps, 0.1, 10, 0.1
    initval = 0
    r = run_eps_greedy_alpha(qstar, initval, runs, steps, eps, k, alpha)
    a = compute_average_reward(r, runs, steps)
    o = count_optimal_action_selection(qstar, r, runs, steps)
    plots.append(([runs, steps, eps, k, alpha], [a, o], runs_list[5]))

    ##### Run SEVEN - UCB with c=2 and eps = 0
    runs, steps, eps, k, alpha = 2000, global_steps, 0.0, 10, 'avg'
    initval = 0
    r = run_upper_confidence_bound(qstar, initval, runs, steps, eps, k, alpha, c=2)
    a = compute_average_reward(r, runs, steps)
    o = count_optimal_action_selection(qstar, r, runs, steps)
    plots.append(([runs, steps, eps, k, alpha], [a, o], runs_list[6]))


    t=[]
    for a,b,c in plots:
        x = np.linspace(0, a[1], a[1])
        t.append((x, b[0], c))
    plot_lines(t, 'Average Rewards', 'x', 'y')

    t=[]
    for a,b,c in plots:
        x = np.linspace(0, a[1], a[1])
        t.append((x, b[1], c))
    plot_lines(t, 'Optimal Action %', 'x', 'y')