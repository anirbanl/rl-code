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

qest=np.zeros(k)
rewardlist=[]
for run in range(runs):
    l=[]
    qest=np.zeros(k)
    count=np.zeros(k)
    rewardsum=np.zeros(k)
    action=np.random.randint(0,k)
    count[action]+=1
    qvalues=qstar[run]
    reward=np.random.normal(qvalues[action],1)
    rewardsum[action]+=reward
    l.append((action,reward))
    qest[action]=rewardsum[action]*1.0/count[action]
    for step in range(steps-1):
        action=np.argmax(qest)
        count[action]+=1
        reward=np.random.normal(qvalues[action],1)
        rewardsum[action]+=reward
        l.append((action,reward))
        qest[action]=rewardsum[action]*1.0/count[action]
    rewardlist.append(l)

rewardarray=np.zeros((runs,steps),dtype=np.float)
for run in range(runs):
    for step in range(steps):
        rewardarray[run][step]=rewardlist[run][step][1]
avgreward=np.average(rewardarray,axis=0)

x=np.linspace(0,1000,1000)
t=[]
t.append((x,avgreward,'greedy'))
plot_lines(t, 'Average Rewards', 'x','y')

countarray=np.zeros((runs,steps))
for run in range(runs):
    opt_action = np.argmax(qstar[run])
    for step in range(steps):
        countarray[run][step]=int(opt_action==rewardlist[run][step][0])
opt_action_pc = np.sum(countarray,axis=0)/len(rewardlist)

x=np.linspace(0,1000,1000)
t=[]
t.append((x,opt_action_pc,'greedy'))
plot_lines(t, 'Optimal Action %', 'x','y')

