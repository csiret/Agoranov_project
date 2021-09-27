from matplotlib.pyplot import axis
from numpy import linalg
from numpy.core.fromnumeric import shape
from numpy.lib.function_base import average
from numpy.linalg.linalg import norm
import pandas as pd
import numpy as np

"""Calculating herding measures, average velocity, cosine herding"""


#average velocity herding measure
def average_velocity(community, t, significant_time, investor_no):
    summed_vectors = np.zeros(shape=(community.shape[0], investor_no))
    if t < significant_time:
        for time in range(t):
            summed_vectors += community[:, :investor_no, time]
    else:
        for time in range(t-significant_time, t):
            summed_vectors += community[:, :investor_no, time]
    return np.linalg.norm(np.average(summed_vectors, axis=1))/np.average(np.linalg.norm(summed_vectors, axis=0))

#cosine hearding measure, contemporaneous, following and leading
def cont_cos_herding(i, community, t):    
    sum = np.zeros(shape=(community.shape[0]))
    n = community.shape[1]
    investor = community[:, i, t]
    for j in range(n):
        if j == i:
            continue
        sum += community[:, j, t]
    benchmark = np.average(sum)
    return np.dot(investor, benchmark)/(np.linalg.norm(investor)*np.linalg.norm(benchmark))

def fol_cos_herding(i, community, t, time_step):
    sum = np.zeros(shape=(community.shape[0]))
    n = community.shape[1]
    investor = community[:, i, t]
    for j in range(n):
        if j == i:
            continue
        sum += community[:, j, t-time_step]
    benchmark = sum/(n-1)
    return np.dot(investor, benchmark)/(np.linalg.norm(investor)*np.linalg.norm(benchmark))

def lead_cos_herding(i, community, t, time_step):
    n = community.shape[1]
    sum = np.zeros(shape=(community.shape[0]))
    investor = community[:, i, t-time_step]
    for j in range(n):
        if j == i:
            continue
        sum += community[:, j, t]
    benchmark = sum/(n-1)
    return np.dot(investor, benchmark)/(np.linalg.norm(investor)*np.linalg.norm(benchmark))

#Lakonishok herding measure, industry imbalance 
def herding(industry, community, t, significant_time):
    growth_rate = (np.sum(community[industry, :, t])*(t-significant_time))/np.sum(community[industry, :, significant_time:t+1], axis = (0,1))
    benchmark_growth_rate = (np.sum(community[:, :, t], axis=(0,1))*(t-significant_time))/np.sum(community[:, :, significant_time:t+1], axis=(0,1,2))
    return abs(growth_rate - benchmark_growth_rate)

#same as cosine leading measure but taking into account only one investor, effectively similarity across a certain time step
def self_similarity(community, investor, t, time_step):
    investor_movement = community[:, investor, t]
    investor_benchmark = community[:, investor, t-time_step]
    return np.dot(investor_movement, investor_benchmark)/(np.linalg.norm(investor_movement)*np.linalg.norm(investor_benchmark))

#difference of investment counts for one investor across a certain time step, trend measure for sectors
def self_difference(community, investor, t, time_step):
    investor_movement = community[:, investor, t]
    investor_benchmark = community[:, investor, t-time_step]
    return investor_movement-investor_benchmark


#same as lead_cos_herding but for system with only top investors
def lead_cos_herding_check(i, community, t, time_step, leaders):
    n = community.shape[1]
    sum = np.zeros(shape=(community.shape[0]))
    investor = community[:, i, t-time_step]
    for j in range(n):
        if j in leaders:
            continue
        sum += community[:, j, t]
    benchmark = sum/(n-1)
    return np.dot(investor, benchmark)/(np.linalg.norm(investor)*np.linalg.norm(benchmark))        