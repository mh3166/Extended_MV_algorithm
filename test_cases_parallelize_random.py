#!/apps/anaconda-2.3.0/bin/python

import csv
import networkx as nx
import mv_algorithm_fast_test as mv0f
import os
import sys
import time
import numpy as np
# sys.setrecursionlimit(100000)

edge_list = []

def writeLog(fil, table):
    c1 = csv.writer(fil)
    for val in table:
        c1.writerow(val)

test_arr = []
ed_array = [2**(i/16.0) for i in range(0,48)]
n_exp_array = [19,20]

for n_size in n_exp_array:
    for d in ed_array:
        for t in range(5,10):
            test_arr.append([d,n_size,t])

param = test_arr[int(os.environ['SGE_TASK_ID'])-1]
# param = test_arr[1000]
# param = [2**(23/16.0),20,12]
print param
results = []
node_exp = param[1]
n = 2 ** (node_exp)
ed = param[0]

h = nx.fast_gnp_random_graph(n, ed / float(n))
# h = nx.Graph()
# h.add_nodes_from(range(n ** 2))
# p = 1 - (1 - ed) ** 0.5
# for i in range(n):
#     for j in range(n):
#         # print i*n+j
#         p1 = np.random.uniform(0, 1)
#         p2 = np.random.uniform(0, 1)
#         p3 = np.random.uniform(0, 1)
#         p4 = np.random.uniform(0, 1)
#         if p1 < p:
#             h.add_edge(i * n + j, ((i + 1) % n) * n + j % n)
#         if p2 < p:
#             h.add_edge(i * n + j, ((i - 1) % n) * n + j % n)
#         if p3 < p:
#             h.add_edge(i * n + j, (i % n) * n + (j + 1) % n)
#         if p4 < p:
#             h.add_edge(i * n + j, (i % n) * n + (j - 1) % n)
len_edges = len(h.edges())
deg_arr = []
for n1 in h.nodes():
    deg_arr.append(h.degree(n1))

start_time = time.clock()
l0, pid0, pm0 = mv0f.mv_max_cardinality(h, 1, True, 1)
print len(l0)/2.0
l0_time = time.clock() - start_time
print 'solve time: ' + str(l0_time)
print ' '

start_time = time.clock()
l1, pid1, pm1 = mv0f.mv_max_cardinality(h, 1, True, 1)
print len(l1)/2.0
l1_time = time.clock() - start_time
print 'solve time: ' + str(l1_time)
print ' '

start_time = time.clock()
l2, pid2, pm2 = mv0f.mv_max_cardinality(h, 1, False, 100)
print len(l2)/2.0
l2_time = time.clock() - start_time
print 'solve time: ' + str(l2_time)
print ' '

start_time = time.clock()
l3, pid3, pm3 = mv0f.mv_max_cardinality(h, 1, True, 100)
print len(l3)/2.0
l3_time = time.clock() - start_time
print 'solve time: ' + str(l3_time)
print ' '
results.append(['expected_degree','node_exponent','N','E',
                'phases_0','phases_1','phases_2','phases_3',
                'time_0','time_1','time_2','time_3',
                'percent_matched_0', 'percent_matched_1','percent_matched_2','percent_matched_3'])
results.append([ed, node_exp, n, len_edges, pid0, pid1, pid2, pid3, l0_time, l1_time, l2_time, l3_time, pm0, pm1, pm2, pm3])
print results
fil = open(os.getcwd() + "/results_grid_4_" + str(node_exp) + "_" + str(ed) + "_" + str(param[2]) + ".csv", "wb")
writeLog(fil, results)
