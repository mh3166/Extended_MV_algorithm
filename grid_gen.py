import networkx as nx
import numpy as np
import mv_algorithm as mv0
import mv_algorithm_fast_bridge_order_test as mv0f
import time
import os
import csv
import sys

def writeLog(fil, table):
    c1 = csv.writer(fil)
    for val in table:
        c1.writerow(val)

node_exp = 10
# node_exp = int(sys.argv[1])
n = 2**node_exp
h = nx.Graph()
h.add_nodes_from(range(n**2))
# p = 0.2
ed = 0.76
p = 1 - (1 - ed) ** 0.5
trials = 1
results = []


for t in range(trials):

    for i in range(n):
        for j in range(n):
            # print i*n+j
            setp = np.random.uniform(0, 1)
            p1 = np.random.uniform(0, 1)
            p2 = np.random.uniform(0, 1)
            p3 = np.random.uniform(0, 1)
            p4 = np.random.uniform(0, 1)
            # if setp < .25:
            #     if p2 < p:
            #         h.add_edge(i * n + j, ((i - 1) % n) * n + j % n)
            #     if p3 < p:
            #         h.add_edge(i * n + j, (i % n) * n + (j + 1) % n)
            #     if p4 < p:
            #         h.add_edge(i * n + j, (i % n) * n + (j - 1) % n)
            # elif 0.25 <= setp <0.5:
            #     if p1 < p:
            #         h.add_edge(i * n + j, ((i + 1) % n) * n + j % n)
            #     if p3 < p:
            #         h.add_edge(i * n + j, (i % n) * n + (j + 1) % n)
            #     if p4 < p:
            #         h.add_edge(i * n + j, (i % n) * n + (j - 1) % n)
            # elif 0.5 <= setp < 0.75:
            #     if p1 < p:
            #         h.add_edge(i * n + j, ((i + 1) % n) * n + j % n)
            #     if p2 < p:
            #         h.add_edge(i * n + j, ((i - 1) % n) * n + j % n)
            #     if p4 < p:
            #         h.add_edge(i * n + j, (i % n) * n + (j - 1) % n)
            # else:
            #     if p1 < p:
            #         h.add_edge(i * n + j, ((i + 1) % n) * n + j % n)
            #     if p2 < p:
            #         h.add_edge(i * n + j, ((i - 1) % n) * n + j % n)
            #     if p3 < p:
            #         h.add_edge(i * n + j, (i % n) * n + (j + 1) % n)
            # if i == 0:
            #     if j == 0:
            #         if p1 < p:
            #             h.add_edge(i * n + j, ((i + 1) % n) * n + j % n)
            #         if p3 < p:
            #             h.add_edge(i * n + j, (i % n) * n + (j + 1) % n)
            #     elif j == n-1:
            #         if p1 < p:
            #             h.add_edge(i * n + j, ((i + 1) % n) * n + j % n)
            #         if p4 < p:
            #             h.add_edge(i * n + j, (i % n) * n + (j - 1) % n)
            #     else:
            #         if p1 < p:
            #             h.add_edge(i * n + j, ((i + 1) % n) * n + j % n)
            #         if p3 < p:
            #             h.add_edge(i * n + j, (i % n) * n + (j + 1) % n)
            #         if p4 < p:
            #             h.add_edge(i * n + j, (i % n) * n + (j - 1) % n)
            # elif i == n-1:
            #     if j == 0:
            #         if p1 < p:
            #             h.add_edge(i * n + j, ((i - 1) % n) * n + j % n)
            #         if p3 < p:
            #             h.add_edge(i * n + j, (i % n) * n + (j + 1) % n)
            #     elif j == n - 1:
            #         if p1 < p:
            #             h.add_edge(i * n + j, ((i - 1) % n) * n + j % n)
            #         if p4 < p:
            #             h.add_edge(i * n + j, (i % n) * n + (j - 1) % n)
            #     else:
            #         if p1 < p:
            #             h.add_edge(i * n + j, ((i - 1) % n) * n + j % n)
            #         if p3 < p:
            #             h.add_edge(i * n + j, (i % n) * n + (j + 1) % n)
            #         if p4 < p:
            #             h.add_edge(i * n + j, (i % n) * n + (j - 1) % n)
            # else:
            #     if j == 0:
            #         if p1 < p:
            #             h.add_edge(i * n + j, ((i + 1) % n) * n + j % n)
            #         if p2 < p:
            #             h.add_edge(i * n + j, ((i - 1) % n) * n + j % n)
            #         if p3 < p:
            #             h.add_edge(i * n + j, (i % n) * n + (j + 1) % n)
            #     elif j == n - 1:
            #         if p1 < p:
            #             h.add_edge(i * n + j, ((i + 1) % n) * n + j % n)
            #         if p2 < p:
            #             h.add_edge(i * n + j, ((i - 1) % n) * n + j % n)
            #         if p4 < p:
            #             h.add_edge(i * n + j, (i % n) * n + (j - 1) % n)
            #     else:
            #         if p1 < p:
            #             h.add_edge(i * n + j, ((i + 1) % n) * n + j % n)
            #         if p2 < p:
            #             h.add_edge(i * n + j, ((i - 1) % n) * n + j % n)
            #         if p3 < p:
            #             h.add_edge(i * n + j, (i % n) * n + (j + 1) % n)
            #         if p4 < p:
            #             h.add_edge(i * n + j, (i % n) * n + (j - 1) % n)

            if p1 < p:
                h.add_edge(i * n + j, ((i + 1) % n) * n + j % n)
            if p2 < p:
                h.add_edge(i * n + j, ((i - 1) % n) * n + j % n)
            if p3 < p:
                h.add_edge(i * n + j, (i % n) * n + (j + 1) % n)
            if p4 < p:
                h.add_edge(i * n + j, (i % n) * n + (j - 1) % n)

    print nx.is_bipartite(h)
    start_time = time.clock()
    print len(nx.bipartite.maximum_matching(h))/2.0
    print 'solve_time: ' + str(time.clock() - start_time)
    len_edges = len(h.edges())
    print 'edges: ' + str(len_edges)
    deg_arr = []
    for n in h.nodes():
        deg_arr.append(h.degree(n))
    print 'average degree: ' + str(np.mean(deg_arr))
    print np.std(deg_arr)
    cc = nx.number_connected_components(h)
    print cc
    start_time = time.clock()
    print 'bfs time: ' + str(time.clock() - start_time)
    l0, pid0 = mv0f.mv_max_cardinality(h, 1, True, 100)
    print len(l0)/2.0
    l0_time = time.clock() - start_time
    print 'solve time: ' + str(l0_time)
    print ' '

    start_time = time.clock()
    l1, pid1 = mv0f.mv_max_cardinality(h, 1, False, 100)
    print len(l1)/2.0
    l1_time = time.clock() - start_time
    print 'solve time: ' + str(l1_time)
    results.append([ed, 20, n, len_edges, cc, pid0, pid1, l0_time, l1_time])
fil = open(os.getcwd() + "/results_grid_large_" + str(node_exp) + "_" + str(ed * 4.0) + ".csv", "wb")
writeLog(fil, results)