# Queens College
# Discrete Structures (CSCI 220)
# Winter 2024
# Assignment 11: "Graphs and Graph Algorithms"
# name Jennifer Umana
# Collaborated with Class


from random import random
import numpy as np
from matplotlib import pyplot as plt

import Assignment8 as as8
import networkx as nx
import copy


# [1] Define a function read_graph(file_name) that reads in a graph from a file in the form of an adjacency matrix.
def read_graph(file_name):
    with open(file_name) as f:
        lines = f.readlines()
        return [[1 if c == "1" else 0 for c in line.strip().split(" ")] for line in lines]


# [2] Define a function print_adj_matrix(matrix) that nicely prints a graph stored as an adjacency matrix.
def print_matrix(matrix):
    print(np.array(matrix))


# [3] Define a function adjacency_table(matrix) that converts an adjacency matrix into an adjacency table.
def adjacency_table(matrix):
    table = []
    n = len(matrix)
    for i in range(n):
        row = []
        for j in range(n):
            if matrix[i][j] == 1:
                row.append(j)
        table.append(row)
    return table


# define a function incidence matrix that provides incidence matrix for a graph
def incidence_matrix(matrix, dir):
    edges = list(edge_set(matrix, dir))
    n_edges = len(edges)
    n_vertices = len(matrix)
    inc_matrix = [[0] * n_edges for j in range(n_vertices)]
    for k in range(n_edges):
        i, j = edges[k]
        inc_matrix[i][k] = 1
        inc_matrix[j][k] = 1
    return inc_matrix


# [4] Define a function print_adj_table(table) that nicely prints a graph stored as an adjacency table.
def print_adj_table(table):
    for i in range(len(table)):
        print(i, ":", table[i])


# [5] Define a function edge_set(matrix) that converts an adjacency matrix into a list of edges.
def edge_set(matrix, dir):
    edges = set()
    for i in range(len(matrix)):
        for j in range(len(matrix)):
            if matrix[i][j] == 1 and dir or i > j:
                edges.add((i, j))
    return edges


# [6] Define a function random_graph(n, s, p) with n vertices, s if symmetric, and probability of edge p.
def random_graph(n, dir, p):
    matrix = [[0] * n for i in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            if random() < p:
                matrix[i][j] = 1
                if not dir:
                    matrix[i][j] = 1
            if dir and random() < p:
                matrix[j][i] = 1
    return matrix

# [7] Define functions that determine the properties of the relation E. (See Overview above.)
# Reflexive - if aRa is true for every a in S.
# Irreflexive - if aRa may sometimes be true but not always true for every a in S
# Anti-reflexive - if aRa is false for every a in S
# Symmetric - if aRb is true implies bRa is true for every a and b in S
# Asymmetric - if aRb is true implies bRa is false for every a and b in S
# Antisymmetric - if aRb is true implies bRa is false for every a and b in S unless a and b are equal
# Transitive - if aRb is true and bRc is true, then aRc must also be true for every a, b, c in S
# Intransitive - if aRb is true and bRc is true, then aRc is not necessarily true for every a, b, c in S
# Antitransitive - if aRb is true and bRc is true, then aRc is false for every a, b, c in S


def reflexive(matrix):
    for i in range(len(matrix)):
        if matrix[i][i] == 0:
            return False
    return True


def anti_reflexive(matrix):
    for i in range(len(matrix)):
        if matrix[i][i] == 1:
            return False
    return True


def irreflexive(matrix):
    return not reflexive(matrix) and not anti_reflexive(matrix)


def symmetric(matrix):
    for i in range(len(matrix)):
        for j in range(i+1, len(matrix)):
            if matrix[i][j] != matrix[j][i]:
                return False
    return True


def anti_symmetric(matrix):
    for i in range(len(matrix)):
        for j in range(i+1, len(matrix)):
            if matrix[i][j] != matrix[j][i] and i != j:
                return False
    return True


def asymmetric(matrix):
    for i in range(len(matrix)):
        for j in range(i+1, len(matrix)):
            if matrix[i][j] == matrix[j][i]:
                return False
    return True


def transitive(matrix):
    n = len(matrix)
    for i in range(n):
        for j in range(n):
            for k in range(n):
                if matrix[i][j] == 1 and matrix[j][k] == 1 and matrix[i][k] == 0:
                    return False
    return True


def anti_transitive(matrix):
    n = len(matrix)
    for i in range(n):
        for j in range(n):
            for k in range(n):
                if matrix[i][j] == 1 and matrix[j][k] == 1 and matrix[i][k] == 1:
                    return False
    return True


def intransitive(matrix):
    return not transitive(matrix) and not anti_transitive(matrix)


# [8] Define a function print_properties(graph) that prints the properties determined in the previous task.
def print_properties(description, matrix, properties):
    data = [[prop.__name__, str(prop(matrix))] for prop in properties]
    headers = ["Property name", "Value"]
    alignments = ["l"] * 2
    as8.print_table(description, headers, data, alignments)


# [9] Define a function print_verticies(graph) that lists each vertex and its in-degree, out-degree, and neighbors.
def print_vertex_properties(description, matrix):
    data = [[i, outdegree(matrix, i), indegree(matrix, i), neighbors(matrix, i)] for i in range(len(matrix))]
    headers = ["Vertex", "Outdegree", "Indegree", "Neighbors"]
    alignments = ["r", "r", "r", "l"]
    as8.print_table(description, headers, data, alignments)


def neighbors(matrix, v):
    return [j for j in range(len(matrix)) if matrix[v][j] == 1]


def indegree(matrix, v):
    return sum([matrix[i][v] for i in range(len(matrix))])


def outdegree(matrix, v):
    return sum([matrix[v][i] for i in range(len(matrix))])


def outdegrees(matrix):
    return [outdegree(matrix, v) for v in range(len(matrix))]


def indegrees(matrix):
    return [indegree(matrix, v) for v in range(len(matrix))]


# [10] Define a function draw_graph(graph) that draws a graph and saves it as an image file. See
# https://stackoverflow.com/questions/20133479/how-to-draw-directed-graphs-using-networkx-in-python
# https://stackoverflow.com/questions/74312314/draw-a-directed-graph-in-python
def draw_graph(edges, directed, filename):
    G = nx.DiGraph()
    G.add_edges_from(edges)
    val_map = {'A': 1.0, 'D': 0.5714285714285714, 'H': 0.0}
    values = [val_map.get(node, 0.25) for node in G.nodes()]
    pos = nx.spring_layout(G)
    cmap = plt.get_cmap('jet')
    nx.draw_networkx_nodes(G, pos, cmap=cmap, node_color=values, node_size=500)
    nx.draw_networkx_labels(G, pos, font_size=12, font_color='white')
    if directed:
        nx.draw_networkx_edges(G, pos, edgelist=edges, edge_color='r', arrows=directed, arrowsize=10)
    else:
        nx.draw_networkx_edges(G, pos, edgelist=edges, edge_color='r', arrows=False)
    plt.savefig(filename)
    plt.show()


# [11] Define functions that determine if the graph is: cycle, complete, star, wheel, bipartite, connected, Eulerian
def is_connected(matrix):
    adj_table = adjacency_table(matrix)
    visited = []
    dfs_util(adj_table, 0, visited)
    return len(visited) == len(matrix)


def is_strongly_connected(matrix):
    tc = warshall_transitive_closure(matrix)
    n = len(matrix)
    return all([sum(row) == n for row in tc])


def is_eulerian(matrix):
    if not is_connected(matrix):
        return False
    n = len(matrix)
    dir = symmetric(matrix)
    if dir:
        return False
        # out_degree_odd = [outdegree(matrix, i) %2 == 1 for i in range(n)]
        # in_degrees_odd = [indegree(matrix, i) %2 == 1 for i in range(n)]
        # return not (any(out_degree_odd or any(in_degrees_odd)))

    else:
        degree_odd = [outdegree(matrix, i) % 2 == 1 for i in range(n)]
        return not any(degree_odd)


def is_cycle(matrix):
    if not is_connected(matrix):
        return False
    n = len(matrix)
    dir = symmetric(matrix)
    if dir:
        return all([outdegree(matrix, i) == 1 for i in range(n)])
    else:
        return all([outdegree(matrix, i) == 2 for i in range(n)])


def is_complete(matrix):
    n = len(matrix)
    for i in range(n):
        if matrix[i][i] == 1:
            return False
        for j in range(n):
            if i != j and matrix[i][j] == 0:
                return False
    return True


def is_star(matrix):
    n = len(matrix)
    degrees = outdegrees(matrix)
    for center in range(n):
        star = True
        if degrees[center] != n-1:
            star = False
            for i in range(n):
                if i != center and degrees[i] != 1:
                    star = False
            if star:
                return True
    return False


# [12] Define functions that traverse the graph in these two standard orderings
# Breadth-First Search (BFS)
def dfs_util(adj_table, v, visited):
    visited.append(v)
    for w in adj_table[v]:
        if w not in visited:
            dfs_util(adj_table, w, visited)


def dfs(matrix):
    adj_table = adjacency_table(matrix)
    visited = []
    for v in range(len(adj_table)):
        if v not in visited:
            dfs_util(adj_table, v, visited)
    return visited


def bfs_util(adj_table, visited, queue):
    while queue:
        v = queue.pop(0)
        for w in adj_table[v]:
            if w not in visited:
                queue.append(w)
                visited.append(w)


def bfs(matrix):
    adj_table = adjacency_table(matrix)
    visited = []
    queue = []
    for v in range(len(adj_table)):
        if v not in visited:
            queue.append(v)
            visited.append(v)
            bfs_util(adj_table, visited, queue)
    return visited


# [13] Implement Warshall's Transitive-Closure algorithms
def warshall_transitive_closure(matrix):
    n = len(matrix)
    tc = copy.deepcopy(matrix)
    for i in range(n):
        tc[i][i] = 1
    for i in range(n):
        for j in range(n):
            for k in range(n):
                if tc[i][k] == 1 and tc[k][j] == 1:
                    tc[i][j] = 1
    return tc


def do_graph(assn, description, matrix, dir):
    print(description)

    print("\nAdjacency matrix:")
    print_matrix(matrix)

    print("\nAdjacency table:")
    table = adjacency_table(matrix)
    print_adj_table(table)

    print("\nEdge set:")
    edges = edge_set(matrix, dir)
    print(edges)

    print("\nIncidence matrix:")
    inc_matrix = incidence_matrix(matrix, dir)
    print_matrix(inc_matrix)

    print("\nTransitive closure (connectivity) matrix:")
    tc_matrix = warshall_transitive_closure(matrix)
    print_matrix(tc_matrix)

    properties = [reflexive, anti_reflexive, irreflexive, symmetric, anti_symmetric, asymmetric,
                  transitive, intransitive, anti_transitive]
    print_properties("\nRelation Properties for Graph", matrix, properties)

    properties = [outdegrees, indegrees, dfs, bfs, is_connected, is_strongly_connected, is_eulerian, is_star,
                  is_complete, is_cycle]
    print_properties("\nGeneral Properties for Graph", matrix, properties)

    print_vertex_properties("\nVertex Properties for Graph", matrix)
    print()
    image_file = assn + "-" + description.replace(" ", "-") + ".png"
    draw_graph(edges, dir, image_file)


def main():
    assn = "Assignment11"
    matrix = read_graph("Graph1.txt")
    do_graph(assn, "Graph Read from Graph1.txt", matrix, True)
    matrix2 = random_graph(8, False, 0.5)
    do_graph(assn, "Random Graph of Size 8", matrix2, False)
    matrix3 = random_graph(10, True, .9)
    do_graph(assn, "Random Graph of Size 10 with p = .9", matrix3, True)


if __name__ == "__main__":
    main()
