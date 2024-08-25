import queue
import networkx as nx
import matplotlib.pyplot as plt

print(f"Hello, World!")


def order_dfs(graph, start):
    visited = set()
    q = queue.Queue()
    q.put(start)
    while not q.empty():
        current = q.get()
        if current not in visited:
            visited.add(current)
            for neighbor in graph[current]:
                q.put(neighbor)
    return visited


def order_bfs(graph, start):
    visited = set()
    q = queue.Queue()
    q.put(start)
    while not q.empty():
        current = q.get()
        if current not in visited:
            visited.add(current)
            for neighbor in graph[current]:
                q.put(neighbor)
    return visited


