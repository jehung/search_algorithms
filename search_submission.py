# coding=utf-8
"""
This file is your main submission that will be graded against. Only copy-paste
code on the relevant classes included here. Do not add any classes or functions
to this file that are not part of the classes that we want.
"""

from __future__ import division

import heapq
import pickle
import os
from math import sqrt
#import networkx as nx
#import matplotlib.pyplot as plt


class PriorityQueue(object):
    """A queue structure where each element is served in order of priority.

    Elements in the queue are popped based on the priority with higher priority
    elements being served before lower priority elements.  If two elements have
    the same priority, they will be served in the order they were added to the
    queue.

    Traditionally priority queues are implemented with heaps, but there are any
    number of implementation options.

    (Hint: take a look at the module heapq)

    Attributes:
        queue (list): Nodes added to the priority queue.
        current (int): The index of the current node in the queue.
    """

    def __init__(self):
        """Initialize a new Priority Queue.
        """

        self.queue = []
        self.value = 0


    def pop(self):
        """Pop top priority node from queue.
        Top priority is defined as the lowest self.current value for now.

        Returns:
            The node with the highest priority.
        """

        nodes_sorted = sorted(self.queue)
        popped = heapq.heappop(nodes_sorted)
        self.queue = nodes_sorted
        return popped


    def remove(self, node_index):
        """Remove the samllest node from the queue.

        This method can be used in ucs.

        Args:
            node_id (int): Index of node in queue.
        """
        self.queue.pop(node_index)
        return self.queue


    def dequeue(self):
        self.queue.pop()


    def __iter__(self):
        """Queue iterator.
        """

        return iter(sorted(self.queue))


    def __str__(self):
        """Priority Queuer to string.
        """

        return 'PQ:%s' % self.queue


    def append(self, (value, node)):
        """Append a node to the queue.

        Args:
            node: Comparable Object to be added to the priority queue.
        """
        heapq.heappush(self.queue, (value, node))


    def __contains__(self, key):
        """Containment Check operator for 'in'

        Args:
            key: The key to check for in the queue.

        Returns:
            True if key is found in queue, False otherwise.
        """

        return key in [n[-1] for _, n in self.queue]


    def __eq__(self, other):
        """Compare this Priority Queue with another Priority Queue.
s
        Args:
            other (PriorityQueue): Priority Queue to compare against.

        Returns:
            True if the two priority queues are equivalent.
        """

        return self == other


    def size(self):
        """Get the current size of the queue.

        Returns:
            Integer of number of items in queue.
        """

        return len(self.queue)


    def clear(self):
        """Reset queue to empty (no nodes).
        """

        self.queue = []


    def top(self):
        """Get the top item in the queue, but not to remove.

        Returns:
            The first item stored in the queue.
        """

        return self.queue[0]


    def find(self, key):
        all_keys = [key for _, key in self.queue]
        for k in all_keys:
            if key == k[-1]:
                return all_keys.index(k)

        return None


def breadth_first_search(graph, start, goal):
    """Implement breadth-first-search.

    See README.md for exercise description.

    Args:
        graph (explorable_graph): Undirected graph to search.
        start (str): Key for the start node.
        goal (str): Key for the end node.

    Returns:
        The best path as a list from the start and goal nodes (including both).

    Note: recall that BFS guarantees finding the shortest path, not the cheapest path
    """
    depth = 1
    if start == goal:
        return []
    frontier = [([start], start)]
    print('frontier.queue', frontier)
    while frontier:
        (path, node) = frontier.pop(0)
        print(path, node)
        print(graph.neighbors(node))
        for neighbor in graph.neighbors(node):
            if neighbor not in path:
                if neighbor == goal:
                    return path + [neighbor]
                else:
                    frontier.append((path+[neighbor], neighbor))
    return None


def null_heuristic(graph, v, goal):
    """Null heuristic used as a base line.

    Args:
        graph (explorable_graph): Undirected graph to search.
        v (str): Key for the node to calculate from.
        goal (str): Key for the end node to calculate to.

    Returns:
        0
    """

    return 0


def uniform_cost_search(graph, start, goal):
    """Implement uniform_cost_search (i.e., ucs, cheapest-first).

    See README.md for exercise description.

    Args:
        graph (explorable_graph): Undirected graph to search.
        start (str): Key for the start node.
        goal (str): Key for the end node.

    Returns:
        The best path as a list from the start and goal nodes (including both).
    """
    if start == goal:
        return []
    frontier = PriorityQueue()

    frontier.append((0, [start]))
    explored = set()
    while frontier:
        (cost, node) = frontier.pop()
        if node[-1] == goal:
            return node
        explored.add(node[-1])
        for neighbor in graph.neighbors(node[-1]):
            found_in_frontier = frontier.find(neighbor)

            print('nei', neighbor)
            if neighbor not in explored and found_in_frontier is None:
                frontier.append((cost+graph[node[-1]][neighbor]['weight'], node+[neighbor]))
                print('part a', frontier.queue)
            elif found_in_frontier is not None and found_in_frontier:
                frontier.remove(found_in_frontier)
                frontier.append((cost + graph[node[-1]][neighbor]['weight'], node + [neighbor]))
                print('part b', frontier.queue)

    return None



def null_heuristic(graph, v, goal):
    """Null heuristic used as a base line.

    Args:
        graph (explorable_graph): Undirected graph to search.
        v (str): Key for the node to calculate from.
        goal (str): Key for the end node to calculate to.

    Returns:
        0
    """

    return 0


def euclidean_dist_heuristic(graph, v, goal):
    """Implement the euclidean distance heuristic.

    See README.md for description.

    Args:
        graph (explorable_graph): Undirected graph to search.
        v (str): Key for the node to calculate from.
        goal (str): Key for the end node to calculate to.

    Returns:
        Euclidean distance between `v` node and `goal` node as a list.
    """
    v_pos = graph.node[v]['pos']
    goal_pos = graph.node[goal]['pos']
    #print (v_pos[0]-goal_pos[0])**2 + (v_pos[1]-goal_pos[1])**2
    return sqrt((v_pos[0]-goal_pos[0])**2 + (v_pos[1]-goal_pos[1])**2)


def a_star(graph, start, goal, heuristic=euclidean_dist_heuristic):
    """ Implement A* algorithm.

    See README.md for description.

    Args:
        graph (explorable_graph): Undirected graph to search.
        start (str): Key for the start node.
        goal (str): Key for the end node.
        heuristic: callable, a function to determine distance heuristic.
            Default: euclidean_dist_heuristic.

    Returns:
        The best path as a list from the start and goal nodes (including both).
    """
    if start == goal:
        return []
    frontier = PriorityQueue()

    d = heuristic(graph, start, goal)
    frontier.append((heuristic(graph,start,goal), [start]))
    explored = set()
    while frontier:
        (cost, node) = frontier.pop()
        if node[-1] == goal:
            return node
        explored.add(node[-1])
        for neighbor in graph.neighbors(node[-1]):
            found_in_frontier = frontier.find(neighbor)

            print('nei', neighbor)
            print('explored, explored')
            d = heuristic(graph, neighbor, goal) - heuristic(graph, node[-1], goal)
            if neighbor not in explored and found_in_frontier is None:
                frontier.append((cost+graph[node[-1]][neighbor]['weight']+d, node+[neighbor]))
                print('part a', frontier.queue)
            elif found_in_frontier is not None:
                frontier.remove(found_in_frontier)
                frontier.append((cost + graph[node[-1]][neighbor]['weight']+d, node + [neighbor]))
                print('part b', frontier.queue)

    return None






def bidirectional_ucs(graph, start, goal, heuristic=euclidean_dist_heuristic):
    """Bidirectional Search.

    See README.md for exercise description.

    Args:
        graph (explorable_graph): Undirected graph to search.
        start (str): Key for the start node.
        goal (str): Key for the end node.

    Returns:
        The best path as a list from the start and goal nodes (including both).
    """
    if start == goal:
        return []
    frontier_start = PriorityQueue()
    frontier_goal = PriorityQueue()
    frontier_start.append((0, [start]))
    frontier_goal.append((0, [goal]))
    explored_start = set()
    explored_goal = set()
    while frontier_start and frontier_goal:
        (cost_start, node_start) = frontier_start.pop()
        (cost_goal, node_goal) = frontier_goal.pop()
        print('node_start', node_start)
        print('node_goal', node_goal)
        if frontier_goal.find(node_start[-1]) or node_start[-1]==goal:
            return node_start
        if frontier_start.find(node_goal[-1]) or node_goal[-1] == start:
            return node_goal[::-1]
        explored_start.add(node_start[-1])
        explored_goal.add(node_goal[-1])
        for neighbor_start in graph.neighbors(node_start[-1]):
            print('nei start', neighbor_start)
            print('explored_start', explored_start)
            found_in_frontier_start = frontier_start.find(neighbor_start)
            if neighbor_start not in explored_start and found_in_frontier_start is None:
                frontier_start.append((cost_start+graph[node_start[-1]][neighbor_start]['weight'], node_start+[neighbor_start]))
                print('queue part a start', frontier_start.queue)
            elif found_in_frontier_start is not None:
                print('queue part b start', frontier_start)
                frontier_start.remove(found_in_frontier_start)
                frontier_start.append((cost_start + graph[node_start[-1]][neighbor_start]['weight'], node_start + [neighbor_start]))
                print('part b', frontier_start.queue)
                #and (cost_start+graph[node_start[-1]][neighbor_start]['weight'], neighbor_start)==frontier_start.queue[0]:
                #node_start[-1] = neighbor_start
        for neighbor_goal in graph.neighbors(node_goal[-1]):
            print('nei goal', neighbor_goal)
            print('explored_goal', explored_goal)
            found_in_frontier_goal = frontier_goal.find(neighbor_goal)
            if neighbor_goal not in explored_goal and found_in_frontier_goal is None:
                frontier_goal.append((cost_goal+graph[node_goal[-1]][neighbor_goal]['weight'], node_goal+[neighbor_goal]))
                print('queue part a start', frontier_start.queue)
            elif found_in_frontier_goal is not None:
                #frontier_start.__contains__(neighbor_start):
                frontier_goal.remove(found_in_frontier_goal)
                frontier_goal.append((cost_goal + graph[node_goal[-1]][neighbor_goal]['weight'], node_goal + [neighbor_goal]))
                print('queue part b goal', frontier_goal.queue)


    return None




def bidirectional_a_star(graph, start, goal, heuristic=euclidean_dist_heuristic):
    """Bidirectional A*.

    See README.md for exercise description.

    Args:
        graph (explorable_graph): Undirected graph to search.
        start (str): Key for the start node.
        goal (str): Key for the end node.
        heuristic: Function to determine distance heuristic.
            Default: euclidean_dist_heuristic.

    Returns:
        The best path as a list from the start and goal nodes (including both).
    """
    if start == goal:
        return []
    frontier_start = PriorityQueue()
    frontier_goal = PriorityQueue()
    frontier_start.append((heuristic(graph,start,goal), [start]))
    frontier_goal.append((heuristic(graph,start,goal), [goal]))
    explored_start = set()
    explored_goal = set()
    while frontier_start and frontier_goal:
        (cost_start, node_start) = frontier_start.pop()
        (cost_goal, node_goal) = frontier_goal.pop()
        print('node_start', node_start)
        print('node_goal', node_goal)
        if frontier_goal.find(node_start[-1]) or node_start[-1]==goal:
            return node_start
        if frontier_start.find(node_goal[-1]) or node_goal[-1] == start:
            return node_goal[::-1]
        explored_start.add(node_start[-1])
        explored_goal.add(node_goal[-1])
        for neighbor_start in graph.neighbors(node_start[-1]):
            print('nei start', neighbor_start)
            print('explored_start', explored_start)
            d = heuristic(graph, neighbor_start, goal) - heuristic(graph, node_start[-1], goal)
            found_in_frontier_start = frontier_start.find(neighbor_start)
            if neighbor_start not in explored_start and found_in_frontier_start is None:
                frontier_start.append((cost_start+graph[node_start[-1]][neighbor_start]['weight']+d, node_start+[neighbor_start]))
                print('queue part a start', frontier_start.queue)
            elif found_in_frontier_start is not None:
                print('queue part b start', frontier_start)
                frontier_start.remove(found_in_frontier_start)
                frontier_start.append((cost_start + graph[node_start[-1]][neighbor_start]['weight']+d, node_start + [neighbor_start]))
                print('part b', frontier_start.queue)
                #and (cost_start+graph[node_start[-1]][neighbor_start]['weight'], neighbor_start)==frontier_start.queue[0]:
                #node_start[-1] = neighbor_start
        for neighbor_goal in graph.neighbors(node_goal[-1]):
            print('nei goal', neighbor_goal)
            print('explored_goal', explored_goal)
            d = heuristic(graph, neighbor_goal, start) - heuristic(graph, node_goal[-1], start)
            found_in_frontier_goal = frontier_goal.find(neighbor_goal)
            if neighbor_goal not in explored_goal and found_in_frontier_goal is None:
                frontier_goal.append((cost_goal+graph[node_goal[-1]][neighbor_goal]['weight']+d, node_goal+[neighbor_goal]))
                print('queue part a start', frontier_start.queue)
            elif found_in_frontier_goal is not None:
                #frontier_start.__contains__(neighbor_start):
                frontier_goal.remove(found_in_frontier_goal)
                frontier_goal.append((cost_goal + graph[node_goal[-1]][neighbor_goal]['weight']+d, node_goal + [neighbor_goal]))
                print('queue part b goal', frontier_goal.queue)


    return None








# Extra Credit: Your best search method for the race
#
def load_data(pickfile):
    """Loads data from data.pickle and return the data object that is passed to
    the custom_search method.

    Will be called only once. Feel free to modify.

    Returns:
         The data loaded from the pickle file.
    """

    pickle_file_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), pickfile)
    print(pickle_file_path)
    data = pickle.load(open(pickle_file_path, 'rb'))
    print(data)
    return data


def custom_search(graph, start, goal, data=None):
    """Race!: Implement your best search algorithm here to compete against the
    other student agents.

    See README.md for exercise description.

    Args:
        graph (explorable_graph): Undirected graph to search.
        start (str): Key for the start node.
        goal (str): Key for the end node.
        data :  Data used in the custom search.
            Default: None.

    Returns:
        The best path as a list from the start and goal nodes (including both).
    """

    # TODO: finish this function!
    raise NotImplementedError


'''
#'graph' is a graph object
graph = load_data('romania_graph.pickle')
print(graph.adj)
nx.draw(graph)
plt.show()


#start = random.choice(graph.nodes())
#end = random.choice(graph.nodes())
start = 'o'
end = 'n'
#start_num = random.randint(0, len(graph.nodes()))
#end_num = random.randint(0, len(graph.nodes()))
#start = graph.node[graph.nodes()[start_num]]
#end = graph.node[graph.nodes()[end_num]]
#print('start', start)
#print('end', end)


#bfs = breadth_first_search(graph, start, end)
#ucs = uniform_cost_search(graph, start, end)
#print(ucs)



euclidean_dist_heuristic(graph, start, end)
a_star = a_star(graph, start, end)
print(a_star)


#ucs = uniform_cost_search(graph, start, end)
#print(bi_ucs)
#bi_ucs = bidirectional_ucs(graph, start, end)
#print(bi_ucs)
#bia = bidirectional_a_star(graph, start, end)
#print(bia)
'''