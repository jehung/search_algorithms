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
import random
from math import sqrt
import matplotlib.pyplot as plt
import networkx as nx

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


    def remove(self):
        """Remove the samllest node from the queue.

        This method can be used in ucs.

        Args:
            node_id (int): Index of node in queue.
        """
        self.pop()
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

        return key in [n for _, n in self.queue]


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
    print('frontier.queue', frontier.queue)
    while frontier:
        (cost, node) = frontier.pop()
        if node[-1] == goal:
            return node
        explored.add(node[-1])
        for neighbor in graph.neighbors(node[-1]):
            if neighbor not in explored and not frontier.__contains__(neighbor):
                frontier.append((cost+graph[node[-1]][neighbor]['weight'], node+[neighbor]))
            elif frontier.__contains__(neighbor) and (cost+graph[node[-1]][neighbor]['weight'], neighbor)==frontier.queue[0]:
                node[-1] = neighbor

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
    print('ans', graph[start])
    frontier.append((heuristic(graph, start, goal), [start]))
    explored = set()
    print('frontier.queue', frontier.queue)
    while frontier:
        (cost, node) = frontier.pop()
        print('now', (cost, node))
        if node[-1] == goal:
            return node
        explored.add(node[-1])
        for neighbor in graph.neighbors(node[-1]):
            #print('dist r-c', heuristic(graph, 'r', 'c'))
            #print('dist t-p + p-c', heuristic(graph, 'r', 'p')+heuristic(graph, 'p', 'c'))
            #cost += graph[node[-1]][neighbor]['weight']##+heuristic(graph, neighbor, goal)
            if neighbor not in explored and not frontier.__contains__(neighbor):
                frontier.append((cost+graph[node[-1]][neighbor]['weight'], node + [neighbor]))
            elif frontier.__contains__(neighbor) and (cost+graph[node[-1]][neighbor]['weight'], neighbor) == \
                    frontier.queue[0]:
                node[-1] = neighbor
    return None



def goal_check(start_list, stop_list):
    for item1 in start_list:
        for item2 in stop_list:
            if item1 == item2:
                return True



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
    frontier_start.append((heuristic(graph, start, goal), [start]))
    frontier_goal.append((heuristic(graph, goal, start), [goal]))
    explored_start = set()
    explored_goal = set()
    while frontier_start and frontier_goal:
        (cost_start, node_start) = frontier_start.pop()
        (cost_goal, node_goal) = frontier_goal.pop()
        if frontier_goal.__contains__(node_start[-1]) or node_start[-1]==goal:
            return node_start
        if frontier_start.__contains__(node_goal[-1]) or node_goal[-1]==start:
            return node_goal[::-1]
        explored_start.add(node_start[-1])
        explored_goal.add(node_goal[-1])
        for neighbor_start in graph.neighbors(node_start[-1]):
            if neighbor_start not in explored_start and not frontier_start.__contains__(neighbor_start):
                frontier_start.append((cost_start+graph[node_start[-1]][neighbor_start]['weight'], node_start+[neighbor_start]))
            elif frontier_start.__contains__(neighbor_start) and (cost_start+graph[node_start[-1]][neighbor_start]['weight'], neighbor_start)==\
                    frontier_start.queue[0]:
                node_start[-1] = neighbor_start
        for neighbor_goal in graph.neighbors(node_goal[-1]):
            if neighbor_goal not in explored_goal and not frontier_goal.__contains__(neighbor_goal):
                frontier_goal.append((cost_goal+graph[node_goal[-1]][neighbor_goal]['weight'], node_goal+[neighbor_goal]))
            elif frontier_start.__contains__(neighbor_start) and (
                cost_goal+graph[node_goal[-1]][neighbor_goal]['weight'], neighbor_goal) == \
                    frontier_goal.queue[0]:
                node_goal[-1] = neighbor_goal


    return None




def bidirectional_a_star(graph, start, goal, heuristic=euclidean_dist_heuristic):
    """Exercise 2: Bidirectional A*.

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

    # TODO: finish this function!
    raise NotImplementedError


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



#'graph' is a graph object
graph = load_data('romania_graph.pickle')
print(graph.adj)
nx.draw(graph)
plt.show()
print('here', graph.nodes())

#start = random.choice(graph.nodes())
#end = rsandom.choice(graph.nodes())
start = 'b'
end = 'e'
#start_num = random.randint(0, len(graph.nodes()))
#end_num = random.randint(0, len(graph.nodes()))
#start = graph.node[graph.nodes()[start_num]]
#end = graph.node[graph.nodes()[end_num]]
#print('start', start)
#print('end', end)


#bfs = breadth_first_search(graph, start, end)
#ucs = uniform_cost_search(graph, start, end)




#euclidean_dist_heuristic(graph, start, end)
#a_star = a_star(graph, start, end)
#print(a_star)


bi_ucs = bidirectional_ucs(graph, start, end)
print(bi_ucs)