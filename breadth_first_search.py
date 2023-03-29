# import networkx as nx
# import copy
# from networkx.algorithms.traversal import breadth_first_search as bfs
from collections import deque
from operator import itemgetter

# __all__ = ['bfs_edges', 'bfs_tree']
__all__ = ['bfs_edges']


def generic_bfs_edges(G, label, source, neighbors=None, depth_limit=None):
    
    visited = {source}
    if depth_limit is None:
        depth_limit = len(G)

    neigh = list(neighbors(source))

    neighbor_list = []
    for nei in neigh:
        neighbor_dict = {}
        neighbor_dict['vertex'] = int(nei)
        neighbor_dict['label'] = label[int(nei)]
        nbrs = list(neighbors(nei))
        tree = ''
        for nbr in nbrs:
            if nbr not in visited and nbr not in neigh:
                tree += str(label[int(nbr)])
        neighbor_dict['children'] = sorted(tree)
        neighbor_list.append(neighbor_dict)

    neighbor_list_sorted = sorted(neighbor_list, key=itemgetter('label', 'children'))
    sortedneighbor = []
    for neighbor_dict_sorted in neighbor_list_sorted:
        vertex = neighbor_dict_sorted['vertex']
        sortedneighbor.append(vertex)

    queue = deque([(source, depth_limit, iter(sortedneighbor))])

    while queue:
        parent, depth_now, children = queue[0]
        try:
            child = next(children)
            if child not in visited:
                yield parent, child
                visited.add(child)
                if depth_now > 1:
                    chil = list(neighbors(child))
                    child_list = []

                    for chi in chil:
                        child_dict = {}
                        child_dict['vertex'] = int(chi)
                        child_dict['label'] = label[int(chi)]
                        nbrs = list(neighbors(chi))
                        tree = ''
                        for nbr in nbrs:
                            if nbr not in visited and nbr not in chil:
                                tree += str(label[int(nbr)])
                        child_dict['children'] = sorted(tree)
                        child_list.append(child_dict)

                    child_list_sorted = sorted(child_list, key=itemgetter('label', 'children'))
                    sortedchild = []
                    for child_dict_sorted in child_list_sorted:
                        vertex = child_dict_sorted['vertex']
                        sortedchild.append(vertex)

                    queue.append((child, depth_now - 1, iter(sortedchild)))
        except StopIteration:
            queue.popleft()


def bfs_edges(G, label, source, reverse=False, depth_limit=None):
    
    if reverse and G.is_directed():
        successors = G.predecessors
    else:
        successors = G.neighbors
    # TODO In Python 3.3+, this should be `yield from ...`
    for e in generic_bfs_edges(G, label, source, successors, depth_limit):
        yield e
