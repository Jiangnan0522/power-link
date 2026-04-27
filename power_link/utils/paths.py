"""Bidirectional Dijkstra and Yen-style k-shortest-paths.

Adapted from NetworkX's ``simple_paths`` module so that we can run the search
on a DGL graph without first materialising it as a NetworkX graph (which is
prohibitive for large KGs).
"""

from heapq import heappop, heappush
from itertools import count


def get_neg_path_score_func(g, weight, exclude_node=None):
    """Return ``score(u, v)`` for the shortest-path search.

    Edge weight is ``1 / g.edata[weight][edge]`` — Dijkstra wants a *cost*,
    while the explainer's mask is a *probability*. ``exclude_node`` is kept
    for API compatibility with the NetworkX implementation.
    """
    if exclude_node is None:
        exclude_node = []
    u, v = g.edges()
    eweights = g.edata[weight].reciprocal()
    neg_path_score_map = {edge: eweights[i] for i, edge in enumerate(zip(u.tolist(), v.tolist()))}

    def neg_path_score_func(u, v):
        return neg_path_score_map[(u, v)]

    return neg_path_score_func


def bidirectional_dijkstra(g, src_nid, tgt_nid, weight=None, ignore_nodes=None, ignore_edges=None):
    """Shortest path from ``src_nid`` to ``tgt_nid`` using bidirectional Dijkstra."""
    if src_nid == tgt_nid:
        return (0, [src_nid])

    src, tgt = g.edges()

    def Gpred(i):
        return src[tgt == i].tolist()

    def Gsucc(i):
        return tgt[src == i].tolist()

    if ignore_nodes:
        def filter_iter(nodes):
            def iterate(v):
                for w in nodes(v):
                    if w not in ignore_nodes:
                        yield w
            return iterate

        Gpred = filter_iter(Gpred)
        Gsucc = filter_iter(Gsucc)

    if ignore_edges:
        def filter_pred_iter(pred_iter):
            def iterate(v):
                for w in pred_iter(v):
                    if (w, v) not in ignore_edges:
                        yield w
            return iterate

        def filter_succ_iter(succ_iter):
            def iterate(v):
                for w in succ_iter(v):
                    if (v, w) not in ignore_edges:
                        yield w
            return iterate

        Gpred = filter_pred_iter(Gpred)
        Gsucc = filter_succ_iter(Gsucc)

    push = heappush
    pop = heappop
    dists = [{}, {}]
    paths = [{src_nid: [src_nid]}, {tgt_nid: [tgt_nid]}]
    fringe = [[], []]
    seen = [{src_nid: 0}, {tgt_nid: 0}]
    c = count()
    push(fringe[0], (0, next(c), src_nid))
    push(fringe[1], (0, next(c), tgt_nid))
    neighs = [Gsucc, Gpred]
    finalpath = []
    finaldist = 0
    dir = 1
    if not weight:
        weight = lambda u, v: 1

    while fringe[0] and fringe[1]:
        dir = 1 - dir
        (dist, _, v) = pop(fringe[dir])
        if v in dists[dir]:
            continue
        dists[dir][v] = dist
        if v in dists[1 - dir]:
            return (finaldist, finalpath)

        for w in neighs[dir](v):
            if dir == 0:
                minweight = weight(v, w)
                vwLength = dists[dir][v] + minweight
            else:
                minweight = weight(w, v)
                vwLength = dists[dir][v] + minweight

            if w in dists[dir]:
                if vwLength < dists[dir][w]:
                    raise ValueError("Contradictory paths found: negative weights?")
            elif w not in seen[dir] or vwLength < seen[dir][w]:
                seen[dir][w] = vwLength
                push(fringe[dir], (vwLength, next(c), w))
                paths[dir][w] = paths[dir][v] + [w]
                if w in seen[0] and w in seen[1]:
                    totaldist = seen[0][w] + seen[1][w]
                    if finalpath == [] or finaldist > totaldist:
                        finaldist = totaldist
                        revpath = paths[1][w][:]
                        revpath.reverse()
                        finalpath = paths[0][w] + revpath[1:]
    raise ValueError("No paths found")


class PathBuffer:
    """Sorted, deduplicated heap used by the k-shortest-paths search."""

    def __init__(self):
        self.paths = set()
        self.sortedpaths = list()
        self.counter = count()

    def __len__(self):
        return len(self.sortedpaths)

    def push(self, cost, path):
        hashable_path = tuple(path)
        if hashable_path not in self.paths:
            heappush(self.sortedpaths, (cost, next(self.counter), path))
            self.paths.add(hashable_path)

    def pop(self):
        (cost, num, path) = heappop(self.sortedpaths)
        hashable_path = tuple(path)
        self.paths.remove(hashable_path)
        return path


def k_shortest_paths_generator(g, src_nid, tgt_nid, weight=None, k=5,
                               ignore_nodes_init=None, ignore_edges_init=None):
    """Yield up to ``k`` simple paths from ``src_nid`` to ``tgt_nid`` (Yen-style)."""
    if not weight:
        weight = lambda u, v: 1

    def length_func(path):
        return sum(weight(u, v) for (u, v) in zip(path, path[1:]))

    listA = list()
    listB = PathBuffer()
    prev_path = None
    while not prev_path or len(listA) < k:
        if not prev_path:
            length, path = bidirectional_dijkstra(g, src_nid, tgt_nid, weight, ignore_nodes_init, ignore_edges_init)
            listB.push(length, path)
        else:
            ignore_nodes = set(ignore_nodes_init) if ignore_nodes_init else set()
            ignore_edges = set(ignore_edges_init) if ignore_edges_init else set()
            for i in range(1, len(prev_path)):
                root = prev_path[:i]
                root_length = length_func(root)
                for path in listA:
                    if path[:i] == root:
                        ignore_edges.add((path[i - 1], path[i]))
                try:
                    length, spur = bidirectional_dijkstra(g, root[-1], tgt_nid,
                                                          ignore_nodes=ignore_nodes,
                                                          ignore_edges=ignore_edges,
                                                          weight=weight)
                    path = root[:-1] + spur
                    listB.push(root_length + length, path)
                except ValueError:
                    pass
                ignore_nodes.add(root[-1])

        if listB:
            path = listB.pop()
            yield path
            listA.append(path)
            prev_path = path
        else:
            break


def k_shortest_paths_with_max_length(g, src_nid, tgt_nid, weight=None, k=5,
                                     max_length=None, ignore_nodes=None, ignore_edges=None):
    """Up to ``k`` simple paths from ``src_nid`` to ``tgt_nid`` of length ``≤ max_length``."""
    path_generator = k_shortest_paths_generator(g, src_nid, tgt_nid,
                                                weight=weight, k=k,
                                                ignore_nodes_init=ignore_nodes,
                                                ignore_edges_init=ignore_edges)
    try:
        if max_length:
            paths = [path for path in path_generator if len(path) <= max_length + 1]
        else:
            paths = list(path_generator)
    except ValueError:
        paths = [[]]
    return paths
