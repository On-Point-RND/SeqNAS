import itertools
from collections import Counter
import numpy as np
import torch


OPS = [
    "linear",
    "elementwise_sum",
    "elementwise_prod",
    "blend",
    "activation_tanh",
    "activation_sigm",
    "activation_leaky_relu",
]


def expand(adj, paths, outputs, valid_paths):
    """
    BFS step required for 'get_paths'

    Parameters
    ----------
    adj:
        Adjacency matrix.
    paths:
        all found paths before this step.
    outputs:
        Output nodes (where paths end).
    valid_paths:
        List of complete paths that reached output.

    Returns
    -------
    list:
        All found paths found up to this step.
    list:
        Complete paths that reached output found up to this step.
    """
    paths_new = []
    for path in paths:
        idx = path[-1]
        for i in range(len(adj)):
            if adj[idx, i] == 1:
                new_path = path + [i]
                paths_new.append(new_path)
                if new_path[-1] in outputs:
                    valid_paths.append(new_path)
    return paths_new, valid_paths


def get_paths(inputs, outputs, adj, max_length=30):
    """
    Get paths in graph given inputs, outputs and adjacency matrix.

    Parameters
    ----------
    inputs:
        Input nodes (from which paths start)
    outputs:
        Output nodes (where paths end)
    adj:
        Adjacency matrix
    max_length:
        Maximum path length limitation

    Returns
    -------
    list:
        Complete paths that start in one of the input nodes
        and end in one of the output nodes.
    """
    paths = [[node_idx] for node_idx in inputs]
    valid_paths = []
    for i in range(max_length):
        paths, valid_paths = expand(adj, paths, outputs, valid_paths)
    return valid_paths


def get_path_features(g):
    """
    Get occurence of paths in a given arch

    Parameters
    ----------
    g:
        Architecture

    Returns
    -------
    dict:
        Contains paths in string format and occurence numbers
    """
    set_keys = set(
        list(g.keys()) + list(itertools.chain(*[g[x]["input"] for x in g.keys()]))
    )
    node_mapping = dict((k, i) for i, k in enumerate(set_keys))
    N = len(node_mapping)
    adj = np.zeros([N, N])
    ops = [None for _ in range(N)]
    inputs = dict(
        (idx, k)
        for k, idx in node_mapping.items()
        if np.any([k.startswith(pref) for pref in ["x", "h_prev"]])
    )
    outputs = dict((idx, k) for k, idx in node_mapping.items() if k.startswith("h_new"))
    for k1 in g:
        for k2 in g[k1]["input"]:
            idx1 = node_mapping[k1]
            idx2 = node_mapping[k2]
            # k2 -> k1
            ops[idx1] = g[k1]["op"]
            adj[idx2, idx1] = 1
    paths = get_paths(inputs, outputs, adj)
    path_cnt = Counter()
    for path in paths:
        path_ops = [ops[x] for x in path[1:]]
        path_str = "-".join(path_ops)
        path_cnt[path_str] += 1
    return path_cnt


def get_feature_vector(archs, paths_set_enum):
    """
    Get path encodings for archs. To encode an architecture, we check which paths are
    present in the architecture, and set the corresponding features to 1.

    Parameters
    ----------
    archs:
        Architectures
    paths_set_enum:
        Dict with paths and their indices.

    Returns
    -------
    ndarray:
        Stacked feature vectors
    """
    x = torch.zeros([len(archs), len(paths_set_enum)], dtype=torch.float)
    for i, arch in enumerate(archs):
        for f in get_path_features(arch):
            idx = paths_set_enum.get(f, None)
            if idx is not None:
                x[i, idx] = 1
    return x


def get_paths_limited_by_length(length):
    """
    Get and enumerate all possible paths whose length does not exceed the given one.
    Uses globally defined 'OPS'.

    Parameters
    ----------
    length:
        Max length for paths

    Returns
    -------
    dict:
        Contains paths and assigned indices.
    """
    #     inputs = ['x', 'h_prev', 'h_prev_1', 'h_prev_2']
    #     outputs = ['h_new_0', 'h_new_1', 'h_new_2']
    fixed_length_paths = OPS.copy()
    paths_set = fixed_length_paths
    for i in range(length - 1):
        new_paths = []
        for path in fixed_length_paths:
            for op in OPS:
                new_paths.append(path + "-" + op)
        fixed_length_paths = new_paths
        paths_set.extend(new_paths)
    return dict((p, i) for i, p in enumerate(paths_set))
