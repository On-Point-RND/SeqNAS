from __future__ import print_function

from collections import defaultdict
import collections
from datetime import datetime
import os
import json
import logging

import numpy as np

import torch
from torch.autograd import Variable

from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw

import numpy as np
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
import os
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import json
from pathlib import Path
import networkx as nx


def running_mean(x, N):
    """
    Computes running mean.

    Parameters
    ----------
    x:
        The sequence to smooth out.
    op:
        The window size.

    Returns
    -------
    ndarray:
        Smoothed sequence.
    """
    return pd.Series(x).rolling(window=N, center=True).mean().values


def events_dict(dpath):
    """
    Reads events in file or directory and returns dictionary.

    Parameters
    ----------
    dpath:
        Path to tensoroard events log.

    Returns
    -------
    dict:
        Dictionary that for each tag contains a tuple (steps, scalar values)
    """
    dpath = str(dpath)
    summary_iterator = EventAccumulator(dpath).Reload()
    tags = summary_iterator.Tags()["scalars"]
    out = {}
    for tag in tags:
        steps = [e.step for e in summary_iterator.Scalars(tag)]
        events = summary_iterator.Scalars(tag)
        out[tag] = (steps, [e.value for e in events])
    return out


def read_logs(dir_path, output_df=True):
    """
    Reads logs in directory. Returns dictionary or dataframe.

    Parameters
    ----------
    x:
        Path to directory with logs.
    output_df:
        The function returns a dataframe if this parameter is True, it returns a dict otherwise.

    Returns
    -------
    dict or dataframe:
        Read  data of dict type or dataframe type depending on the arguments.
    bool:
        False if reading data has failed by some reason. True otherwise.
    """
    directory = Path(dir_path)
    data_list = []
    failed = []
    infos = directory.rglob("*.json")
    logdirs = (
        set([info.parent for info in infos])
        - set(directory.rglob("*/.*"))
        - set(directory.rglob(".*"))
    )
    for logdir in logdirs:
        try:
            info_path = logdir / "info.json"
            info = json.loads(info_path.read_text())
            events = events_dict(logdir)
            info["logdir"] = str(logdir)
            if len(events) == 0:
                info["epochs"] = 0
            else:
                info.update(events)
                info["epochs"] = len(events["Loss/val"][0])
            data_list.append(info)
        except:
            failed.append(logdir)
    print("Num of successfully read trials: ", len(data_list))
    if output_df:
        df = pd.DataFrame(data_list)
        # df = df.set_index(df.arch)
        return df, failed
    else:
        return data_list, failed


def get_feature_tensor(features_dict):
    return torch.tensor(np.vstack(list(features_dict.values())).T)


def collate_sequences(batch):
    seq_id, seq_data, seq_label = zip(*batch)
    labels = torch.tensor(seq_label)
    features = []
    lengths = []
    for f in seq_data:
        # Considering there are no null values in features (all feature values arrays are the same length)
        lengths.append(len(list(f.values())[0]))
        features.append(get_feature_tensor(f))
    # print(features)
    packed = torch.nn.utils.rnn.pack_sequence(features, False)
    padded = torch.nn.utils.rnn.pad_packed_sequence(packed)
    # print(packed)

    return [padded[0], labels, padded[1]]


def _pad_sequences(batch):
    # batch is a list of tuples: tensor, int, int
    it = iter(batch)
    elem_size = len(next(it))
    if not all(len(elem) == elem_size for elem in it):
        raise RuntimeError("each element in list of batch should be of equal size")
    transposed = zip(*batch)
    padded = torch.nn.utils.rnn.pad_sequence(next(transposed), padding_value=0.0)
    labels = torch.tensor(next(transposed))
    lengths = torch.tensor(next(transposed), dtype=torch.long)
    return [padded, labels, lengths]


def _train_test_val_split(inds, train_size=0.75, test_size=0.125, random_state=42):
    train_inds, test_val_ids = train_test_split(
        inds, random_state=random_state, train_size=train_size
    )

    val_inds, test_inds = train_test_split(
        test_val_ids, random_state=random_state, test_size=test_size / (1 - train_size)
    )
    return train_inds, test_inds, val_inds


def repackage_hidden(h):
    """Wraps hidden states in new Tensors,
    to detach them from their history."""
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)


def make_graph(recipe):
    """
    Return networkx directed graph based on the recipe.

    Parameters
    ----------
    recipe:
        Arch in recipe format.

    Returns
    -------
    nx.DiGraph:
        Networkx graph based on the recipe.
    """
    G = nx.DiGraph()

    for key in recipe.keys():
        op = recipe[key]["op"]
        if key.startswith("h_new_"):
            op = key + ":" + op
        G.add_node(key, name=key, op=op)
        for inp in recipe[key]["input"]:
            if "h_prev" in inp or inp == "x":
                G.add_node(inp, name=inp, op=inp)
            else:
                G.add_node(inp, name=inp)
            G.add_edge(inp, key)
    return G


def recipe2matrixops(recipe):
    """
    Return the adjacency matrix and the node operations given recipe.

    Parameters
    ----------
    recipe:
        Arch in recipe format.

    Returns
    -------
    ndarray:
        Adjacency matrix of the graph.
    ndarray:
        Operations of nodes in order consistent with adjacency matrix.
    """
    G = make_graph(recipe)
    labels = nx.get_node_attributes(G, "op")
    nodelist_with_ops = np.array(list(labels.items()))

    matrix = nx.to_numpy_array(G, nodelist=nodelist_with_ops[:, 0])
    ops = nodelist_with_ops[:, 1]

    return matrix, ops


def graph_edit_distance(matrixops1, matrixops2):
    """
    Compute graph edit distance between 2 graphs.

    Parameters
    ----------
    matrixops1:
        Tuple with an adjacency matrix and op attributes of the first graph.
    matrixops2:
        Tuple with an adjacency matrix and op attributes of the second graph.


    Returns
    -------
    int:
        Edit distance between the graphs.
    """
    m1, l1 = matrixops1
    m2, l2 = matrixops2

    # Pad
    n1, n2 = m1.shape[0], m2.shape[0]
    max_n = max(n1, n2)
    m1 = np.pad(m1, ((0, max_n - m1.shape[0]), (0, max_n - m1.shape[0])))
    m2 = np.pad(m2, ((0, max_n - m2.shape[0]), (0, max_n - m2.shape[0])))
    l1 = np.pad(l1, (0, max_n - l1.shape[0]), constant_values=None)
    l2 = np.pad(l2, (0, max_n - l2.shape[0]), constant_values=None)

    d = 100000000
    for p in permutations(range(len(m1))):
        p = list(p)
        d_p = (m1 != m2[p][:, p]).sum() + (l1 != l2[p]).sum()
        d = min(d, d_p)
    return d


def get_gpu():
    """
    Scan available gpu devices and choose the one with a lot of free memory and low utilization.

    Returns
    -------
    string:
        cuda device
    """
    t = os.popen("nvidia-smi -q -d Memory |grep -A4 GPU|grep Free").readlines()
    memory_available = np.array([int(x.split()[2]) for x in t])
    memory_available = memory_available / memory_available.sum()
    t = os.popen("nvidia-smi -q -d Utilization |grep -A4 GPU|grep Gpu").readlines()
    utilization = np.array([int(x.split()[2]) for x in t]) / 100
    idx = np.argmax(memory_available - 0.5 * utilization)
    return f"cuda:{idx}"


def _node_equal(x, y):
    return x["op"] == y["op"]


def recipe2rnntype(recipe):
    """
    Compare the recipe with standard architectures. If graphs of some basic architecture and the recipe are isomorphic
    then return an adopted name of the arch. Otherwise return 'CustomRNN'.

    Parameters
    ----------
    recipe:
        Arch in recipe format.

    Returns
    -------
    string:
        Can be 'RNN', 'LSTM', 'GRU' or 'CustomRNN' depending on the matching result.
    """
    rnn = make_graph(
        {
            "f": {"op": "linear", "input": ["x", "h_prev_0"]},
            "h_new_0": {"op": "activation_tanh", "input": ["f"]},
        }
    )
    lstm = make_graph(
        {
            "i": {"op": "linear", "input": ["x", "h_prev_0"]},
            "i_act": {"op": "activation_tanh", "input": ["i"]},
            "j": {"op": "linear", "input": ["x", "h_prev_0"]},
            "j_act": {"op": "activation_sigm", "input": ["j"]},
            "f": {"op": "linear", "input": ["x", "h_prev_0"]},
            "f_act": {"op": "activation_sigm", "input": ["f"]},
            "o": {"op": "linear", "input": ["x", "h_prev_0"]},
            "o_act": {"op": "activation_tanh", "input": ["o"]},
            "h_new_1_part1": {"op": "elementwise_prod", "input": ["f_act", "h_prev_1"]},
            "h_new_1_part2": {"op": "elementwise_prod", "input": ["i_act", "j_act"]},
            "h_new_1": {
                "op": "elementwise_sum",
                "input": ["h_new_1_part1", "h_new_1_part2"],
            },
            "h_new_1_act": {"op": "activation_tanh", "input": ["h_new_1"]},
            "h_new_0": {"op": "elementwise_prod", "input": ["h_new_1_act", "o_act"]},
        }
    )
    gru = make_graph(
        {
            "r": {"op": "linear", "input": ["x", "h_prev_0"]},
            "r_act": {"op": "activation_sigm", "input": ["r"]},
            "z": {"op": "linear", "input": ["x", "h_prev_0"]},
            "z_act": {"op": "activation_sigm", "input": ["z"]},
            "rh": {"op": "elementwise_prod", "input": ["r_act", "h_prev_0"]},
            "h_tilde": {"op": "linear", "input": ["x", "rh"]},
            "h_tilde_act": {"op": "activation_tanh", "input": ["h_tilde"]},
            "h_new_0": {"op": "blend", "input": ["z_act", "h_prev_0", "h_tilde_act"]},
        }
    )

    recipe_graph = make_graph(recipe)
    if nx.is_isomorphic(recipe_graph, rnn, node_match=_node_equal):
        return "RNN"
    elif nx.is_isomorphic(recipe_graph, lstm, node_match=_node_equal):
        return "LSTM"
    elif nx.is_isomorphic(recipe_graph, gru, node_match=_node_equal):
        return "GRU"
    else:
        return "CustomRNN"


#### ENAS utils


# try:
#     import scipy.misc
#     imread = scipy.misc.imread
#     imresize = scipy.misc.imresize
#     imsave = imwrite = scipy.misc.imsave
# except:
#     import cv2
#     imread = cv2.imread
#     imresize = cv2.imresize
#     imsave = imwrite = cv2.imwrite


##########################
# Network visualization
##########################


def add_node(graph, node_id, label, shape="box", style="filled"):
    if label.startswith("x"):
        color = "white"
    elif label.startswith("h"):
        color = "skyblue"
    elif label == "tanh":
        color = "yellow"
    elif label == "ReLU":
        color = "pink"
    elif label == "identity":
        color = "orange"
    elif label == "sigmoid":
        color = "greenyellow"
    elif label == "avg":
        color = "seagreen3"
    else:
        color = "white"

    if not any(label.startswith(word) for word in ["x", "avg", "h"]):
        label = f"{label}\n({node_id})"

    graph.add_node(
        node_id,
        label=label,
        color="black",
        fillcolor=color,
        shape=shape,
        style=style,
    )


def draw_network(dag, path):
    makedirs(os.path.dirname(path))
    graph = pgv.AGraph(
        directed=True, strict=True, fontname="Helvetica", arrowtype="open"
    )  # not work?

    checked_ids = [-2, -1, 0]

    if -1 in dag:
        add_node(graph, -1, "x[t]")
    if -2 in dag:
        add_node(graph, -2, "h[t-1]")

    add_node(graph, 0, dag[-1][0].name)

    for idx in dag:
        for node in dag[idx]:
            if node.id not in checked_ids:
                add_node(graph, node.id, node.name)
                checked_ids.append(node.id)
            graph.add_edge(idx, node.id)

    graph.layout(prog="dot")
    graph.draw(path)


def make_gif(paths, gif_path, max_frame=50, prefix=""):
    import imageio

    paths.sort()

    skip_frame = len(paths) // max_frame
    paths = paths[::skip_frame]

    images = [imageio.imread(path) for path in paths]
    max_h, max_w, max_c = np.max(np.array([image.shape for image in images]), 0)

    for idx, image in enumerate(images):
        h, w, c = image.shape
        blank = np.ones([max_h, max_w, max_c], dtype=np.uint8) * 255

        pivot_h, pivot_w = (max_h - h) // 2, (max_w - w) // 2
        blank[pivot_h : pivot_h + h, pivot_w : pivot_w + w, :c] = image

        images[idx] = blank

    try:
        images = [Image.fromarray(image) for image in images]
        draws = [ImageDraw.Draw(image) for image in images]
        font = ImageFont.truetype("assets/arial.ttf", 30)

        steps = [
            int(os.path.basename(path).rsplit(".", 1)[0].split("-")[1])
            for path in paths
        ]
        for step, draw in zip(steps, draws):
            draw.text(
                (max_h // 20, max_h // 20),
                f"{prefix}step: {format(step, ',d')}",
                (0, 0, 0),
                font=font,
            )
    except IndexError:
        pass

    imageio.mimsave(gif_path, [np.array(img) for img in images], duration=0.5)


##########################
# Torch
##########################


def detach(h):
    if type(h) == Variable:
        return Variable(h.data)
    else:
        return tuple(detach(v) for v in h)


def get_variable(inputs, cuda=False, **kwargs):
    if type(inputs) in [list, np.ndarray]:
        inputs = torch.Tensor(inputs)
    if cuda:
        out = Variable(inputs.cuda(), **kwargs)
    else:
        out = Variable(inputs, **kwargs)
    return out


def update_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def batchify(data, bsz, use_cuda):
    # code from https://github.com/pytorch/examples/blob/master/word_language_model/main.py
    nbatch = data.size(0) // bsz
    data = data.narrow(0, 0, nbatch * bsz)
    data = data.view(bsz, -1).t().contiguous()
    if use_cuda:
        data = data.cuda()
    return data


##########################
# ETC
##########################

Node = collections.namedtuple("Node", ["id", "name"])


class keydefaultdict(defaultdict):
    def __missing__(self, key):
        if self.default_factory is None:
            raise KeyError(key)
        else:
            ret = self[key] = self.default_factory(key)
            return ret


def to_item(x):
    """Converts x, possibly scalar and possibly tensor, to a Python scalar."""
    if isinstance(x, (float, int)):
        return x

    if float(torch.__version__[0:3]) < 0.4:
        assert (x.dim() == 1) and (len(x) == 1)
        return x[0]

    return x.item()


def get_logger(name=__file__, level=logging.INFO):
    logger = logging.getLogger(name)

    if getattr(logger, "_init_done__", None):
        logger.setLevel(level)
        return logger

    logger._init_done__ = True
    logger.propagate = False
    logger.setLevel(level)

    formatter = logging.Formatter("%(asctime)s:%(levelname)s::%(message)s")
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    handler.setLevel(0)

    del logger.handlers[:]
    logger.addHandler(handler)

    return logger


logger = get_logger()


def prepare_dirs(args):
    """Sets the directories for the model, and creates those directories.

    Args:
        args: Parsed from `argparse` in the `config` module.
    """
    if args.load_path:
        if args.load_path.startswith(args.log_dir):
            args.model_dir = args.load_path
        else:
            if args.load_path.startswith(args.dataset):
                args.model_name = args.load_path
            else:
                args.model_name = "{}_{}".format(args.dataset, args.load_path)
    else:
        args.model_name = "{}_{}".format(args.dataset, get_time())

    if not hasattr(args, "model_dir"):
        args.model_dir = os.path.join(args.log_dir, args.model_name)
    args.data_path = os.path.join(args.data_dir, args.dataset)

    for path in [args.log_dir, args.data_dir, args.model_dir]:
        if not os.path.exists(path):
            makedirs(path)


def get_time():
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def save_args(args):
    param_path = os.path.join(args.model_dir, "params.json")

    logger.info("[*] MODEL dir: %s" % args.model_dir)
    logger.info("[*] PARAM path: %s" % param_path)

    with open(param_path, "w") as fp:
        json.dump(args.__dict__, fp, indent=4, sort_keys=True)


def save_dag(args, dag, name):
    save_path = os.path.join(args.model_dir, name)
    logger.info("[*] Save dag : {}".format(save_path))
    json.dump(dag, open(save_path, "w"))


def load_dag(args):
    load_path = os.path.join(args.dag_path)
    logger.info("[*] Load dag : {}".format(load_path))
    with open(load_path) as f:
        dag = json.load(f)
    dag = {int(k): [Node(el[0], el[1]) for el in v] for k, v in dag.items()}
    save_dag(args, dag, "dag.json")
    draw_network(dag, os.path.join(args.model_dir, "dag.png"))
    return dag


def makedirs(path):
    if not os.path.exists(path):
        logger.info("[*] Make directories : {}".format(path))
        os.makedirs(path)


def remove_file(path):
    if os.path.exists(path):
        logger.info("[*] Removed: {}".format(path))
        os.remove(path)


def backup_file(path):
    root, ext = os.path.splitext(path)
    new_path = "{}.backup_{}{}".format(root, get_time(), ext)

    os.rename(path, new_path)
    logger.info("[*] {} has backup: {}".format(path, new_path))
