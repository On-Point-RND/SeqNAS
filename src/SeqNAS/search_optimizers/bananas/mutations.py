from collections import defaultdict
from copy import deepcopy
import numpy as np

# operations and number of arguments needed
MAIN_OPS = {
    "linear": 2,
    "blend": 3,
    "elementwise_prod": 2,
    "elementwise_sum": 2,
}
ACT_OPS = {
    "activation_tanh": 1,
    "activation_sigm": 1,
    "activation_leaky_relu": 1,
}
OPS = {**MAIN_OPS, **ACT_OPS}


def check_sanity(arch):
    """
    Check every node has path to some output and
    prev hidden state doesn't connected directly to new hidden state

    Parameters
    ----------
    arch:
        'Arch' class object

    Returns
    -------
    bool:
        True if the architecture passed the test, False otherwise.
    """
    q = arch.node_types["output"].copy()
    visited = set(q)
    while len(q) > 0:
        n = q.pop()
        for node in arch.arch[n]["input"]:
            if node not in visited:
                q.add(node)
                visited.add(node)

    is_sanity_check_ok = True
    for node in arch.node_types["base"] | arch.node_types["intermediate"]:
        if node not in visited:
            is_sanity_check_ok = False
            break

    for node in arch.node_types["output"]:
        if len(set(arch.arch[node]["input"]) & arch.node_types["base"] - set("x")) > 0:
            is_sanity_check_ok = False
            break
    return is_sanity_check_ok


def check_connectivity(arch):
    """
    Check a connectivity if we treat the architecture as an undirected graph

    Parameters
    ----------
    arch:
        'Arch' class object.

    Returns
    -------
    bool:
        True if the architecture passed the test, False otherwise.
    """
    q = set([next(iter(arch.node_types["base"]))])
    visited = q.copy()
    while len(q) > 0:
        n = q.pop()
        for node in arch.arch[n]["input"]:
            if node not in visited:
                q.add(node)
                visited.add(node)
        for node in arch.arch[n]["output"]:
            if node not in visited:
                q.add(node)
                visited.add(node)
    return arch.nodes == visited


# -------------------------------------------------------------------------------------


class Arch:
    def __init__(self, recipe):
        """
        Helper class that implements methods to avoid loops and to mutate more efficiently.
        We represent Arch as a DAG (directed acyclic graph) with node-operations. For the objects of this class we
        * Classify nodes to base (input), intermediate and output types,
        * Store not only inputs but also outputs for every node.
        * Have methods to manipulate edges and nodes, find reachable nodes.

        Parameters
        ----------
        recipe:
            architecture in 'recipe' format.
        """
        self.arch = deepcopy(recipe)
        self.classify_nodes()
        self.add_out_edges()

    def reachable_nodes(self, node):
        """
        Compute the set of reachable nodes from the given node recursively

        Parameters
        ----------
        node:
            The node from which the returned nodes are reachable.

        Returns
        -------
        set:
            The set of reachable nodes from the given node.
        """
        nodes = set([node])
        for output_node in self.arch[node]["output"]:
            nodes |= self.reachable_nodes(output_node)
        return nodes

    def reachable_for(self, node):
        """
        Compute the set of nodes from which the given node can be reached.

        Parameters
        ----------
        node:
            The node which is reachable for the returned nodes.

        Returns
        -------
        set:
            The set of nodes from which the given node can be reached.
        """
        nodes = set([node])
        for input_node in self.arch[node]["input"]:
            nodes |= self.reachable_for(input_node)
        return nodes

    def add_out_edges(self):
        """
        Add output edges to the info of each node.
        """
        for node, d in self.arch.items():
            self.arch[node]["output"] = []
        for node, d in self.arch.items():
            for n in d["input"]:
                self.arch[n]["output"].append(node)

    def compute_reachability(self):
        """
        Efficiently compute the set of reachable nodes for each node at once.
        """
        # find reachable nodes
        q = set()
        for node, d in self.arch.items():
            self.arch[node]["reachable"] = set([node])
            if len(d["output"]) == 0:  # sink nodes
                q.add(node)
        counts = defaultdict(int)
        while len(q) > 0:
            # print(q, processed)
            node = q.pop()
            for input_node in self.arch[node]["input"]:
                counts[input_node] += 1
                self.arch[input_node]["reachable"] |= self.arch[node]["reachable"]
                if counts[input_node] == len(self.arch[input_node]["output"]):
                    q.add(input_node)

    def classify_nodes(self):
        """
        Classify nodes to base (input), intermediate and output types.
        """
        self.node_types = dict()
        self.node_types["intermediate"] = set()
        self.node_types["base"] = set()
        self.node_types["output"] = set()
        self.nodes = set()
        self.op_nodes = defaultdict(set)
        self.node_max_id = 0
        for node, d in self.arch.items():
            self.nodes.add(node)
            self.op_nodes[d["op"]].add(node)
            self.nodes |= set(d["input"])
        for node in self.nodes:
            if node.startswith("h_new"):
                self.node_types["output"].add(node)
            elif node.startswith("h_prev") or node.startswith("x"):
                self.node_types["base"].add(node)
                self.arch[node] = {"op": None, "input": []}
            else:
                self.node_types["intermediate"].add(node)
                self.node_max_id = max(self.node_max_id, int(node[5:]))

    def add_edge(self, src_node, dst_node):
        """
        Add edge. The method doesn't preserve a precomputed reachability.

        Parameters
        ----------
        src_node:
            Source node.
        dst_node:
            Destination node.
        """
        self.arch[dst_node]["input"].append(src_node)
        self.arch[src_node]["output"].append(dst_node)

    def del_edge(self, src_node, dst_node):
        """
        Delete edge. The method doesn't preserve a precomputed reachability.

        Parameters
        ----------
        src_node:
            Source node.
        dst_node:
            Destination node.
        """
        self.arch[dst_node]["input"].remove(src_node)
        self.arch[src_node]["output"].remove(dst_node)

    def add_node(self, node, op):
        """
        Add new node.

        Parameters
        ----------
        node:
            The node name.
        op:
            The node operation.
        """
        self.arch[node] = {"op": op, "input": [], "output": []}
        self.op_nodes[op].add(node)
        if node.startswith("node_"):
            self.node_max_id = max(self.node_max_id, int(node[5:]))
            self.node_types["intermediate"].add(node)

    def del_node(self, node):
        """
        Delete node and its connections.

        Parameters
        ----------
        node:
            The node to delete.
        """
        for n in self.arch[node]["input"].copy():
            self.del_edge(n, node)
        for n in self.arch[node]["output"].copy():
            self.del_edge(node, n)
        self.op_nodes[self.arch[node]["op"]].discard(node)
        self.arch.pop(node)
        self.nodes.discard(node)
        for s in self.node_types:
            self.node_types[s].discard(node)

    def get_recipe(self):
        """
        Return arch in recipe format

        Returns
        -------
        dict:
            Arch in recipe format.
        """
        return {
            node: {"op": d["op"], "input": d["input"]}
            for node, d in self.arch.items()
            if node not in self.node_types["base"]
        }


# -------------------------------------------------------------------------------------


def mutate_reconnect(recipe, mutation_iters=5, trials=500, p=None):
    """
    Randomly reconnect input connections for nodes with probability p

    Parameters
    ----------
    recipe:
        Original recipe.
    mutation_iters:
        Maximum number of consecutive mutations to try.
    trials:
        Maximum number of attempts.
    p:
        Probability of applying the mutation to each node. Default is 1/num_of_nodes.

    Returns
    -------
    dict:
        Mutated arch in recipe format. Original recipe if the number of attempts exceeded the maximum.
    int:
        Number of actual trials.
    """
    p = p if p else 1 / len(recipe)
    t = 0
    for t in range(trials):
        a = Arch(recipe)
        for m in range(mutation_iters):
            for node in a.node_types["intermediate"] | a.node_types["output"]:
                try:
                    if np.random.random() < p:
                        num_connections = len(a.arch[node]["input"])
                        for input_node in a.arch[node]["input"].copy():
                            a.del_edge(input_node, node)
                        new_connections = set()
                        for _ in range(num_connections):
                            new_connection = str(
                                np.random.choice(
                                    list(
                                        a.nodes
                                        - a.reachable_nodes(node)
                                        - new_connections
                                    )
                                )
                            )
                            a.add_edge(new_connection, node)
                            new_connections.add(new_connection)
                except:
                    pass
            if check_sanity(a) and check_connectivity(a):
                return a.get_recipe(), t + 1
    return recipe, t + 1


def mutate_delete_node(recipe, p=1, trials=500):
    """
    Delete random intermediate node with probability p and its connections, reassign minimal num of new
    connections so that remaining nodes have output connection and the same number of inputs (except maybe linear)

    Parameters
    ----------
    recipe:
        Original recipe.
    p:
        Probability of applying the mutation.
    trials:
        Maximum number of attempts.

    Returns
    -------
    dict:
        Mutated arch in recipe format. Original recipe if the number of attempts exceeded the maximum.
    int:
        Number of actual trials.
    """
    t = 0
    if np.random.random() < p:
        for t in range(trials):
            a = Arch(recipe)
            d_node = np.random.choice(list(a.node_types["intermediate"]))
            input_nodes = np.random.permutation(a.arch[d_node]["input"])
            output_nodes = np.random.permutation(a.arch[d_node]["output"])
            a.del_node(d_node)
            # nodes that need connections
            need_output = [n for n in input_nodes if len(a.arch[n]["output"]) == 0]
            need_input = [
                n
                for n in output_nodes
                if not n in a.op_nodes["linear"] or len(a.arch[n]["input"]) == 0
            ]
            # print(need_output, need_input)
            try:
                if len(need_output) <= len(need_input):
                    min_num = len(need_output)
                    connections = np.random.choice(
                        input_nodes, len(need_input) - min_num
                    )
                    for i, c in enumerate(connections):
                        a.add_edge(c, need_input[i + min_num])
                else:
                    min_num = len(need_input)
                    for i in range(min_num, len(need_output)):
                        node = need_output[i]
                        # give priority to output_nodes
                        available_nodes = [
                            n for n in output_nodes if n in a.op_nodes["linear"]
                        ]
                        if len(available_nodes) == 0:
                            available_nodes = list(
                                a.op_nodes["linear"] - a.reachable_for(node)
                            )
                        c = np.random.choice(available_nodes)
                        a.add_edge(node, c)
                for i in range(min_num):
                    a.add_edge(need_output[i], need_input[i])
                if check_sanity(a) and check_connectivity(a):
                    return a.get_recipe(), t + 1
            except:
                pass
    return recipe, t + 1


def mutate_add_node(recipe, p=1, trials=500, connect_trials=10):
    """
    Add node with randomly chosen op and connections

    Parameters
    ----------
    recipe:
        Original recipe.
    p:
        Probability of applying the mutation.
    trials:
        Maximum number of attempts.
    connect_trials:
        Maximum number of attempts to make connections.

    Returns
    -------
    dict:
        Mutated arch in recipe format. Original recipe if the number of attempts exceeded the maximum.
    int:
        Number of actual trials.
    """
    t = 0
    if np.random.random() < p:
        for t in range(trials):
            a = Arch(recipe)
            op = np.random.choice(list(OPS.keys()))
            a_node = "node_" + str(a.node_max_id + 1)
            a.add_node(a_node, op)
            num_inputs = OPS[op]
            num_outputs = np.random.randint(
                low=min(OPS.values()), high=max(OPS.values()) + 1
            )
            # print(op, num_inputs, num_outputs)
            try:
                for i in range(num_inputs):
                    c = np.random.choice(
                        list(
                            a.nodes
                            - a.reachable_nodes(a_node)
                            - set(a.arch[a_node]["input"])
                        )
                    )
                    a.add_edge(c, a_node)
                connect_trial = 0
                while (
                    len(a.arch[a_node]["output"]) != num_outputs
                    and connect_trial < connect_trials
                ):
                    c = np.random.choice(
                        list(
                            a.nodes
                            - a.reachable_for(a_node)
                            - set(a.arch[a_node]["output"])
                        )
                    )
                    if c in a.op_nodes["linear"]:
                        a.add_edge(a_node, c)
                        continue
                    for c_input in a.arch[c]["input"].copy():
                        if len(a.arch[c_input]["output"]) > 1:
                            a.del_edge(c_input, c)
                            a.add_edge(a_node, c)
                            break
                    connect_trial += 1
                if check_sanity(a):
                    return a.get_recipe(), t + 1
            except:
                pass
    return recipe, t + 1


def mutate_change_operation(recipe, p=1, trials=500):
    """
    Change operation for randomly chosen node. Add or delete connections if needed.

    Parameters
    ----------
    recipe:
        Original recipe.
    p:
        Probability of applying the mutation.
    trials:
        Maximum number of attempts.

    Returns
    -------
    dict:
        Mutated arch in recipe format. Original recipe if the number of attempts exceeded the maximum.
    int:
        Number of actual trials.
    """
    t = 0
    if np.random.random() < p:
        for t in range(trials):
            a = Arch(recipe)
            node = np.random.choice(list(a.nodes - a.node_types["base"]))
            try:
                op = a.arch[node]["op"]
                if op in MAIN_OPS:
                    new_op = np.random.choice([o for o in MAIN_OPS if o != op])
                    a.arch[node]["op"] = new_op
                    input_nodes = np.random.permutation(list(a.arch[node]["input"]))
                    excess = input_nodes[: max(0, len(input_nodes) - OPS[new_op])]
                    for n in excess:
                        a.del_edge(n, node)
                    for i in range(0, OPS[new_op] - len(input_nodes)):
                        c = np.random.choice(
                            list(
                                a.nodes
                                - a.reachable_nodes(node)
                                - set(a.arch[node]["input"])
                            )
                        )
                        a.add_edge(c, node)
                    a.arch[node]["input"] = np.random.permutation(
                        a.arch[node]["input"]
                    ).tolist()
                else:
                    new_op = np.random.choice([o for o in ACT_OPS if o != op])
                    a.arch[node]["op"] = new_op
                if check_sanity(a):
                    return a.get_recipe(), t + 1
            except:
                pass
    return recipe, t + 1


MUTATIONS = {
    mutate_reconnect,
    mutate_add_node,
    mutate_delete_node,
    mutate_change_operation,
}
