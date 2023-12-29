import math
import torch
import torch.nn
import networkx as nx
import torch.nn.functional as F


class MultiLinear(torch.nn.Module):
    """
    Torch module that implements the operation Y = W1*X1 + ... + Wn*Xn + b,
    where X1,..., Xn are inputs.
    """

    def __init__(self, input_sizes, output_size):
        super(MultiLinear, self).__init__()
        self.input_sizes = input_sizes
        self.output_size = output_size

        weights = []
        for input_size in input_sizes:
            weights.append(torch.nn.Parameter(torch.Tensor(output_size, input_size)))
        self.weights = torch.nn.ParameterList(weights)

        self.bias = torch.nn.Parameter(torch.Tensor(output_size))

        self.reset_parameters()

    def reset_parameters(self):
        for i in range(len(self.weights)):
            torch.nn.init.kaiming_uniform_(self.weights[i], a=math.sqrt(5))

        fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weights[0])
        bound = 1 / math.sqrt(fan_in)
        torch.nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, *inputs):
        result = F.linear(inputs[0], self.weights[0], self.bias)
        for i in range(1, len(self.weights)):
            result = result + F.linear(inputs[i], self.weights[i])
        return result

    def extra_repr(self):
        return "input_sizes={}, output_size={}".format(
            self.input_sizes, self.output_size
        )


class CustomRNNCell(torch.nn.Module):
    """
    Module that builds a cell according to a recipe.
    Recipe is a dict that defines a computation graph.
    """

    elementwise_ops_dict = {"prod": torch.mul, "sum": torch.add}

    def __init__(self, input_size, hidden_size, recipe):
        super(CustomRNNCell, self).__init__()

        self.activations_dict = {
            "tanh": torch.nn.Tanh(),
            "sigm": torch.nn.Sigmoid(),
            "leaky_relu": torch.nn.LeakyReLU(),
        }

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.recipe = recipe
        self.hidden_tuple_size = 0

        components_dict = {}

        self.G = nx.DiGraph()
        for k in recipe.keys():
            if k not in components_dict:
                component = self._make_component(recipe[k])
                if component is not None:
                    components_dict[k] = component
                if k.startswith("h_new"):
                    suffix = k.replace("h_new_", "")
                    if suffix.isdigit():
                        self.hidden_tuple_size = max(
                            [self.hidden_tuple_size, int(suffix) + 1]
                        )

                if k not in self.G.nodes():
                    self.G.add_node(k)
                for i, n in enumerate(recipe[k]["input"]):
                    if n not in self.G.nodes():
                        self.G.add_node(k)
                    self.G.add_edge(n, k)

        self.components = torch.nn.ModuleDict(components_dict)
        self.nodes_order = list(nx.algorithms.dag.topological_sort(self.G))

    def forward(self, x, hidden_tuple):
        calculated_nodes = {}
        for n in self.nodes_order:
            if n == "x":
                calculated_nodes["x"] = x.unsqueeze(0)
            elif n.startswith("h_prev") and n.replace("h_prev_", "").isdigit():
                calculated_nodes[n] = hidden_tuple[
                    int(n.replace("h_prev_", ""))
                ].unsqueeze(0)
            elif n in self.components:
                inputs = [calculated_nodes[k] for k in self.recipe[n]["input"]]
                calculated_nodes[n] = self.components[n](*inputs)
            else:
                # simple operations
                op = self.recipe[n]["op"]
                inputs = [calculated_nodes[k] for k in self.recipe[n]["input"]]
                if op in ["elementwise_prod", "elementwise_sum"]:
                    op_func = CustomRNNCell.elementwise_ops_dict[
                        op.replace("elementwise_", "")
                    ]
                    calculated_nodes[n] = op_func(inputs[0], inputs[1])
                    for inp in range(2, len(inputs)):
                        calculated_nodes[n] = op_func(calculated_nodes[n], inputs[i])
                elif op == "blend":
                    calculated_nodes[n] = (
                        inputs[0] * inputs[1] + (1 - inputs[0]) * inputs[2]
                    )
                elif op.startswith("activation"):
                    op_func = self.activations_dict[op.replace("activation_", "")]
                    calculated_nodes[n] = op_func(inputs[0])
        return tuple(
            [calculated_nodes[f"h_new_{i}"][0] for i in range(self.hidden_tuple_size)]
        )

    def _make_component(self, spec):
        if spec["op"] == "linear":
            input_sizes = [
                self.input_size if inp == "x" else self.hidden_size
                for inp in spec["input"]
            ]
            return MultiLinear(input_sizes, self.hidden_size)


class CustomRNN(torch.nn.Module):
    """
    Wrapper over a CustomRNNCell. Manages hidden states correctly.
    """

    def __init__(self, input_size, hidden_size, recipe, bidirectional=False):
        super(CustomRNN, self).__init__()
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.cell_fw = CustomRNNCell(input_size, hidden_size, recipe)
        self.reset_parameters()

    def _forward_pass(self, inputs, hidden_tuple=None):
        batch_size = inputs.size(1)
        hidden_tuple = self._init_hidden_tuple(batch_size, hidden_tuple)
        outputs = []
        for x in torch.unbind(inputs, dim=0):
            hidden_tuple = self.cell_fw(x, hidden_tuple)
            outputs.append(hidden_tuple[0].clone())
        return torch.stack(outputs, dim=0), tuple(
            [x.unsqueeze(0) for x in hidden_tuple]
        )

    def _backward_pass(self, inputs, hidden_tuple=None):
        batch_size = inputs.size(1)
        hidden_tuple = self._init_hidden_tuple(batch_size, hidden_tuple)
        outputs = []
        for x in torch.unbind(torch.flip(inputs, [0]), dim=0):
            hidden_tuple = self.cell_bw(x, hidden_tuple)
            outputs.append(hidden_tuple[0].clone())
        return torch.stack(outputs, dim=0), tuple(
            [x.unsqueeze(0) for x in hidden_tuple]
        )

    def forward(self, inputs, hidden_tuple):
        if self.bidirectional:
            hidden_tuple_fw, hidden_tuple_bw = [], []
            for hidden in hidden_tuple:
                hidden_tuple_fw.append(hidden[0].unsqueeze(0))
                hidden_tuple_bw.append(hidden[1].unsqueeze(0))
        else:
            hidden_tuple_fw = hidden_tuple

        outputs_fw, hidden_tuple_fw = self._forward_pass(inputs, hidden_tuple_fw)
        if self.bidirectional:
            outputs_bw, hidden_tuple_bw = self._backward_pass(inputs, hidden_tuple_bw)
            outputs = torch.cat([outputs_fw, outputs_bw], -1)
            hidden_tuple = []
            for i in range(len(hidden_tuple_fw)):
                hidden_tuple.append(
                    torch.cat([hidden_tuple_fw[i], hidden_tuple_bw[i]], 0)
                )
        else:
            outputs = outputs_fw
            hidden_tuple = hidden_tuple_fw

        return outputs, hidden_tuple

    def _init_hidden_tuple(self, batch_size, hidden_tuple):
        if hidden_tuple is None:
            hidden_tuple = tuple(
                [
                    self.init_hidden(batch_size)
                    for _ in range(self.cell.hidden_tuple_size)
                ]
            )
        self.check_hidden_size(hidden_tuple, batch_size)
        hidden_tuple = tuple([x[0] for x in hidden_tuple])
        return hidden_tuple

    def init_hidden(self, batch_size):
        # num_layers == const (1)
        return torch.zeros(1, batch_size, self.hidden_size).to(
            next(self.parameters()).device
        )

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for param in self.parameters():
            torch.nn.init.uniform_(param, -stdv, stdv)

    def check_hidden_size(self, hidden_tuple, batch_size):
        expected_hidden_size = (1, batch_size, self.hidden_size)
        msg = "Expected hidden size {}, got {}"
        for hx in hidden_tuple:
            if hx.size() != expected_hidden_size:
                raise RuntimeError(msg.format(expected_hidden_size, tuple(hx.size())))
