import logging
from functools import partial
from typing import Callable, List, Mapping, Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn

from snorkel.analysis import Scorer

Outputs = Mapping[str, List[torch.FloatTensor]]


class Operation:
    """A single operation (forward pass of a module) to execute in a Task.

    See ``Task`` for more detail on the usage and semantics of an Operation.

    Parameters
    ----------
    name
        The name of this operation (defaults to module_name since for most workflows,
        each module is only used once per forward pass)
    module_name
        The name of the module in the module pool that this operation uses
    inputs
        The inputs that the specified module expects, given as tuples which reference a
        previous operation (or the original input) and an index (if the specified
        operation outputs a single value or sequence of values) or a key (if the
        specified operation outputs a dictionary of values)

    Attributes
    ----------
    name
        See above
    module_name
        See above
    inputs
        See above
    """

    def __init__(
        self,
        module_name: str,
        inputs: Sequence[Union[str, Tuple[str, str]]],
        name: Optional[str] = None,
    ) -> None:
        self.name = name or module_name
        self.module_name = module_name
        self.inputs = inputs

    def __repr__(self) -> str:
        return (
            f"Operation(name={self.name}, "
            f"module_name={self.module_name}, "
            f"inputs={self.inputs})"
        )


class Task:
    r"""A single task (a collection of modules and specified path through them).

    Parameters
    ----------
    name
        The name of the task
    module_pool
        A ModuleDict mapping module names to the modules themselves
    op_sequence
        A list of ``Operation``\s to execute in order, defining the flow of information
        through the network for this task
    scorer
        A ``Scorer`` with the desired metrics to calculate for this task
    loss_func
        A function that converts final logits into loss values.
        Defaults to cross_entropy_from_outputs() (which expects integer labels) if none
        is provided. To use probalistic labels for training, use the method
        cross_entropy_with_probs_from_outputs() instead.
        Note that whatever function is used will receive as inputs the outputs dict,
        labels, and a mask denoting which data points are 'active' (have labels)
    output_func
        A function that converts final logits into 'outputs' (e.g. probabilities)
        Defaults to softmax_from_outputs()

    Attributes
    ----------
    name
        See above
    module_pool
        See above
    op_sequence
        See above
    scorer
        See above
    loss_func
        See above
    output_func
        See above
    """

    def __init__(
        self,
        name: str,
        module_pool: nn.ModuleDict,
        op_sequence: Sequence[Operation],
        scorer: Scorer = Scorer(metrics=["accuracy"]),
        loss_func: Optional[Callable[..., torch.Tensor]] = None,
        output_func: Optional[Callable[..., torch.Tensor]] = None,
    ) -> None:
        self.name = name
        self.module_pool = module_pool
        self.op_sequence = op_sequence
        self.loss_func = loss_func or F.cross_entropy
        self.output_func = output_func or partial(F.softmax, dim=1)
        self.scorer = scorer

        logging.info(f"Created task: {self.name}")

    def __repr__(self) -> str:
        cls_name = type(self).__name__
        return f"{cls_name}(name={self.name})"
