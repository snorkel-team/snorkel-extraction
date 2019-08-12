from functools import partial
from typing import Any, Dict, List, Optional, Set, Union

import numpy as np
import torch
import torch.nn as nn

from snorkel.analysis import Scorer
from snorkel.classification import (
    DictDataLoader,
    DictDataset,
    MultitaskClassifier,
    Operation,
    Task,
    cross_entropy_from_outputs,
    softmax_from_outputs,
)

from .utils import add_slice_labels, convert_to_slice_tasks


class BinarySlicingClassifier(MultitaskClassifier):
    """
    Parameters
    ----------

    Attributes
    ----------
    base_task
    """

    # Default string for dataset naming conventions
    base_input_name = "_base_input_"
    base_task_name = "base_task"

    def __init__(
        self,
        representation_net: nn.Module,
        slice_names: List[str],
        head_dim: int,
        name: str = "BinarySlicingClassifier",
        scorer: Scorer = Scorer(metrics=["accuracy", "f1"]),
        **kwargs: Any,
    ) -> None:

        module_pool = nn.ModuleDict(
            {
                "representation_net": representation_net,
                # By convention, initialize binary classification as 2-dim output
                "prediction_head": nn.Linear(head_dim, 2),
            }
        )

        task_flow = [
            Operation(
                name="input_op",
                module_name="representation_net",
                inputs=[("_input_", self.base_input_name)],
            ),
            Operation(
                name="head_op", module_name="prediction_head", inputs=[("input_op", 0)]
            ),
        ]

        self.base_task = Task(
            name=self.base_task_name,
            module_pool=module_pool,
            task_flow=task_flow,
            loss_func=partial(cross_entropy_from_outputs, "head_op"),
            output_func=partial(softmax_from_outputs, "head_op"),
            scorer=scorer,
        )

        slice_tasks = convert_to_slice_tasks(self.base_task, slice_names)

        # Initialize a MultitaskClassifier under the hood
        super().__init__(tasks=slice_tasks, name=name, **kwargs)
        self.slice_names = slice_names

    def make_slice_dataloaders(
        self,
        datasets: List[DictDataset],
        S: np.ndarray,
        slice_names: List[str],
        **dataloader_kwargs: Any,
    ) -> List[DictDataLoader]:
        """Modifies dataloaders in-place to accomodate slice labels."""

        # TODO: some assert statements about data structure?
        # TODO: check for "data" field in X_dict?

        dataloaders = []
        for ds in datasets:
            if self.base_task_name not in ds.Y_dict:
                raise ValueError(
                    f"Base task ({self.base_task_name}) labels missing from {ds}"
                )

            if self.base_input_name not in ds.X_dict:
                raise ValueError(f"{ds} must have {self.base_input_name} as X_dict key")

            dl = DictDataLoader(ds, **dataloader_kwargs)
            add_slice_labels(dl, self.base_task, S, slice_names)
            dataloaders.append(dl)

        return dataloaders

    @torch.no_grad()
    def score(
        self,
        dataloaders: List[DictDataLoader],
        as_dataframe: bool = False,
        eval_slices_on_base_task: bool = True,
    ) -> Dict[str, float]:

        eval_mapping : Dict[str, Optional[str]] = {}
        if eval_slices_on_base_task:
            # Collect all labels
            all_labels : Union[List, Set] = []
            for dl in dataloaders:
                all_labels.extend(dl.dataset.Y_dict.keys())  # type: ignore
            all_labels = set(all_labels)

            # By convention, evaluate on "pred" labels, not "ind" labels
            # See ``snorkel.slicing.utils.add_slice_labels`` for more about label creation
            eval_mapping.update({
                label: self.base_task_name for label in all_labels if "pred" in label
            })
            eval_mapping.update({
                label: None for label in all_labels if "ind" in label
            })

        return super().score(
            dataloaders=dataloaders,
            remap_labels=eval_mapping,
            as_dataframe=as_dataframe,
        )
