import unittest
from functools import partial
import os
import shutil

import torch
import torch.nn as nn

from snorkel.classification.models.advanced import AdvancedClassifier, Operation, Task
from snorkel.classification.models.advanced.utils import ce_loss, softmax
from snorkel.classification.scorer import Scorer


class TaskTest(unittest.TestCase):

    def test_onetask_model(self):
        task1 = create_task("task1")
        model = AdvancedClassifier(tasks=[task1])
        self.assertEqual(len(model.task_names), 1)
        self.assertEqual(len(model.task_flows), 1)
        self.assertEqual(len(model.module_pool), 2)

    def test_twotask_all_overlap_model(self):
        """Add two tasks with identical modules and flows"""
        task1 = create_task("task1")
        task2 = create_task("task2")
        model = AdvancedClassifier(tasks=[task1, task2])
        self.assertEqual(len(model.task_names), 2)
        self.assertEqual(len(model.task_flows), 2)
        self.assertEqual(len(model.module_pool), 2)

    def test_twotask_none_overlap_model(self):
        """Add two tasks with totally separate modules and flows"""
        task1 = create_task("task1", module_suffixes=["A", "A"])
        task2 = create_task("task2", module_suffixes=["B", "B"])
        model = AdvancedClassifier(tasks=[task1, task2])
        self.assertEqual(len(model.task_names), 2)
        self.assertEqual(len(model.task_flows), 2)
        self.assertEqual(len(model.module_pool), 4)

    def test_twotask_partial_overlap_model(self):
        """Add two tasks with overlapping modules and flows"""
        task1 = create_task("task1", module_suffixes=["A", "A"])
        task2 = create_task("task2", module_suffixes=["A", "B"])
        model = AdvancedClassifier(tasks=[task1, task2])
        self.assertEqual(len(model.task_names), 2)
        self.assertEqual(len(model.task_flows), 2)
        self.assertEqual(len(model.module_pool), 3)

    def test_from_modules(self):
        task = create_task("task")
        modules = [task.module_pool["linear1"][0], task.module_pool["linear2"]]
        model_t = AdvancedClassifier(tasks=[task])
        model_m = AdvancedClassifier.from_modules(modules=modules)
        X_dict = {"data": torch.Tensor([0.4, 0.6])}
        t_out = model_t.forward(X_dict, ["task"])
        m_out = model_m.forward(X_dict, ["task"])
        self.assertTrue(torch.eq(t_out["op1"][0], m_out["op1"][0]).all())

    def test_save_load(self):
        CHECKPOINT_DIR = "test/classification/models/advanced/checkpoints"
        CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "model.pth")
        if not os.path.exists(CHECKPOINT_DIR):
            os.makedirs(CHECKPOINT_DIR)

        modules0 = [nn.Linear(2, 4), nn.Linear(4, 2)]
        modules1 = [nn.Linear(2, 4), nn.Linear(4, 2)]

        model = AdvancedClassifier.from_modules(modules=modules0)
        self.assertTrue(torch.eq(modules0[0].weight, model.module_pool["module0"].module[0].weight).all())
        model.save(CHECKPOINT_PATH)
        model = AdvancedClassifier.from_modules(modules=modules1)
        self.assertFalse(torch.eq(modules0[0].weight, model.module_pool["module0"].module[0].weight).all())
        model.load(CHECKPOINT_PATH)
        self.assertTrue(torch.eq(modules0[0].weight, model.module_pool["module0"].module[0].weight).all())

        if os.path.exists(CHECKPOINT_DIR):
            shutil.rmtree(CHECKPOINT_DIR)


def create_task(task_name, module_suffixes=("", "")):
    module1_name = f"linear1{module_suffixes[0]}"
    module2_name = f"linear2{module_suffixes[1]}"

    module_pool = nn.ModuleDict(
        {
            module1_name: nn.Sequential(nn.Linear(2, 4), nn.ReLU()),
            module2_name: nn.Linear(4, 2),
        }
    )

    op0 = Operation(module_name=module1_name, inputs=[("_input_", "data")], name="op0")
    op1 = Operation(module_name=module2_name, inputs=[(op0.name, 0)], name="op1")

    task_flow = [op0, op1]

    task = Task(
        name=task_name,
        module_pool=module_pool,
        task_flow=task_flow,
        loss_func=partial(ce_loss, op1.name),
        output_func=partial(softmax, op1.name),
        scorer=Scorer(metrics=["accuracy"]),
    )

    return task


if __name__ == "__main__":
    unittest.main()

