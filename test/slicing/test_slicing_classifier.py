import unittest
from types import SimpleNamespace

import torch
import torch.nn as nn

from snorkel.analysis import Scorer
from snorkel.classification import DictDataLoader, DictDataset, Trainer
from snorkel.slicing import BinarySlicingClassifier, SFApplier, slicing_function


@slicing_function()
def f(x) -> int:
    return x.num > 42


@slicing_function()
def g(x) -> int:
    return x.num > 10


DATA = [3, 43, 12, 9, 3]


class SliceCombinerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        data_points = [SimpleNamespace(num=num) for num in DATA]
        applier = SFApplier([f, g])
        cls.S = applier.apply(data_points, progress_bar=False)

    def test_classifier(self):
        hidden_dim = 10
        input_dim = 2

        representation_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        slice_names = ["hello", "world"]
        slicing_cls = BinarySlicingClassifier(
            representation_net,
            slice_names=slice_names,
            head_dim=hidden_dim,
            name="BinarySlicingClassifier",
            scorer=Scorer(metrics=["f1"]),
        )

        # Repeated data value for [N x 2] dim Tensor
        X = torch.FloatTensor([(x, x) for x in DATA])
        # Alternating labels
        Y = torch.LongTensor([int(i % 2 == 0) for i in range(len(DATA))])

        dataset_name = "test_dataset"
        input_name = BinarySlicingClassifier.base_input_name
        label_name = BinarySlicingClassifier.base_task_name
        base_dataset = DictDataset(
            name=dataset_name,
            split="test",
            X_dict={input_name: X},
            Y_dict={label_name: Y},
        )

        dataloaders = slicing_cls.make_slice_dataloaders(
            datasets=[base_dataset], S=self.S, slice_names=slice_names, batch_size=4
        )
        
        trainer = Trainer(num_epochs=1)
        trainer.fit(slicing_cls, dataloaders)

        results = slicing_cls.score(dataloaders)

        # Check that we eval on 'pred' labels
        self.assertIn(f"{label_name}/test_dataset/test/f1", results)
        self.assertIn(f"{label_name}_slice:hello_pred/test_dataset/test/f1", results)
        self.assertIn(f"{label_name}_slice:world_pred/test_dataset/test/f1", results)

        # No 'ind' labels!
        self.assertNotIn(f"{label_name}_slice:hello_ind/test_dataset/test/f1", results)
        self.assertNotIn(f"{label_name}_slice:world_ind/test_dataset/test/f1", results)
