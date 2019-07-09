import shutil
import tempfile
import unittest
import os
import json

from snorkel.classification.training import TensorBoardWriter


class TestLogManager(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_tensorboard_writer(self):
        # Note: this just tests API calls. We rely on
        # tensorboardX's unit tests for correctness.
        run_name = "my_run"
        config = dict(a=8, b="my text")
        writer = TensorBoardWriter(run_name=run_name, log_root=self.test_dir)
        writer.add_scalar("my_value", value=0.5, step=2)
        writer.write_config(config)
        log_path = os.path.join(self.test_dir, run_name, "config.json")
        with open(log_path, "r") as f:
            file_config = json.load(f)
        self.assertEqual(config, file_config)
        writer.close()
