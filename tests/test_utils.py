import unittest
import os
import shutil
from pathlib import Path

import utils


class TestUtils(unittest.TestCase):
    models_dir = os.getcwd() + '/models'

    def setUp(self):
        os.mkdir(TestUtils.models_dir)

    def tearDown(self):
        shutil.rmtree(TestUtils.models_dir)

    def test_save_model(self):
        model = "This is dumb model"
        model_name = 'vgg11'
        expected_file_location = Path(TestUtils.models_dir + "/" + model_name + ".pth")
        utils.save_model(model, model_name, models_dir=TestUtils.models_dir)
        self.assertTrue(expected_file_location.is_file())
