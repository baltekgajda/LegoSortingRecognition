import unittest

from torchvision.datasets import ImageFolder

from data_loader import reduce_dataset_size, split_dataset_into_subsets_dict, load_data


class TestDataLoader(unittest.TestCase):

    def setUp(self):
        self.data_path = '../data/Base Images'
        self.dataset = ImageFolder(root=self.data_path)

    def test_dataset_size_reduction(self):
        resize_factor = 0.3
        expected_reduced_dataset_size = int(len(self.dataset) * resize_factor)

        reduced_dataset = reduce_dataset_size(self.dataset, resize_factor)

        self.assertEqual(len(reduced_dataset), expected_reduced_dataset_size)

    def test_dataset_splitting(self):
        training_size = 0.6
        validation_size = 0.2

        subsets_dict = split_dataset_into_subsets_dict(self.dataset, training_size, validation_size)

        self.assertEqual(len(subsets_dict['train']), int(len(self.dataset) * training_size))
        self.assertEqual(len(subsets_dict['val']), int(len(self.dataset) * validation_size))
        subsets_length = len(subsets_dict['train']) + len(subsets_dict['val']) + len(subsets_dict['test'])
        self.assertEqual(subsets_length, len(self.dataset))

    def test_when_loading_data_then_subsets_are_split_with_given_proportion(self):
        training_size = 0.6
        validation_size = 0.2

        data_loaders_dict = load_data(
            data_path=self.data_path,
            input_size=224,
            train_size=training_size,
            val_size=validation_size,
        )

        self.assertAlmostEqual(
            len(data_loaders_dict['train'].sampler) / training_size,
            len(data_loaders_dict['val'].sampler) / validation_size
        )
        self.assertAlmostEqual(
            len(data_loaders_dict['train'].sampler) / training_size,
            len(data_loaders_dict['test'].sampler) / (1 - training_size - validation_size)
        )

    def test_when_loading_data_then_augmentation_is_applied_only_to_test_set(self):
        data_loaders_dict = load_data(
            data_path=self.data_path,
            input_size=224,
        )

        self.assertEqual(
            data_loaders_dict['val'].dataset.dataset.transform,
            data_loaders_dict['test'].dataset.dataset.transform
        )
        self.assertNotEqual(
            data_loaders_dict['train'].dataset.dataset.transform,
            data_loaders_dict['test'].dataset.dataset.transform
        )

    def test_when_loading_data_with_resize_parameter_then_subsets_are_reduced(self):
        loaders_dict = load_data(
            data_path=self.data_path,
            input_size=224,
        )
        reduced_loaders_dict = load_data(
            data_path=self.data_path,
            input_size=224,
            dataset_resize_factor=0.2
        )
        self.assertAlmostEqual(
            len(reduced_loaders_dict['train'].sampler),
            len(loaders_dict['train'].sampler) * 0.2,
            places=0  # round to integers
        )
        self.assertAlmostEqual(
            len(reduced_loaders_dict['val'].sampler),
            len(loaders_dict['val'].sampler) * 0.2,
            places=0
        )
        self.assertAlmostEqual(
            len(reduced_loaders_dict['test'].sampler),
            len(loaders_dict['test'].sampler) * 0.2,
            places=0
        )
