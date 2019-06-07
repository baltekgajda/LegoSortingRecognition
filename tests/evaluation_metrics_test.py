import unittest
import evaluation_metrics


class EvaluationMetricsTest(unittest.TestCase):
    def test_accuracy_on_binary_classification(self):
        y_true = [0, 1]
        y_pred = [1, 1]
        expected_result = 0.5
        actual_result = evaluation_metrics.accuracy(y_true, y_pred)
        self.assertEqual(expected_result, actual_result)

    def test_accuracy_on_multi_class_classification(self):
        y_true = [0, 1, 1, 2, 2]
        y_pred = [0, 0, 1, 1, 2]
        expected_result = 0.6
        actual_result = evaluation_metrics.accuracy(y_true, y_pred)
        self.assertEqual(expected_result, actual_result)

    def test_top_1_error(self):
        y_true = [1, 1]
        probs = [
            [0.9, 0.1],
            [0.8, 0.2]
        ]
        expected_result = 0
        actual_result = evaluation_metrics.top_n_error(y_true, probs, n=1)
        self.assertEqual(expected_result, actual_result)

    def test_top_5_error(self):
        y_true = [3, 5]
        probs = [
            [0.22, 0.18, 0.12, 0.08, 0.27, 0.13],
            [0.21, 0.19, 0.11, 0.09, 0.26, 0.14]
        ]
        expected_result = 0.5
        actual_result = evaluation_metrics.top_n_error(y_true, probs, n=1)
        self.assertEqual(expected_result, actual_result)

    def test_get_cumulative_ranks(self):
        y_true = [1, 1]
        probs = [
            [0.1, 0.8, 0.1],
            [0.1, 0.1, 0.8]
        ]
        expected_results = [1, 2]
        actual_result = evaluation_metrics.get_cumulative_ranks(y_true, probs)
        self.assertEqual(expected_results, actual_result)

    def test_get_rank(self):
        y_true = 0
        probs = [0.1, 0.1, 0.3, 0.5]
        expected_value = 3
        actual_value = evaluation_metrics.get_rank(y_true, probs)
        self.assertEqual(expected_value, actual_value)


if __name__ == '__main__':
    unittest.main()
