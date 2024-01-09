import unittest
import numpy as np
from ls_spa import ls_spa, ShapleyResults


class TestLSSPA(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(128)

        n = 100
        diagonal = np.sqrt(np.diag(np.arange(1, n+1)))
        A = rng.standard_normal((n, n))
        X, _ = np.linalg.qr(A)

        self.X_train_easy = X @ diagonal
        self.X_test_easy = self.X_train_easy.copy()
        self.y_train_easy = X[:, 0]
        self.y_test_easy = self.y_train_easy.copy()

        hard_theta = rng.standard_normal(n)
        self.X_train_hard = rng.multivariate_normal(np.zeros(n), A @ A.T, n)
        self.X_test_hard = rng.multivariate_normal(np.zeros(n), A @ A.T, n)
        self.y_train_hard = (self.X_train_hard @ hard_theta
                             + rng.standard_normal(n))
        self.y_test_hard = (self.X_test_hard @ hard_theta
                            + rng.standard_normal(n))


    def test_return_type(self):
        # Test if the function returns an instance of ShapleyResults
        result = ls_spa(self.X_train_easy, self.X_test_easy,
                        self.y_train_easy, self.y_test_easy)
        self.assertIsInstance(result, ShapleyResults)


    def test_linear_regression(self):
        # Test if LSSPA finds the right theta
        theta_easy = np.linalg.lstsq(self.X_train_easy, self.y_train_easy,
                                     rcond=None)[0]
        easy_results = ls_spa(self.X_train_easy, self.X_test_easy,
                              self.y_train_easy, self.y_test_easy,
                              num_batches=2, batch_size=2)
        np.testing.assert_almost_equal(theta_easy, easy_results.theta)

        theta_hard = np.linalg.lstsq(self.X_train_hard, self.y_train_hard,
                                     rcond=None)[0]
        hard_results = ls_spa(self.X_train_hard, self.X_test_hard,
                              self.y_train_hard, self.y_test_hard,
                              num_batches=2, batch_size=2)
        np.testing.assert_almost_equal(theta_hard, hard_results.theta)


    # def test_regularization(self):
    #     # Test if the regularization parameter affects the output
    #     result_with_reg = ls_spa(self.X_train, self.X_test, self.y_train, self.y_test, reg=0.1)
    #     result_without_reg = ls_spa(self.X_train, self.X_test, self.y_train, self.y_test, reg=0.0)
    #     # Add your logic to compare results

    # def test_batch_size_effect(self):
    #     # Test if changing the batch size affects the output
    #     # Similar to test_regularization_effect, compare outputs with different batch sizes

    # def test_tolerance_effect(self):
    #     # Test if the tolerance parameter affects the output
    #     # Similar to test_regularization_effect, compare outputs with different tolerances

    # def test_random_seed_consistency(self):
    #     # Test if using the same seed produces consistent results
    #     result1 = ls_spa(self.X_train, self.X_test, self.y_train, self.y_test, seed=42)
    #     result2 = ls_spa(self.X_train, self.X_test, self.y_train, self.y_test, seed=42)
    #     # Compare result1 and result2

    # def test_invalid_input(self):
    #     # Test the function with invalid input and check if it raises appropriate errors
    #     with self.assertRaises(SomeExpectedException):
    #         ls_spa(invalid_input)

    # # You can add more test cases as needed

if __name__ == '__main__':
    unittest.main()

