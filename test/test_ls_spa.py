import unittest
import numpy as np
from ls_spa import ls_spa, ShapleyResults, merge_sample_mean, merge_sample_cov


class TestOnlineStats(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(128)

        n = 100
        self.old_N = 2*n
        self.new_N = 3*n
        N = self.old_N + self.new_N

        A = rng.standard_normal((n, 3*n))
        S = A @ A.T
        self.X = rng.multivariate_normal(np.zeros(n), S, N)


    def test_merge_sample_mean(self):
        batch_1 = self.X[:self.old_N]
        batch_2 = self.X[self.old_N:]

        old_mean = np.mean(batch_1, axis=0)
        new_mean = np.mean(batch_2, axis=0)
        full_mean = np.mean(self.X, axis=0)
        merged_mean = merge_sample_mean(old_mean, new_mean,
                                        self.old_N, self.new_N)
        np.testing.assert_almost_equal(full_mean, merged_mean)


    def test_merge_sample_cov(self):
        batch_1 = self.X[:self.old_N]
        batch_2 = self.X[self.old_N:]

        old_mean = np.mean(batch_1, axis=0)
        new_mean = np.mean(batch_2, axis=0)
        old_cov = np.cov(batch_1, rowvar=False, bias=True)
        new_cov = np.cov(batch_2, rowvar=False, bias=True)
        full_cov = np.cov(self.X, rowvar=False, bias=True)
        merged_cov = merge_sample_cov(old_mean, new_mean,
                                      old_cov, new_cov,
                                      self.old_N, self.new_N)
        np.testing.assert_almost_equal(full_cov, merged_cov)


class TestLSSPA(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(128)

        n = 100
        diagonal = np.sqrt(np.diag(np.arange(1, n+1)))
        self.diagonal = diagonal
        A = rng.standard_normal((n, n))
        X, _ = np.linalg.qr(A)

        self.X_train_easy = X @ diagonal
        self.X_test_easy = self.X_train_easy.copy()
        self.y_train_easy = X[:, 0]
        self.y_test_easy = self.y_train_easy.copy()

        hard_theta = rng.standard_normal(n)
        X_train_hard = rng.multivariate_normal(np.zeros(n), A @ A.T, n)
        self.X_train_hard = X_train_hard - np.mean(X_train_hard, axis=0,
                                                   keepdims=True)
        X_test_hard = rng.multivariate_normal(np.zeros(n), A @ A.T, n)
        self.X_test_hard = X_test_hard - np.mean(X_train_hard, axis=0,
                                                 keepdims=True)
        y_train_hard = self.X_train_hard @ hard_theta + rng.standard_normal(n)
        self.y_train_hard = y_train_hard - np.mean(y_train_hard)
        y_test_hard = self.X_test_hard @ hard_theta + rng.standard_normal(n)
        self.y_test_hard = y_test_hard - np.mean(y_test_hard)


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
                              max_samples=4, batch_size=2)
        np.testing.assert_almost_equal(theta_easy, easy_results.theta)

        theta_hard = np.linalg.lstsq(self.X_train_hard, self.y_train_hard,
                                     rcond=None)[0]
        hard_results = ls_spa(self.X_train_hard, self.X_test_hard,
                              self.y_train_hard, self.y_test_hard,
                              max_samples=4, batch_size=2)
        np.testing.assert_almost_equal(theta_hard, hard_results.theta)


    def test_rsquared(self):
        theta_hard = np.linalg.lstsq(self.X_train_hard, self.y_train_hard,
                                     rcond=None)[0]
        y_hat = self.X_test_hard @ theta_hard
        tss = np.sum(self.y_test_hard ** 2)
        rss = np.sum((self.y_test_hard - y_hat) ** 2)
        r_squared = 1 - rss / tss
        hard_results = ls_spa(self.X_train_hard, self.X_test_hard,
                              self.y_train_hard, self.y_test_hard,
                              max_samples=4, batch_size=2)
        np.testing.assert_almost_equal(r_squared, hard_results.r_squared)


    def test_regularization(self):
        # Test if the regularization parameter affects the output
        N, p = self.X_train_hard.shape
        X_train_regular = np.vstack((self.X_train_hard / np.sqrt(N),
                                     np.sqrt(0.1) * np.eye(p)))
        y_train_regular = np.concatenate((self.y_train_hard / np.sqrt(N),
                                          np.zeros(p)))
        theta_regular = np.linalg.lstsq(X_train_regular, y_train_regular,
                                        rcond=None)[0]
        result_regular = ls_spa(self.X_train_hard, self.X_test_hard,
                                self.y_train_hard, self.y_test_hard, reg=0.1,
                                max_samples=4, batch_size=2)
        np.testing.assert_almost_equal(theta_regular, result_regular.theta)


    def test_random_seed_consistency(self):
        # Test if using the same seed produces consistent results
        result1 = ls_spa(self.X_train_hard, self.X_test_hard,
                         self.y_train_hard, self.y_test_hard, seed=42,
                         max_samples=4, batch_size=2)
        result2 = ls_spa(self.X_train_hard, self.X_test_hard,
                         self.y_train_hard, self.y_test_hard, seed=42,
                         max_samples=4, batch_size=2)
        np.testing.assert_almost_equal(result1.attribution, result2.attribution)


    def test_correctness_easy(self):
        p = self.X_train_easy.shape[1]
        proposal = np.zeros(p)
        for i in range(p):
            with_p = self.X_train_easy[:, 0:i+1]
            without_p = self.X_train_easy[:, 0:i]
            theta_with_p = np.linalg.lstsq(with_p, self.y_train_easy,
                                           rcond=None)[0]
            theta_without_p = np.linalg.lstsq(without_p, self.y_train_easy,
                                              rcond=None)[0]
            y_hat_with_p = self.X_test_easy[:, 0:i+1] @ theta_with_p
            y_hat_without_p = self.X_test_easy[:, 0:i] @ theta_without_p
            tss = np.sum(self.y_test_easy ** 2)
            rss_with_p = np.sum((self.y_test_easy - y_hat_with_p) ** 2)
            rss_without_p = np.sum((self.y_test_easy - y_hat_without_p) ** 2)
            r_squared_with_p = 1 - rss_with_p / tss
            r_squared_without_p = 1 - rss_without_p / tss
            proposal[i] += r_squared_with_p - r_squared_without_p

        easy_results = ls_spa(self.X_train_easy, self.X_test_easy,
                              self.y_train_easy, self.y_test_easy,
                              max_samples=256*256, batch_size=256)
        np.testing.assert_almost_equal(proposal, easy_results.attribution)


if __name__ == '__main__':
    unittest.main()

