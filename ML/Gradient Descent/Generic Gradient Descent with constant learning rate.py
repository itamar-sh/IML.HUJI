import numpy as np

from IMLearn import BaseModule


class GradientDescent_constant_eta:
    def __init__(self, learning_rate: float,
                 tol: float = 1e-5,
                 max_iter: int = 1000):
        self.learning_rate_ = learning_rate
        self.tol_ = tol
        self.max_iter_ = max_iter

    def fit(self, f: BaseModule, X: np.ndarray, y: np.ndarray):
        output = []
        for t in range(self.max_iter_):
            # gradient = f.compute_jacobian(X=X, y=y)   # dont necessarily use X, y. It can be constant. so return self.weight
            gradient = np.sign(f.weights)
            step_val = self.learning_rate_ * gradient

            f.weights = f.weights - step_val

            output.append(f.weights)  # Average output

            delta_val = np.linalg.norm(step_val)

            if delta_val < self.tol_:
                break

        return np.mean(output, axis=0)  # average case

if __name__ == '__main__':
    f = BaseModule(np.array([np.sqrt(2), np.e / 3]))
    gd = GradientDescent_constant_eta(0.1)
    empty_val = np.empty(0)
    empty_val2 = np.empty(0)
    result = gd.fit(f, empty_val, empty_val2)
    print(result)

