from IMLearn.learners import UnivariateGaussian, MultivariateGaussian
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
import plotly.express as px
pio.templates.default = "simple_white"


def test_univariate_gaussian():
    # Question 1 - Draw samples and print fitted model
    univariateGaussian = UnivariateGaussian()
    univariateGaussian.fit(np.random.normal(10, 1, 1000))
    print((univariateGaussian.mu_, univariateGaussian.var_))

    # Question 2 - Empirically showing sample mean is consistent
    expectations = np.zeros(shape=(1, 100))
    for i in range(10, 1001, 10):
        samples = np.random.normal(10, 1, i)
        cur_mu = univariateGaussian.mu_
        univariateGaussian.fit(samples)
        expectations[0][i // 10 - 1] = abs(univariateGaussian.mu_-cur_mu)

    ms = np.linspace(0, 200, 200).astype(int)

    go.Figure([go.Scatter(x=ms*10, y=expectations[0], mode='markers+lines', name=r'$\widehat\mu$')],
              layout=go.Layout(title=r"$\text{Estimation of Expectation As Function Of Number Of Samples}$",
                               xaxis_title="$m\\text{ - number of samples}$",
                               yaxis_title="r$\hat\mu$",
                               height=400)).show()

    # Question 3 - Plotting Empirical PDF of fitted model
    values = np.random.normal(10, 1, 1000)
    univariateGaussian.fit(values)
    pdfs = univariateGaussian.pdf(values)

    go.Figure([go.Scatter(x=values, y=pdfs[0], mode='markers', name=r'$\widehat\mu$')],
              layout=go.Layout(title=r"$\text{PDF of each sample}$",
                               xaxis_title="value of sample",
                               yaxis_title="PDF value",
                               height=300)).show()

def test_multivariate_gaussian():
    # Question 4 - Draw samples and print fitted model
    multivariateGaussian = MultivariateGaussian()
    samples = np.random.multivariate_normal(np.array([0,0,4,0]),
                             np.array([[1.0  ,0.2, 0.0, 0.5],
                                       [0.2, 2.0 , 0.0,   0.0],
                                       [0.0 , 0.0 , 1.0,   0.0],
                                       [0.5, 0.0 , 0.0 ,  1.0]]),1000)
    multivariateGaussian.fit(samples)
    print(multivariateGaussian.mu_)
    print(multivariateGaussian.cov_)

    # Question 5 - Likelihood evaluation
    size = 200
    f1 = np.linspace(-10, 10, size)
    f3 = np.linspace(-10, 10, size)
    cur_cov = np.array([[1.0, 0.2, 0.0, 0.5],
                        [0.2, 2.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0, 0.0],
                        [0.5, 0.0, 0.0, 1.0]])
    results = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            cur_mu = np.array([f1[i], 0, f3[j], 0])
            results[i][j] = multivariateGaussian.log_likelihood(cur_mu, cur_cov, samples)

    pic = px.imshow(results, x=f3, y=f1, labels=dict(x="f3 values", y="f1 values"))
    pic.show()

    # Question 6 - Maximum likelihood
    max_f1 = f1[0]
    max_f3 = f3[0]
    max_loglikelihood = results[0][0]
    for i in range(size):
        for j in range(size):
            if results[i][j] > max_loglikelihood:
                max_loglikelihood = results[i][j]
                max_f1 = f1[i]
                max_f3 = f3[j]
    tup = round(max_f1, 4), round(max_f3, 4)
    print(tup)
    return tup


if __name__ == '__main__':
    np.random.seed(0)
    test_univariate_gaussian()
    test_multivariate_gaussian()
