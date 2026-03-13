import numpy as np


# -------------------------------------------------
# Question 1 – Exponential Distribution
# -------------------------------------------------

def exponential_pdf(x, lam=1):
    """
    Return PDF of exponential distribution.
    f(x) = lam * exp(-lam*x) for x >= 0
    """
    if x < 0:
        return 0
    return lam * np.exp(-lam * x)


def exponential_interval_probability(a, b, lam=1):
    """
    Compute P(a < X < b) using analytical formula.
    P(a < X < b) = e^(-lam*a) - e^(-lam*b)
    """
    return np.exp(-lam * a) - np.exp(-lam * b)


def simulate_exponential_probability(a, b, n=100000, lam=1):
    """
    Simulate exponential samples and estimate
    P(a < X < b).
    """
    samples = np.random.exponential(1/lam, n)
    return np.mean((samples > a) & (samples < b))


# -------------------------------------------------
# Question 2 – Bayesian Classification
# -------------------------------------------------

def gaussian_pdf(x, mu, sigma):
    """
    Return Gaussian PDF.
    """
    coeff = 1 / (sigma * np.sqrt(2 * np.pi))
    exponent = np.exp(-((x - mu) ** 2) / (2 * sigma ** 2))
    return coeff * exponent


def posterior_probability(time):
    """
    Compute P(B | X = time)
    using Bayes rule.

    Priors:
    P(A)=0.3
    P(B)=0.7

    Distributions:
    A ~ N(40,4)
    B ~ N(45,4)
    """


    P_A = 0.3
    P_B = 0.7

    mu_A = 40
    mu_B = 45

    num = P_B * np.exp(-(time - mu_B)**2 / 4)
    den = P_A * np.exp(-(time - mu_A)**2 / 4) + num

    return num / den

def simulate_posterior_probability(time, n=100000):
    """
    Estimate P(B | X=time) using simulation.
    """

    P_A = 0.3
    P_B = 0.7

    mu_A = 40
    mu_B = 45
    sigma = 2  # variance = 4

    groups = np.random.choice(["A", "B"], size=n, p=[P_A, P_B])

    times = np.zeros(n)

    times[groups == "A"] = np.random.normal(mu_A, sigma, np.sum(groups == "A"))
    times[groups == "B"] = np.random.normal(mu_B, sigma, np.sum(groups == "B"))

    tol = 0.5
    mask = np.abs(times - time) < tol

    if np.sum(mask) == 0:
        return 0

    return np.mean(groups[mask] == "B")
