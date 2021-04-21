import numpy as np
from matplotlib import pyplot as plt

def gaussian_kl_divergence(q_mean, q_logvar, p_mean = 0, p_logvar = 0):
    """KL Divergence between two Gaussian distributions.

    Given q ~ N(mu_1, sigma^2_1) and p ~ N(mu_2, sigma^2_2), this function
    returns,

    KL(q||p) = log (sigma^2_2 / sigma^2_1) +
          (sigma^2_1 + (mu_1 - mu_2)^2) / (2 sigma^2_2) - 0.5

    Args:
    q_mean: Mean of proposal distribution.
    q_logvar: Log-variance of proposal distribution
    p_mean: Mean of prior distribution.
    p_logvar: Log-variance of prior distribution

    Returns:
    The KL divergence between q and p ( KL(q||p) ).
    """
    return 0.5 * (p_logvar - q_logvar + (np.exp(q_logvar) \
                      + (q_mean - p_mean)**2) / np.exp(p_logvar) - 1)



def posterior_collapse(q_mean, q_logvar, p_mean=0, p_logvar=0, delta = 0.01, path='./post_collapse.pdf',title=None , max_threshold = 6):
    ''' posterior colapse '''

    kl = gaussian_kl_divergence(q_mean, q_logvar, p_mean, p_logvar) 
    #max_threshold = 6
    eps = np.linspace(0,max_threshold,40)
    collapse = []
    for e in eps:
        test1 = kl <= e
        prob1 = np.mean(test1,axis=0)
        test2 = prob1 >= (1-delta)
        collapse.append(np.mean(test2))

    plt.figure()
    plt.plot(eps,collapse)
    plt.grid()
    plt.xlabel(r"$\varepsilon$")
    plt.ylabel('Collapse (%)')
    plt.ylim([-0.01, 1.01])
    plt.xlim([0, max_threshold])
    if title is not None:
        plt.title(r'$\omega =$'+title)
    plt.savefig(path)

    return eps,collapse

def variance_collapse(var, delta = 0.01, path='./variance_collapse.pdf'):
    ''' variance colapse 
        Args:
        var: Array N x d where N is the number of observations and 
             d is the dimension 
    '''
    max_threshold = 0.13 
    eps = np.linspace(0,max_threshold,40)
    collapse = []
    for e in eps:
        test1 = var <= e
        prob1 = np.mean(test1,axis=0)
        test2 = prob1 >= (1-delta)
        collapse.append(np.mean(test2))

    plt.figure()
    plt.plot(eps,collapse)
    plt.grid()
    plt.xlabel(r"$\varepsilon$")
    plt.ylabel('Variance collapse (%)')
    plt.ylim([-0.01, 1.01])
    plt.xlim([0, max_threshold])
    plt.title('Variance collapse')
    plt.savefig(path)

    return eps,collapse 

if __name__ == "__main__":
    x = np.random.randn(10,3)
    q_mean = np.random.randn(10,3)
    q_logvar = np.random.randn(10,3)

    kl = gaussian_kl_divergence(q_mean, q_logvar)
    posterior_collapse(q_mean, q_logvar)
