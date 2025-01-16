from math import comb

def binomial_probability(n : int , k : int, p : float) -> float:
    """
    This function calculates the probability of getting exactly k successes in n trials
    with the probability of success in a single trial being p.
    Args:
        n : int : number of trials
        k : int : number of successes
        p : float : probability of success in a single trial
    Returns:
        float : probability of getting exactly k successes in n trials
    Raises:
        ValueError : if n, k or p is invalid
    """

    if n < 0 or k < 0 or k > n or p < 0 or p > 1:
        raise ValueError("Invalid input")
    return comb(n, k) * p**k * (1 - p)**(n - k)