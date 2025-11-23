def union(a,b):
    n=len(a)
    c=[]
    for i in range(n):
        c[i]=max(a[i],b[i])
    return c

def intersection(a,b):
    n=len(a)
    c=[]
    for i in range(n):
        c[i]=min(a[i],b[i])

    return c

def complement(a):
    n=len(a)
    c=[]

    for i in range(n):
        c[i]=1-a[i]

    return c

import numpy as np

def min_max(a,b):
    m,n=a.shape
    n2,p=b.shape

    if n!=n2:
        return

    R = np.zeros((m, p))

    for i in range(m):
        for j in range(p):
            mins=[]

            for k in range(n):
                minVal=min(a[i][k],b[k][j])
                mins.append(minVal)

            R[i][j]=max(mins)

    return R

def max_product(a,b):
    m,n=a.shape
    n2,p=b.shape

    if n!=n2:
        return

    R = np.zeros((m, p))

    for i in range(m):
        for j in range(p):
            mins=[]

            for k in range(n):
                minVal=a[i][k]*b[k][j]
                mins.append(minVal)

            R[i][j]=max(mins)

    return R


def fuzzy_set_composition(P, R):
    """
    Computes the composition of a fuzzy set P with a fuzzy relation R (P o R).

    For fuzzy set P (1 x m) and fuzzy relation R (m x p), the resulting
    fuzzy set H' (1 x p) has elements H'[j] = max_i(min(P[i], R[i, j])).
    """
    # Ensure P is treated as a row vector (1 x m)
    P_vector = np.array(P).flatten()
    m = len(P_vector)
    m2, p = R.shape

    # Check for compatibility (length of P must equal rows in R)
    if m != m2:
        raise ValueError("Incompatible dimensions for fuzzy set composition. Length of P must equal the number of rows in R.")

    # Initialize the result fuzzy set H'
    H_prime = np.zeros(p)

    # Perform the fuzzy set composition
    for j in range(p):  # columns of R (elements of H')
        min_values = []
        for i in range(m):  # elements of P and rows of R
            # Compute min(P[i], R[i, j])
            min_val = min(P_vector[i], R[i, j])
            min_values.append(min_val)

        # Compute max of the minimums
        H_prime[j] = max(min_values)

    return H_prime
