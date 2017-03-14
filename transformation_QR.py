import numpy as np

EPSILON = 1e-10


def convergence_reached(S):
    """ PARAMS : S une matrice bidiagonale
        RETURN : 1 si la convergence est atteinte, 0 sinon
        Verifie la convergence de l'algorithme bidiagonal_to_diagonal
        Cela revient a regarder si la somme des carres de la diagonale sup
        est inferieure a EPSILON choisi arbitrairement"""
    (n, m) = np.shape(S)
    r = min(n, m)
    
    s = 0
    for i in range(r-1):
        s += np.square(S[i,i+1])

    return s < EPSILON


def bidiagonal_to_diagonal(S):
    """ PARAMS : S une matrice bidiagonale
        RETURN : (U, S, V) qui correspondent a la decomposition SVD
        La matrice S converge vers une matrice diagonale """
    (n, m) = np.shape(S)
    U = np.matrix(np.eye(n, n))
    V = np.matrix(np.eye(m, m))

    while (not(convergence_reached(S))):
        (Q1, R1) = np.linalg.qr(np.transpose(S))
        (Q2, R2) = np.linalg.qr(np.transpose(R1))
        S = R2
        U = U * Q2
        V = np.transpose(Q1) * V

    return (U, S, V)


def test_bidiagonal_to_diagonal():
    S_bidiag = np.matrix([[2, 4, 0, 0],
                   [0, 1, 3, 0],
                   [0, 0, 5, 2],
                   [0, 0, 0, 1],
                   [0, 0, 0, 0],
                   [0, 0, 0, 0]])

    (U, S_diag, V) = bidiagonal_to_diagonal(S_bidiag)

    print S_bidiag
    print S_diag

    print U * S_diag * V

    if (np.array_equal(U * S_diag * V, S_bidiag)):
        print("U x S x V = BD est verifie")
    else:
        print("U x S x V = BD n'est pas verifie")
    
        
    
    
