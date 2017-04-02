import numpy as np
import time

EPSILON = 1e-10

def sum_bidiagonal_sup(S):
    """ PARAMS : S une matrice bidiagonale
        RETURN : La somme des carres de la diagonale sup
    Cette fonction est utilisee pour verifier la convergence de l'algorithme"""
    (n, m) = np.shape(S)
    r = min(n, m)
    
    s = 0
    for i in range(r-1):
        s += np.square(S[i,i+1])

    return s


def convergence_indicator(S):
    """ PARAMS : S une matrice bidiagonale
        RETURN : La somme termes extradiagonaux non nuls (a EPSILON pret)
    Cette fonction est utilisee pour montrer la convergence de l'algorithme (logbook)"""
    (n, m) = np.shape(S)
    r = min(n, m)
    
    s = 0
    for i in range(r-1):
        if (abs(S[i,i+1]) >= EPSILON):
            s += 1

    return s


def is_diagonal(S):
    """ PARAMS : S une matrice bidiagonale
        RETURN : True si la convergence est atteinte (ie. la matrice est diagonale)
                 False sinon
    Verifie la convergence de l'algorithme bidiagonal_to_diagonal
    Cela revient a regarder si la somme des carres de la diagonale sup
    est inferieure a EPSILON choisi arbitrairement"""
    return sum_bidiagonal_sup(S) < EPSILON


def compare_bidiagonal(A, B):
    """ PARAMS : A et B deux matrices bidiagonales avec des termes extrabidiagonaux
                 possiblements non nuls mais tres proches de 0 qu'on neglige
        RETURN : True si A = B
                 (a EPSILON pret et en negligeant les termes proches de 0)
                 False sinon
        Cette fonction sert a verifier la veracite de l'invariant U x S x V = BD """
    (n, m) = np.shape(A)
    (n2, m2) = np.shape(B)

    if (n != n2 or m != m2):
        return False

    r = min(n, m)

    for i in range(r - 1):       
        if (abs(A[i, i] - B[i, i]) >= EPSILON or
            abs(A[i, i+1] - B[i, i+1]) >= EPSILON):
            return False

    return True


def bidiagonal_to_diagonal(S):
    """ PARAMS : S une matrice bidiagonale
        RETURN : (U, S, V, logbook) qui correspondent a la decomposition SVD
        La matrice S converge vers une matrice diagonale
        Le logbook permet de verifier cette convergence a chaque iteration """

    d = time.clock()

    logbook = { 'time_exec': None, 'iteration': [], 'convergence': [] }
    
    (n, m) = np.shape(S)
    U = np.matrix(np.eye(n, n))
    V = np.matrix(np.eye(m, m))

    ite = 0

    while (not(is_diagonal(S))):
        # On ajoute les informations courantes au logbook
        logbook["iteration"].append(ite)
        logbook["convergence"].append(sum_bidiagonal_sup(S))

        # Puis on applique l'algorithme
        (Q1, R1) = np.linalg.qr(np.transpose(S), mode='complete')
        (Q2, R2) = np.linalg.qr(np.transpose(R1), mode='complete')
        S = R2
        U = U * Q2
        V = np.transpose(Q1) * V
        ite += 1

    f = time.clock()

    logbook["time_exec"] = f - d

    return (U, S, V, logbook)

    
        
    
    
