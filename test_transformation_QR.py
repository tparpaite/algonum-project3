from transformation_QR import *

def test_bidiagonal_to_diagonal():
    S_bidiag = np.matrix([[2, 4, 0, 0],
                   [0, 1, 3, 0],
                   [0, 0, 5, 2],
                   [0, 0, 0, 1],
                   [0, 0, 0, 0],
                   [0, 0, 0, 0]])

    (U, S_diag, V) = bidiagonal_to_diagonal(S_bidiag)

    if (compare_bidiagonal(U * S_diag * V, S_bidiag)):
        print("U x S x V = BD est verifie")
    else:
        print("U x S x V = BD n'est pas verifie")


def test_convergence():
    
