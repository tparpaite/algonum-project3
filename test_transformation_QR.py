import numpy as np
import matplotlib.pyplot as plt
from transformation_QR import *
from bidiag import *

N_TESTS = 10


def merge_logbook(logbook_array):
    logbook = { 'time_exec': None, 'iteration': None, 'convergence': [] }
    logbook["time_exec"] = np.mean([log["time_exec"] for log in logbook_array])
    n_iteration = np.amax([len(log["iteration"]) for log in logbook_array])
    logbook["iteration"] = range(n_iteration)

    # Mean for convergence field
    for i in range(n_iteration):
        s = 0
        for log in logbook_array:
            if (i < len(log["iteration"])):
                s += log["convergence"][i]
        mean = s / len(logbook_array)
        logbook["convergence"].append(mean)

    return logbook


def test_bidiagonal_to_diagonal():
    logbook_array = [None] * N_TESTS

    for i in range(N_TESTS):
        M = np.matrix(np.random.randint(1000, size=(50, 50)))
        QLeft, S_bidiag, QRight = mat_bidiagonale(M)

        (U, S_diag, V, logbook_array[i]) = bidiagonal_to_diagonal(S_bidiag)

        if (not(compare_bidiagonal(U * S_diag * V, S_bidiag))):
            print("Erreur : U x S x V = BD n'est pas verifie")
            exit(1)
        
    logbook_merge = merge_logbook(logbook_array)
    logbooks_to_figure(logbook_merge)


# Dessine la courbe en fonction des donnees contenues dans un logbook
# X : Nombre d'iterations
# Y : Convergence (somme des termes externes a la diagonale)
def create_curve(fig, logbook, color, label):
    X = logbook['iteration']
    Y = logbook['convergence']

    return fig.plot(X, Y, linestyle="-", marker="o", color=color, label=label, markevery=100)

    
def logbooks_to_figure(logbook_basic):
    fig, axes = plt.subplots(figsize=(15,6))

    # On ajuste les axes
    axes.set_xlabel("Iteration")
    axes.set_ylabel("Nombre d'elements extradiagonaux non nuls (convergence)")
    axes.set_xlim([0, 2000])
    
    # Creation des courbes
    curve_basic = create_curve(axes, logbook_basic, "b", "Algorithme basique")

    # Affichage graphique
    curves = curve_basic
    labels = [c.get_label() for c in curves]
    axes.legend(curves, labels, loc="best")

    # Sauvegarde au format PDF
    filename = "figure_convergence.pdf"
    fig.savefig(filename)
    fig.show()
    print filename + " successfully generated"
