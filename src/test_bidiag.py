#!/usr/bin/python2.7
# coding: utf-8
from test import *
from bidiag import *

def est_bidiag(M,Qleft,B,Qright,epsilon):
    (n,p)=np.shape(M)
    m=Qleft*B*Qright #on verifie Qleft*B*Qright=M, B bidiagonale
    l=Qleft*Qleft.T  #Qleft et Qright orthogonales
    r=Qright*Qright.T
    return      (eg_matrix(M,m,epsilon)\
            and (eg_matrix(np.matrix([[0 if(i==j or j==i+1) else B[i,j] for j in range(p)] for i in range(n)]),np.matrix(np.zeros((n,p))),epsilon))\
            and (eg_matrix(l,np.matrix(np.eye(n)),epsilon))\
            and (eg_matrix(r,np.matrix(np.eye(p)),epsilon)))

def test_bidiagonale_1(epsilon):
    print("Test * 1 * bidiagonale :")
    print("------------------------")
    n=5
    M=np.random.randint(1000,size=(n,n))
    (Qleft,B,Qright)=mat_bidiagonale(M)
    print_test(est_bidiag(M,Qleft,B,Qright,epsilon),"cas matrice de taille    5,   5")
    M=np.random.randint(1000,size=(n,2*n))
    (Qleft,B,Qright)=mat_bidiagonale(M)
    print_test(est_bidiag(M,Qleft,B,Qright,epsilon),"cas matrice de taille    5,  10")
    M=np.random.randint(1000,size=(2*n,n))
    (Qleft,B,Qright)=mat_bidiagonale(M)
    print_test(est_bidiag(M,Qleft,B,Qright,epsilon),"cas matrice de taille   10,   5")
    
def test_bidiagonale_2(nb_test,taille_max,epsilon):
    valeur_borne=100
    print("Test * 2 * bidiagonale :")
    print("------------------------")
    print("sur "+str(nb_test)+" matrices aléatoires,")
    print("de taille entre (1 et "+str(taille_max)+") x (1 et "+str(taille_max)+"),")
    print("avec des valeurs entre -"+str(valeur_borne)+" et "+str(valeur_borne)+",")
    print("avec une erreur maximale de "+str(epsilon)+" :")
    j=0
    for i in range(nb_test):
        n=np.random.randint(1,taille_max+1)
        m=np.random.randint(1,taille_max+1) # M n'est pas forcement carré
        valeur=np.random.randint(1,valeur_borne)
        M=np.matrix(np.random.randint(-valeur,valeur,size=(n,m)))
        (Qleft,B,Qright)=mat_bidiagonale(M)
        if(est_bidiag(M,Qleft,B,Qright,epsilon)):
            j=j+1
    print_test(j==nb_test,"décomposition de matrices quelconques")

##########################################################

def test(nb_test,epsilon):
    # les tests_1 sont basiques
    # les tests_2 aléatoires sur un grand nombre d'exemples.
    d=time.clock()
    test_bidiagonale_1(epsilon)
    test_bidiagonale_2(nb_test,taille_max,epsilon)
    f=time.clock()
    return (f-d)
    
nb_test=100
taille_max=50
epsilon=10**(-10)
t=test(nb_test,epsilon)
print("Temps d'exécution des tests : "+str(int(10*t)/10.)+" sec.")
# /!\ les temps sont seulement indicatifs :
#nb_test, taille_max
#   1000,      50     ->  env.  1'30"
#    500,      50     ->  env.    42"
#    100,      50     ->  env.     9"

#    500,     100     ->  env.  2'45"
#    100,     100     ->  env.    43"

#    100,     250     ->  env.  3'10"
