#!/usr/bin/python2.7
# coding: utf-8
from test import *
from householder import *


def definition(taille_max,valeur_borne,norme_max,epsilon):#definie les vecteurs usuelles pour tous les test_2
    n=np.random.randint(2,taille_max+1)
    valeur=np.random.randint(1,valeur_borne+1)
    norme=np.random.randint(1,norme_max+1)
    cond=True
    while(cond):#simule une boucle do-while pour avoir |x| et |y| != 0
        x=np.matrix(np.random.randint(-valeur,valeur,size=(n,1)))
        y=np.matrix(np.random.randint(-valeur,valeur,size=(n,1)))
        if(np.linalg.norm(x)*np.linalg.norm(y)>epsilon):
            cond=False
    x=x*(norme/np.linalg.norm(x))
    y=y*(norme/np.linalg.norm(y))        
    u=u_householder(x,y)# |x|=|y|
    H=np.matrix(np.eye(n)-2*u*u.T)
    return (x,y,u,H,n,valeur,norme)#tester dans test_u_householder_2

##########################################################

def test_u_householder_1(epsilon):
    print("Test * 1 * u_householder :")
    print("--------------------------")
    n=3
    x=np.mat([[3],[4],[0]])
    y=np.mat([[0],[0],[5]])
    u=u_householder(x,y)
    H=np.matrix(np.eye(n)-2*u*u.T)
    H_th=np.matrix([[0.64,-0.48,0.6],[-0.48,0.36,0.8],[0.6,0.8,0]])
    print_test(eg_matrix(H,H_th,epsilon),"matrice H")
    print_test(eg_matrix(y,H*x,epsilon),"H projette bien x sur y")
    print_test(eg_matrix(x,H*y,epsilon),"H projette bien y sur x")

def test_u_householder_2(nb_test,epsilon):#test aussi definition
    taille_max=50
    valeur_borne=100
    norme_max=1000
    print("Test * 2 * u_householder :")
    print("--------------------------")
    print("sur "+str(nb_test)+" vecteurs de norme entre 1 et "+str(norme_max)+",")
    print("de taille entre 2 et "+str(taille_max)+",")
    print("avec des valeurs entre -"+str(valeur_borne)+" et "+str(valeur_borne)+",")
    print("avec une erreur maximale de "+str(epsilon)+" :")
    j=0
    for i in range(nb_test):
        (x,y,u,H,n,valeur,norme)=definition(taille_max,valeur_borne,norme_max,epsilon)
        if((np.linalg.norm(x)-np.linalg.norm(y)<epsilon)\
           and (eg_matrix(H,H.T,epsilon))\
           and (eg_matrix(H*H.T,np.matrix(np.eye(n)),epsilon))\
           and (eg_matrix(H*x,y,epsilon))\
           and (eg_matrix(H*y,x,epsilon))):
           j=j+1 # |x|=|y|, H symétrique, orthogonale, projette x sur y et y sur x
    print_test(j==nb_test,"représentation des matrices de Householder")

##########################################################
    
def test_mult_col_householder_G_1(epsilon):
    print("Test * 1 * mult_col_householder_G :")
    print("-----------------------------------")
    n=3
    x=np.mat([[3],[4],[0]])
    y=np.mat([[0],[0],[5]])
    u=u_householder(x,y)
    c=np.random.randint(12,size=(n,1))
    R=mult_col_householder_G(u,c)
    H=np.matrix(np.eye(n)-2*u*(u.T))
    R_th=H*c
    print_test(eg_matrix(R,R_th,epsilon),"multiplication d'une matrice de Householder par une colonne")

def test_mult_col_householder_G_2(nb_test,epsilon):
    taille_max=500
    valeur_borne=100
    norme_max=1000
    print("Test * 2 * mult_col_householder_G :")
    print("-----------------------------------")
    print("sur "+str(nb_test)+" matrices,")
    print("de taille entre 2 et "+str(taille_max)+",")
    print("avec des valeurs entre -"+str(valeur_borne)+" et "+str(valeur_borne)+",")
    print("avec une erreur maximale de "+str(epsilon)+" :")
    j=0
    for i in range(nb_test):
        (x,y,u,H,n,valeur,norme)=definition(taille_max,valeur_borne,norme_max,epsilon)
        c=np.matrix(np.random.randint(-valeur,valeur,size=(n,1)))
        R=mult_col_householder_G(u,c)
        R_th=H*c
        if(eg_matrix(R,R_th,epsilon)):
            j=j+1
    print_test(j==nb_test,"multiplication d'une matrice de Householder par des colonnes quelconques")

def test_mult_mat_householder_G_1(epsilon):
    print("Test * 1 * mult_mat_householder_G :")
    print("-----------------------------------")
    n=3
    x=np.mat([[3],[4],[0]])
    y=np.mat([[0],[0],[5]])
    u=u_householder(x,y)
    M=np.matrix(np.random.randint(12,size=(n,n)))
    R=mult_mat_householder_G(u,M)
    H=np.matrix(np.eye(n)-2*u*(u.T))
    R_th=H*M
    print_test(eg_matrix(R,R_th,epsilon),"multiplication d'une matrice de Householder par une matrice")
    
def test_mult_mat_householder_G_2(nb_test,epsilon):
    taille_max=100
    valeur_borne=100
    norme_max=1000
    print("Test * 2 * mult_mat_householder_G :")
    print("-----------------------------------")
    print("sur "+str(nb_test)+" matrices aléatoires,")
    print("de taille entre (1 et "+str(taille_max)+") x (1 et "+str(taille_max)+"),")
    print("avec des valeurs entre -"+str(valeur_borne)+" et "+str(valeur_borne)+",")
    print("avec une erreur maximale de "+str(epsilon)+" :")
    j=0
    for i in range(nb_test):
        (x,y,u,H,n,valeur,norme)=definition(taille_max,valeur_borne,norme_max,epsilon)
        m=np.random.randint(1,taille_max+1) #M n'est pas forcement carre
        M=np.matrix(np.random.randint(-valeur,valeur,size=(n,m)))
        R=mult_mat_householder_G(u,M)
        R_th=H*M
        if(eg_matrix(R,R_th,epsilon)):
            j=j+1
    print_test(j==nb_test,"multiplication d'une matrice de Householder par des matrices quelconques")

##########################################################

def test_mult_l_householder_D_1(epsilon):
    print("Test * 1 * mult_l_householder_D :")
    print("---------------------------------")
    n=3
    x=np.mat([[3],[4],[0]])
    y=np.mat([[0],[0],[5]])
    u=u_householder(x,y)
    c=np.random.randint(12,size=(1,n))
    R=mult_l_householder_D(c,u)
    H=np.matrix(np.eye(n)-2*u*(u.T))
    R_th=c*H
    print_test(eg_matrix(R,R_th,epsilon),"multiplication d'une ligne par une matrice de Householder")

def test_mult_l_householder_D_2(nb_test,epsilon):
    taille_max=500
    valeur_borne=100
    norme_max=1000
    print("Test * 2 * mult_l_householder_D :")
    print("---------------------------------")
    print("sur "+str(nb_test)+" lignes,")
    print("de taille entre 2 et "+str(taille_max)+",")
    print("avec des valeurs entre -"+str(valeur_borne)+" et "+str(valeur_borne)+",")
    print("avec une erreur maximale de "+str(epsilon)+" :")
    j=0
    for i in range(nb_test):
        (x,y,u,H,n,valeur,norme)=definition(taille_max,valeur_borne,norme_max,epsilon)
        c=np.matrix(np.random.randint(-valeur,valeur,size=(1,n)))
        R=mult_l_householder_D(c,u)
        R_th=c*H
        if(eg_matrix(R,R_th,epsilon)):
            j=j+1
    print_test(j==nb_test,"multiplication de lignes quelconques par une matrice de Householder")

def test_mult_mat_householder_D_1(epsilon):
    print("Test * 1 * mult_mat_householder_D :")
    print("-----------------------------------")
    n=3
    x=np.mat([[3],[4],[0]])
    y=np.mat([[0],[0],[5]])
    u=u_householder(x,y)
    M=np.matrix(np.random.randint(12,size=(n,n)))
    R=mult_mat_householder_D(M,u)
    H=np.matrix(np.eye(n)-2*u*(u.T))
    R_th=M*H
    print_test(eg_matrix(R,R_th,epsilon),"multiplication d'une matrice par une matrice de Householder")
    
def test_mult_mat_householder_D_2(nb_test,epsilon):
    taille_max=100
    valeur_borne=100
    norme_max=1000
    print("Test * 2 * mult_mat_householder_D :")
    print("-----------------------------------")
    print("sur "+str(nb_test)+" matrices aléatoires,")
    print("de taille entre (1 et "+str(taille_max)+") x (1 et "+str(taille_max)+"),")
    print("avec des valeurs entre -"+str(valeur_borne)+" et "+str(valeur_borne)+",")
    print("avec une erreur maximale de "+str(epsilon)+" :")
    j=0
    for i in range(nb_test):
        (x,y,u,H,n,valeur,norme)=definition(taille_max,valeur_borne,norme_max,epsilon)
        m=np.random.randint(1,taille_max+1) #M n'est pas forcement carre
        M=np.matrix(np.random.randint(-valeur,valeur,size=(m,n)))
        R=mult_mat_householder_D(M,u)
        R_th=M*H
        if(eg_matrix(R,R_th,epsilon)):
               j=j+1
    print_test(j==nb_test,"multiplication de matrices quelconques par une matrice de Householder")

##########################################################

def test(nb_test,epsilon):
    # les tests_1 sont basiques
    # les tests_2 aléatoires sur un grand nombre d'exemples.
    d=time.clock()
    test_u_householder_1(epsilon)
    test_u_householder_2(nb_test,epsilon)
    test_mult_col_householder_G_1(epsilon)
    test_mult_col_householder_G_2(nb_test,epsilon)
    test_mult_mat_householder_G_1(epsilon)
    test_mult_mat_householder_G_2(nb_test,epsilon)
    test_mult_l_householder_D_1(epsilon)
    test_mult_l_householder_D_2(nb_test,epsilon)
    test_mult_mat_householder_D_1(epsilon)
    test_mult_mat_householder_D_2(nb_test,epsilon)
    f=time.clock()
    return (f-d)
    
nb_test=100
epsilon=10**(-10)
t=test(nb_test,epsilon)
print("Temps d'exécution des "+str(nb_test)+" tests : "+str(int(10*t)/10.0)+" sec.\n")
# /!\ les temps sont seulement indicatifs :
# 10000 ->  env.  5'30"
#  1000 ->  env.    32"
#   500 ->  env.    17"
#   100 ->  env.     3"
