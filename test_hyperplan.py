#!/usr/bin/python2.7
# coding: utf-8
from hyperplan import *
import time

def print_test(boolean,msg):
    if(boolean):
        print("OK"+" : "+msg+"\n")
    else:
        print("ERROR"+" : "+msg+"\n")

def eg_matrix(A,B,epsilon):
    (n,m)=np.shape(A)
    if((n,m)!=np.shape(B)):
        return False
    else:
        return (np.linalg.norm([[A[i,j]-B[i,j] for j in range(m)] for i in range(n)])<epsilon)
        
def test_u_householder(epsilon):
    print("Test * 1 * u_householder :")
    print("--------------------------")
    n=3
    x=np.mat([[3],[4],[0]])
    y=np.mat([[0],[0],[5]])
    u=u_householder(x,y)
    H=np.eye(n)-2*u*u.T
    H_th=np.matrix([[0.64,-0.48,0.6],[-0.48,0.36,0.8],[0.6,0.8,0]])
    print_test(eg_matrix(H,H_th,epsilon),"matrice H")
    print_test(eg_matrix(y,H*x,epsilon),"H projette bien x sur y")
    print_test(eg_matrix(x,H*y,epsilon),"H projette bien y sur x")

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
    print("Test * 2 * mult_col_householder_G :")
    print("-----------------------------------")
    print("sur "+str(nb_test)+" matrices,")
    print("de taille entre 1 et "+str(taille_max)+",")
    print("avec des valeurs entre -"+str(valeur_borne)+" et "+str(valeur_borne)+",")
    print("avec une erreur maximale de "+str(epsilon)+" :")
    j=0
    for i in range(nb_test):
        n=np.random.randint(1,taille_max+1)
        valeur=np.random.randint(1,valeur_borne)
        x=np.matrix(np.random.randint(-valeur,valeur,size=(n,1)))# /!\ |x|=|y|
        y=np.matrix(np.random.randint(-valeur,valeur,size=(n,1)))
        u=u_householder(x,y)
        c=np.matrix(np.random.randint(-valeur,valeur,size=(n,1)))
        R=mult_col_householder_G(u,c)
        H=np.matrix(np.eye(n)-2*u*(u.T))
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
    print("Test * 2 * mult_mat_householder_G :")
    print("-----------------------------------")
    print("sur "+str(nb_test)+" matrices,")
    print("de taille entre 1 et "+str(taille_max)+",")
    print("avec des valeurs entre -"+str(valeur_borne)+" et "+str(valeur_borne)+",")
    print("avec une erreur maximale de "+str(epsilon)+" :")
    j=0
    for i in range(nb_test):
        n=np.random.randint(1,taille_max+1)
        valeur=np.random.randint(1,valeur_borne)
        x=np.matrix(np.random.randint(-valeur,valeur,size=(n,1)))
        y=np.matrix(np.random.randint(-valeur,valeur,size=(n,1)))
        u=u_householder(x,y)
        m=np.random.randint(1,taille_max+1) #M n'est pas forcement carre
        M=np.matrix(np.random.randint(-valeur,valeur,size=(n,m)))
        R=mult_mat_householder_G(u,M)
        H=np.matrix(np.eye(n)-2*u*(u.T))
        R_th=H*M
        if(eg_matrix(R,R_th,epsilon)):
            j=j+1
    print_test(j==nb_test,"multiplication d'une matrice de Householder par des matrices quelconques")


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
    print("Test * 2 * mult_l_householder_D :")
    print("---------------------------------")
    print("sur "+str(nb_test)+" lignes,")
    print("de taille entre 1 et "+str(taille_max)+",")
    print("avec des valeurs entre -"+str(valeur_borne)+" et "+str(valeur_borne)+",")
    print("avec une erreur maximale de "+str(epsilon)+" :")
    j=0
    for i in range(nb_test):
        n=np.random.randint(1,taille_max+1)
        valeur=np.random.randint(1,valeur_borne)
        x=np.matrix(np.random.randint(-valeur,valeur,size=(n,1)))
        y=np.matrix(np.random.randint(-valeur,valeur,size=(n,1)))
        u=u_householder(x,y)
        c=np.matrix(np.random.randint(-valeur,valeur,size=(1,n)))
        R=mult_l_householder_D(c,u)
        H=np.matrix(np.eye(n)-2*u*(u.T))
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
    print("Test * 2 * mult_mat_householder_D :")
    print("-----------------------------------")
    print("sur "+str(nb_test)+" matrices,")
    print("de taille entre 1 et "+str(taille_max)+",")
    print("avec des valeurs entre -"+str(valeur_borne)+" et "+str(valeur_borne)+",")
    print("avec une erreur maximale de "+str(epsilon)+" :")
    j=0
    for i in range(nb_test):
        n=np.random.randint(1,taille_max+1)
        valeur=np.random.randint(1,valeur_borne)
        x=np.matrix(np.random.randint(-valeur,valeur,size=(n,1)))
        y=np.matrix(np.random.randint(-valeur,valeur,size=(n,1)))
        u=u_householder(x,y)
        m=np.random.randint(1,taille_max+1) #M n'est pas forcement carre
        M=np.matrix(np.random.randint(-valeur,valeur,size=(m,n)))
        R=mult_mat_householder_D(M,u)
        H=np.matrix(np.eye(n)-2*u*(u.T))
        R_th=M*H
        if(eg_matrix(R,R_th,epsilon)):
               j=j+1
    print_test(j==nb_test,"multiplication de matrices quelconques par une matrice de Householder")

def est_bidiag(M,Qleft,B,Qright,epsilon):
    (n,p)=np.shape(M)#on verifie Qleft*B*Qright=M, B bidiagonale et Qleft et Qright orthogonale
    m=Qleft*B*Qright
    l=Qleft*Qleft.T
    r=Qright*Qright.T
    return     (eg_matrix(M,m,epsilon)\
           and (np.linalg.norm([[0 if(i==j or j==i+1) else B[i,j] for j in range(p)] for i in range(n)])<epsilon)\
           and (np.linalg.norm(l)-np.sqrt(n)<epsilon)\
           and (np.linalg.norm(r)-np.sqrt(p)<epsilon))
        
        
def test_bidiagonale_1(epsilon):
    print("Test * 1 * bidiagonale :")
    print("------------------------")
    n=4
    M=np.random.randint(1200,size=(n,n))
    (Qleft,B,Qright)=mat_bidiagonale(M)
    print_test(est_bidiag(M,Qleft,B,Qright,epsilon),"cas matrice de taille    n,   n")
    M=np.random.randint(1200,size=(n,2*n))
    (Qleft,B,Qright)=mat_bidiagonale(M)
    print_test(est_bidiag(M,Qleft,B,Qright,epsilon),"cas matrice de taille    n, 2*n")
    M=np.random.randint(1200,size=(2*n,n))
    (Qleft,B,Qright)=mat_bidiagonale(M)
    print_test(est_bidiag(M,Qleft,B,Qright,epsilon),"cas matrice de taille  2*n,   n")
    
def test_bidiagonale_2(nb_test,epsilon):
    taille_max=50
    valeur_borne=100
    print("Test * 2 * bidiagonale :")
    print("------------------------")
    print("sur "+str(nb_test)+" matrices,")
    print("de taille entre 1 et "+str(taille_max)+",")
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

def test(nb_test,epsilon):
    test_u_householder(epsilon)
    test_mult_col_householder_G_1(epsilon)
    test_mult_col_householder_G_2(nb_test,epsilon)
    test_mult_mat_householder_G_1(epsilon)
    test_mult_mat_householder_G_2(nb_test,epsilon)
    test_mult_l_householder_D_1(epsilon)
    test_mult_l_householder_D_2(nb_test,epsilon)
    test_mult_mat_householder_D_1(epsilon)
    test_mult_mat_householder_D_2(nb_test,epsilon)
    test_bidiagonale_1(epsilon)
    test_bidiagonale_2(100,epsilon)

debut=time.clock()
nb_test=10000
epsilon=10**(-10)
#test(nb_test,epsilon)
fin=time.clock()
print(fin-debut)
# 10000 ->  env.  4'30"
#  1000 ->  env.    30"
#   500 ->  env.    20"
#   100 ->  env.    10"


###############################################################################

def extract_colors(img_full):#un array de dim n*p et avec q=3 composantes, RGB(*)
    (n,p,q)=np.shape(img_full)
    return [[[img_full[i][j][k] for j in range(p)] for i in range(n)] for k in range(q)]


def one_color(img_full):#idem(*) + conserve les composantes nulles pour l'affichage via plt.imshow
    (n,p,q)=np.shape(img_full)
    return [[[[0 for u in range(k)]+[img_full[i][j][k]]+[0 for v in range(k+1,3)] for j in range(p)] for i in range(n)] for k in range(q)]






### donne un array
#img_full=mp.image.imread("img_takeoff.png") #width=400*300=height
#img_full=mp.image.imread("peint.png")       #      442*262
#img_full=mp.image.imread("couleurs.png")    #      664*634
#img_full=mp.image.imread("couleurs2.png")   #      230*219
#img_full=mp.image.imread("earth.png")       #      500*500


#img_extract_rgb=extract_colors(img_full)
#img_extract_rgb=one_color(img_full)
#print(np.matrix(img_extract_rgb))


#plt.subplot(221)
#plt.axis("off")
#plt.title("NORMAL")
#plt.imshow(img_full)
#plt.subplot(222)
#plt.axis("off")
#plt.title("RED")
#plt.imshow(img_extract_rgb[0])
#plt.subplot(223)
#plt.axis("off")
#plt.title("GREEN")
#plt.imshow(img_extract_rgb[1])
#plt.subplot(224)
#plt.axis("off")
#plt.title("BLUE")
#plt.imshow(img_extract_rgb[2])

#plt.show()

