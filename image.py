#!/usr/bin/python2.7
# coding: utf-8

from bidiag import *

#img_full obtenue avec mp.image.imread("img/batman.png") par ex.
#img_full est un array de 3 matrix taille n*p
def extract_colors(img_full):#un array de dim n*p et avec q=3ou4 composantes, RGB(*)
    (n,p,q)=np.shape(img_full)
    if(q!=3):
        q=3#on enlève la transparence
    return [np.matrix([[img_full[i][j][k] for j in range(p)] for i in range(n)]) for k in range(q)]


def one_color(img_full):#idem(*) + conserve les composantes nulles pour l'affichage via plt.imshow
    (n,p,q)=np.shape(img_full)
    if(q!=3):
        q=3#on enlève la transparence
    return [[[[0 for u in range(k)]+[img_full[i][j][k]]+[0 for v in range(k+1,3)] for j in range(p)] for i in range(n)] for k in range(q)]
# /!\ les types de retour de cette fonction est bâtard :


#img une image sous la forme d'un array n*p*3ou4
def compression_k(img,k):
    (n,p,Q)=np.shape(img)
    m=min(n,p)
    Q=3
    A=extract_colors(img)#pour décomposer en usv sur des matrices n*p
    A=[np.linalg.svd(np.matrix(A[q])) for q in range(Q)]
    for q in range(Q):
        A[q][1][k:]=0#compression au rang k
    for q in range(Q):
        U=np.matrix(A[q][0])
        S=np.matrix([[A[q][1][i] if(i==j) else 0 for j in range(p)] for i in range(n)])
        V=np.matrix(A[q][2])
        A[q]=(U*S)*V
    B=np.zeros((n,p,Q))
    for i in range(n):
        for j in range(p):
            for q in range(Q):
                B[i][j][q]=A[q][i,j]
    print(k)
    return B

