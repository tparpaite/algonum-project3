#!/usr/bin/python2.7
# coding: utf-8

import numpy as np
import matplotlib as mp
import matplotlib.pyplot as plt
from hyperplan import *

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


#Pour améliorer les img en couleurs compressées
def travail1(a):#moyenne sur les 4 voisins
    (n,p,q)=np.shape(a)
    q=3
    c=0
    for i in range(1,n-1):
        for j in range(1,p-1):
            for k in range(q):
                a[i,j,k]=(a[i+1,j,k]+a[i-1,j,k]+a[i,j+1,k]+a[i,j-1,k])/4.0
                c=c+1
    print(c)

def travail2(a):#moyenne les 8 voisins, uniformément
    (n,p,q)=np.shape(a)
    q=3
    c=0
    for i in range(1,n-1):
        for j in range(1,p-1):
            for k in range(q):
                moyenne=(a[i+1,j,k]+a[i-1,j,k]+a[i,j+1,k]+a[i,j-1,k])/4.0
                a[i,j,k]=(4*moyenne+a[i+1,j+1,k]+a[i+1,j-1,k]+a[i-1,j+1,k]+a[i-1,j-1,k])/8.0
                c=c+1
    print(" "*10+str(c))

def travail3(a):#rajoute un coeff sqrt(2) pour prendre en compte les 8 voisins
    (n,p,q)=np.shape(a)
    q=3
    c=0
    for i in range(1,n-1):
        for j in range(1,p-1):
            for k in range(q):
                moyenne=(a[i+1,j,k]+a[i-1,j,k]+a[i,j+1,k]+a[i,j-1,k])/4.0
                a[i,j,k]=(5.656854*moyenne+a[i+1,j+1,k]+a[i+1,j-1,k]+a[i-1,j+1,k]+a[i-1,j-1,k])/9.656854
                c=c+1
    print(" "*20+str(c))
    
def travail4(a):#pas top
    (n,p,q)=np.shape(a)
    q=3
    c=0
    coeff=0.05
    for i in range(1,n-1):
        for j in range(1,p-1):
            for k in range(q):
                moyenne=(a[i+1,j,k]+a[i-1,j,k]+a[i,j+1,k]+a[i,j-1,k])/4.0
                if(np.abs(moyenne-a[i,j,k])>coeff):
                    a[i,j,k]=(moyenne+a[i,j,k])/2.0
                    c=c+1
    print(" "*30+str(c))

