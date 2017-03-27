#!/usr/bin/python2.7
# coding: utf-8

import numpy as np
from householder import *

# Retourne la décomposition d'une matrice A quelconque en Qleft,B,Qright :
# avec B bidiagonale supérieure et Qleft et Qright des matrices orthogonales
# on a A=Qleft*B*Qright
#
# A une matrice quelconque de taille (n,m)
def mat_bidiagonale(A):
    (n,m)=np.shape(A)
    B=np.matrix(np.copy(A))
    Qleft=np.matrix(np.eye(n,n))
    Qright=np.matrix(np.eye(m,m))
    MIN=min(n,m)
    for i in range(MIN):#pour gérer toutes les matrices,avec  n<=m et aussi m<n
        Qa1=np.matrix(np.zeros((n,1)))#colonne
        Qa1[i:n,0]=B[i:n,i]
        Qa2=np.matrix(np.zeros((n,1)))#colonne
        Qa2[i,0]=np.linalg.norm(Qa1)
        Q1=u_householder(Qa1,Qa2)#Qa1 et Qa2 sont de même norme
        Qleft=mult_mat_householder_D(Qleft,Q1)
        B=mult_mat_householder_G(Q1,B)
        #for j in range(i+1,n):
        #    B[j,i]=0
        if(i<(m-2)):
            Qb1=np.matrix(np.zeros((m,1)))#colonne
            Qb1[i:m,0]=B[i,i:m].T
            Qb2=np.matrix(np.zeros((m,1)))#colonne
            Qb2[i,0]=B[i,i]
            Qb2[i+1,0]=np.linalg.norm(B[i,(i+1):m])
            Q2=u_householder(Qb1,Qb2)#Qa1 et Qa2 sont de même norme
            Qright=mult_mat_householder_G(Q2,Qright)
            B=mult_mat_householder_D(B,Q2)
            #for j in range(i+2,m):
            #    B[i,j]=0
    return (Qleft,B,Qright)
