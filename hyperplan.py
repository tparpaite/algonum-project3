#!/usr/bin/python2.7
# coding: utf-8

import numpy as np
import matplotlib as mp
import matplotlib.pyplot as plt

# Retourne un vecteur associé à une matrice de Householder qui envoie x sur y
#
# x et y sont des vecteurs de même norme
#
# Exemple :
#    x=np.mat([[3],[4],[0]])
#    y=np.mat([[0],[0],[5]])
#    u_householder(x,y)
def u_householder(x,y):
    v=x-y
    n=np.linalg.norm(v)
    if(n!=0):
        return v/n
    else:
        return v

# Retourne le produit d'une matrice de Householder par un vecteur colonne
#
# u est le vecteur non nul associé à la matrice de Householder
# c est le vecteur colonne que l'on multiplie à la matrice de Householder
#
# Exemple ?????
def mult_col_householder_G(u,c):#u (dim: n*1,COLONNE) représente H mat.Householder
    res=(u.T)*c                 #c (dim: n*1,COLONNE)
    return c-(2*res[0,0]*u)     #calcul H*c, en O(n), renvoie une colonne

# Retourne le produit d'une matrice de Householder par une matrice M
# Revient à calculer les produits de la matrice par chaque colonne de M
#
# u est le vecteur non nul associé à la matrice de Householder
# M est le second facteur du produit
#
# Exemple ???
def mult_mat_householder_G(u,M):#u (dim: n*1,COLONNE) représente H mat.Householder
    (n,p)=np.shape(u)           #taille (n,1)
    (n_m,p_m)=np.shape(M)       #n_m=n
    R=np.matrix(np.zeros((n_m,p_m)))
    for j in range(p_m):
        R[:,j]=mult_col_householder_G(u,M[:,j])
    return R                    #calcul H*M en O(n_m*p_m)

def mult_l_householder_D(l,u):#u (dim: n*1,COLONNE) représente H mat.Householder
    res=l*u                   #l (dim: 1*n,LIGNE)
    return l-(2*res[0,0]*(u.T))   #calcul l*H, en O(n), renvoie une ligne

def mult_mat_householder_D(M,u):#u (dim: n*1,COLONNE) représente H mat.Householder
    (n,p)=np.shape(u)           #taille (n,1)
    (n_m,p_m)=np.shape(M)       #p_m=n
    R=np.matrix(np.zeros((n_m,p_m)))
    for i in range(n_m):
        R[i,:]=mult_l_householder_D(M[i,:],u)
    return R                    #calcul H*M en O(n_m*p_m)

def mat_bidiagonale(A):#prerequis n<=m
    (n,m)=np.shape(A)
    B=np.matrix(np.copy(A))
    Qleft=np.matrix(np.eye(n,n))
    Qright=np.matrix(np.eye(m,m))
    for i in range(n):
        Qa1=np.matrix(np.zeros((n,1)))#colonne
        Qa1[i:n,0]=B[i:n,i]
        Qa2=np.matrix(np.zeros((n,1)))#colonne
        Qa2[i,0]=np.linalg.norm(Qa1)
        Q1=u_householder(Qa1,Qa2)
        Qleft=mult_mat_householder_D(Qleft,Q1)
        B=mult_mat_householder_G(Q1,B)
        #for j in range(i+1,n):
        #    B[j,i]=0
        if(i<(m-2)):
            Qb1=np.matrix(np.zeros((m,1)))#colonne
            Qb1[i:m,0]=B[i,i:m]
            Qb2=np.matrix(np.zeros((m,1)))#colonne
            Qb2[i,0]=B[i,i]
            Qb2[i+1,0]=np.linalg.norm(B[i,(i+1):m])
            Q2=u_householder(Qb1,Qb2)
            Qright=mult_mat_householder_G(Q2,Qright)
            B=mult_mat_householder_D(B,Q2)
            #for j in range(i+2,m):
            #    B[i,j]=0
    return (Qleft,B,Qright)