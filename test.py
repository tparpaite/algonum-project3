#!/usr/bin/python2.7
# coding: utf-8
import numpy as np
import time

def print_test(boolean,msg):#affiche le résultat du test
    if(boolean):
        print("OK"+" : "+msg+"\n")
    else:
        print("ERROR"+" : "+msg+"\n")

def eg_matrix(A,B,epsilon):#égalité sur les matrices, élmt par élmt
    (n,m)=np.shape(A)      #avec une erreur max de epsilon
    if((n,m)!=np.shape(B)):
        return False
    else:
        return (np.linalg.norm([[A[i,j]-B[i,j] for j in range(m)] for i in range(n)])<epsilon)
