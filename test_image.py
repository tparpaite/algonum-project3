#!/usr/bin/python2.7
# coding: utf-8
import matplotlib as mp
import matplotlib.pyplot as plt
from image import *


#img_full=mp.image.imread("img/psy.png")           #       89*66
img_full=mp.image.imread("img/batman.png")        #width=151*89 =height
#img_full=mp.image.imread("img/couleurs2.png")     #      230*219
#img_full=mp.image.imread("img/grey.png")          #      260*322
#img_full=mp.image.imread("img/peint.png")         #      442*262
#img_full=mp.image.imread("img/img_takeoff.png")   #      400*300
#img_full=mp.image.imread("img/earth.png")         #      500*500
#img_full=mp.image.imread("img/lena.png")          #      512*512
#img_full=mp.image.imread("img/couleurs.png")      #      664*634
#img_full=mp.image.imread("img/beatles_summer.png")#     1000*491
#img_full=mp.image.imread("img/ice_tea.png")       #      940*572
#img_full=mp.image.imread("img/abbey_road.png")    #     1360*768
#img_full=mp.image.imread("img/beatles.png")       #     3240*2025~7min30

#print(img_full[0][0])

(n,p,q)=np.shape(img_full)
m=min(n,p)
pas=m/20
a=[]#liste des images successives
for k in range(1,m/pas/3):
    a.append(compression_k(img_full,k*pas))


def sub(a,b):
    return np.abs(a-b)

def gris_1(a):
    r=np.sqrt(a[0]**2+a[1]**2+a[2]**2)/1.9
    return [r,r,r]

def gris_2(a):
    r=np.abs((a[0]+a[1]+a[2])/3.1)
    return [r,r,r]

def traitement(a):
    (n,p,q)=np.shape(a)
    return [[gris_1(a[i][j])  for j in range(p)] for i in range(n)]

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
    
def travail4(a):
    (n,p,q)=np.shape(a)
    q=3
    c=0
    coeff=0.1
    for i in range(1,n-1):
        for j in range(1,p-1):
            for k in range(q):
                t=np.abs(np.linalg.norm(a[i,j])-np.linalg.norm([a[i+1,j],a[i-1,j],a[i,j+1],a[i,j-1]]))
                a[i,j,k]=(a[i+1,j,k]+a[i-1,j,k]+a[i,j+1,k]+a[i,j-1,k])/4.0
                c=c+1
    print(" "*30+str(c))

def travail5(a):#tres bon sur takeoff  BEST
    (n,p,q)=np.shape(a)
    q=3
    c=0
    coeff=0.05
    for i in range(0,n):
        for j in range(0,p):
            s=np.mean(a[i,j])
            if(s<0.1):
                for k in range(q):
                    a[i,j,k]=np.abs(s)/3.0
                    c=c+1
            if(s>0.9):
                for k in range(q):
                    a[i,j,k]=1
                    c=c+1
    print(" "*40+str(c))

def travail6(a):#tres bon sur takeoff  BEST
    (n,p,q)=np.shape(a)
    q=3
    c=0
    coeff=0.05
    for i in range(0,n):
        for j in range(0,p):
            s=np.mean(a[i,j])
            for k in range(q):
                if(a[i,j,k]<0):
                    a[i,j,k]=0
                    c=c+1
                if(a[i,j,k]>1):
                    a[i,j,k]=1
                    c=c+1
    print(" "*50+str(c))



nn=len(a)

a1=np.copy(a)
a2=np.copy(a)
a3=np.copy(a)
for k in range(nn):
    travail5(a1[k])
    travail4(a2[k])
    travail6(a3[k])


for k in range(nn):
    print(k+1)
    plt.subplot(221)
    plt.axis("off")
    plt.title("SANS")
    plt.imshow(a[k],interpolation='nearest')
    plt.subplot(222)
    plt.axis("off")
    plt.title("SANS")
    plt.imshow(a1[k],interpolation='nearest')
    plt.subplot(223)
    plt.axis("off")
    plt.title("AVEC 1")
    plt.imshow(a2[k],interpolation='nearest')
    plt.subplot(224)
    plt.axis("off")
    plt.title("AVEC 2")
    plt.imshow(a3[k],interpolation='nearest')
    plt.show()
    #plt.subplot(224)
    #plt.axis("off")
    #plt.title("AVEC 3")
    #plt.imshow((a3[k]),interpolation='nearest')
    #plt.show()






#essai d'animation...

#from time import sleep
#plt.ion()
#nb_images = len(a)
#image = plt.imshow(a[0])

#for k in np.arange(nb_images):
#    #print("image numero: %i")%i
#    image.set_data(a[k])
#    plt.draw()
#    sleep(0.50)






#décompose une image en 3 IMAGES : R, G, B

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
