#!/usr/bin/python2.7
# coding: utf-8
import matplotlib as mp
import matplotlib.pyplot as plt
from image import *

#img_full=mp.image.imread("img/psy.png")           #       89*66    k_max=37 
#img_full=mp.image.imread("img/batman.png")        #width=151*89 =height  55
#img_full=mp.image.imread("img/couleurs2.png")     #      230*219        111
#img_full=mp.image.imread("img/grey.png")          #      260*322        143
#img_full=mp.image.imread("img/peint.png")         #      442*262        164
#img_full=mp.image.imread("img/img_takeoff.png")   #      400*300        171
#img_full=mp.image.imread("img/earth.png")         #      500*500        249
#img_full=mp.image.imread("img/lena.png")          #      512*512        255
#img_full=mp.image.imread("img/couleurs.png")      #      664*634        324
#img_full=mp.image.imread("img/beatles_summer.png")#     1000*491        329
#img_full=mp.image.imread("img/ice_tea.png")       #      940*572        355
#img_full=mp.image.imread("img/abbey_road.png")    #     1360*768        490
#img_full=mp.image.imread("img/beatles.png")       #     3240*2025      1245

def traitement(a):
    (n,p,q)=np.shape(a)
    q=3
    c=0
    for i in range(0,n):
        for j in range(0,p):
            for k in range(q):
                if(a[i,j,k]<0):
                    a[i,j,k]=0
                    c=c+1
                if(a[i,j,k]>1):
                    a[i,j,k]=1
                    c=c+1
    return(c)

def sub_blanc(a,b):#utilise pour afficher la difference en l'image a et la b
    return np.ones(np.shape(a))-np.abs(a-b)

def sub_noir(a,b):#utilise pour afficher la difference en l'image a et la b
    return np.abs(a-b)

def distance_matrice(a,b):
    (n,p,q)=np.shape(a)
    s=a-b
    return np.linalg.norm([[np.linalg.norm(s[i,j]) for j in range(p)] for i in range(n)])

def abscisse(n,p):#choisi les valeurs de la compression pour 
    m=min(n,p)    #un graphe exploitable sans tracer tous les points
    max=(n*p)/(1+n+p)
    if(max>100):
        x=[i for i in range(3,10,1)]
        for i in range(10,max/10,2):
            x.append(i)
        for i in range(max/10+5,int(max/(2.5)),10):
            x.append(i)
        for i in range(int(max/(2.5))+5,m/2,15):
            x.append(i)
        for i in range(m/2+10,m,30):
            x.append(i)
        cond=True
        i=len(x)-1
        while(cond):
            if(x[i]>max):
                i=i-1
            else:
                x=x[:i+1]+[max]+x[i+1:]
                cond=False
        x.append(m)
    else:
        x=range(1,m+1,1)
    return x

def list_img_compresse(img_full):
    (n,p,q)=np.shape(img_full)
    max=(n*p)/(1+n+p)
    x=abscisse(n,p)
    print("")
    print("Nb de compressions : "+str(len(x)))
    print("Compression au rang k suivant :")
    print(x)
    print("")
    distance=[]
    L=[]
    print("Compression au rang k   Nb de composantes R, G ou B modifiées   Distance entre img_k et img")
    for g in x:
        a=compression_k(img_full,g)
        c=traitement(a)
        L.append(a)
        distance.append(distance_matrice(img_full,a))
        print("  "*6+str(g)+" "*25+str(c)+" "*30+str(int(100*distance[x.index(g)])/100.0))
    return (L,distance,x,n,p,max)

def graph_compression(img_full):
    (L,distance,x,n,p,max)=list_img_compresse(img_full)
    y=[np.sqrt(1/float(i)) for i in x]
    plt.subplot(121)
    plt.title("Efficacite de la compression SVD sur une image ")
    plt.xlabel("rang k de compression")
    plt.ylabel("distance entre l'originale et la compressee")
    pos=x.index(max)
    plt.plot(x,distance,':o',[x[pos]],[distance[pos]],'ro')
    plt.subplot(122)
    plt.title("de taille "+str(n)+"x"+str(p)+", donc une compression maximale de "+str(max))
    plt.xlabel("1/np.sqrt(k)")
    plt.ylabel("distance entre l'originale et la compressee")
    plt.plot(y,distance,':o',[y[pos]],[distance[pos]],'ro')
    plt.show()

def plot_img_compress_diff(img_full):
    (L,distance,x,n,p,max)=list_img_compresse(img_full)
    i=0
    for img in L:
        plt.subplot(121)
        plt.axis("off")
        plt.title("Image compresse au rang "+str(x[i]))
        plt.imshow(img,interpolation='nearest')
        plt.subplot(122)
        plt.axis("off")#selon moi, on mieux les différence sur un fond noir
        plt.title("Diff_noire entre img_rang_k et l'img_source")
        plt.imshow(sub_noir(img,img_full),interpolation='nearest')
        #plt.subplot(133)
        #plt.axis("off")
        #plt.title("Diff_blanche entre img_rang_k et l'img_source")
        #plt.imshow(sub_blanc(img,img_full),interpolation='nearest')
        plt.show()
        i=i+1

def aff_img_compressees(img_full):
    (n,p,q)=np.shape(img_full)
    max=(n*p)/(1+n+p)
    L=[5,10,20,50,100]
    a=[]
    i=0
    for k in L:
        a.append(compression_k(img_full,k))
        c=traitement(a[i])
        print(k)
        i=i+1
    plt.subplot(231)
    plt.axis("off")
    plt.title("Rang de compression k="+str(L[0]))
    plt.imshow(a[0])
    plt.subplot(232)
    plt.axis("off")
    plt.title("Rang de compression k="+str(L[1]))
    plt.imshow(a[1])
    plt.subplot(233)
    plt.axis("off")
    plt.title("Rang de compression k="+str(L[2]))
    plt.imshow(a[2])
    plt.subplot(234)
    plt.axis("off")
    plt.title("Rang de compression k="+str(L[3]))
    plt.imshow(a[3])
    plt.subplot(235)
    plt.axis("off")
    plt.title("Rang de compression k="+str(L[4]))
    plt.imshow(a[4])
    plt.subplot(236)
    plt.axis("off")
    plt.title("Image originale "+str((n,p))+", kmax="+str(max))
    plt.imshow(img_full)
    plt.show()
    
#décompose une image en 3 IMAGES : R, G, B
def aff_composante_img(img_full):
    img_extract_rgb=extract_colors(img_full)
    img_extract_rgb=one_color(img_full)
    plt.subplot(221)
    plt.axis("off")
    plt.title("3 COMPOSANTES")
    plt.imshow(img_full)
    plt.subplot(222)
    plt.axis("off")
    plt.title("RED")
    plt.imshow(img_extract_rgb[0])
    plt.subplot(223)
    plt.axis("off")
    plt.title("GREEN")
    plt.imshow(img_extract_rgb[1])
    plt.subplot(224)
    plt.axis("off")
    plt.title("BLUE")
    plt.imshow(img_extract_rgb[2])
    plt.show()


graph_compression(img_full)
#plot_img_compress_diff(img_full)
#aff_img_compressees(img_full)
#aff_composante_img(img_full)
