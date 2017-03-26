#!/usr/bin/python2.7
# coding: utf-8
from hyperplan import *
import time

def extract_colors(img_full):#un array de dim n*p et avec q=3 composantes, RGB(*)
    (n,p,q)=np.shape(img_full)
    if(q!=3):
        q=3
    return [[[img_full[i][j][k] for j in range(p)] for i in range(n)] for k in range(q)]


def one_color(img_full):#idem(*) + conserve les composantes nulles pour l'affichage via plt.imshow
    (n,p,q)=np.shape(img_full)
    if(q!=3):
        q=3
    return [[[[0 for u in range(k)]+[img_full[i][j][k]]+[0 for v in range(k+1,3)] for j in range(p)] for i in range(n)] for k in range(q)]

# /!\ les types de retour de ces 2 fonctions sont bâtards :
# /!\ liste de liste de liste, donc à récupérer dans 3 NP.MATRIX différents




### donne un array
#img_full=mp.image.imread("img/batman.png")        #width=151*89 =height
#img_full=mp.image.imread("img/couleurs2.png")     #      230*219
#img_full=mp.image.imread("img/peint.png")         #      442*262
#img_full=mp.image.imread("img/img_takeoff.png")   #      400*300
#img_full=mp.image.imread("img/earth.png")         #      500*500
#img_full=mp.image.imread("img/couleurs.png")      #      664*634
#img_full=mp.image.imread("img/beatles_summer.png")#     1000*491
#img_full=mp.image.imread("img/ice_tea.png")       #      940*572
#img_full=mp.image.imread("img/abbey_road.png")    #     1360*768
#img_full=mp.image.imread("img/beatles.png")       #     3240*2025


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
