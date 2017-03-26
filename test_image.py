#!/usr/bin/python2.7
# coding: utf-8

from image import *


#img_full=mp.image.imread("img/psy.png")           #       89*66
#img_full=mp.image.imread("img/batman.png")        #width=151*89 =height
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
pas=1
a=[]#liste des images successives
for k in range(pas,50,pas):
    a.append(compression_k(img_full,k))

nn=len(a)

a1=np.copy(a)
a2=np.copy(a)
a3=np.copy(a)
for k in range(nn):
    travail1(a1[k])
    travail2(a2[k])
    travail3(a3[k])

for k in range(nn):
    print(k+1)
    plt.subplot(221)
    plt.axis("off")
    plt.title("SANS")
    plt.imshow(a[k],interpolation='nearest')
    plt.subplot(222)
    plt.axis("off")
    plt.title("AVEC 1")
    plt.imshow(a1[k],interpolation='nearest')
    plt.subplot(223)
    plt.axis("off")
    plt.title("AVEC 2")
    plt.imshow(a2[k],interpolation='nearest')
    plt.subplot(224)
    plt.axis("off")
    plt.title("AVEC 3")
    plt.imshow(a3[k],interpolation='nearest')
    plt.show()



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
