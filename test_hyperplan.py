from hyperplan import *

def print_test(boolean):
    if(boolean):
        print("OK\n")
    else:
        print("ERROR\n")

def test_u_householder():
    print("Test * 1 * u_householder :")
    print("--------------------------")
    epsilon=10**(-10)
    n=3
    x=np.mat([[3],[4],[0]])
    y=np.mat([[0],[0],[5]])
    u=u_householder(x,y)
    H=np.eye(n)-2*u*u.T
    H_th=np.matrix([[0.64,-0.48,0.6],[-0.48,0.36,0.8],[0.6,0.8,0]])
    print_test(np.linalg.norm(H-H_th)<epsilon)
    print_test(np.linalg.norm(y-H*x)<epsilon)
    print_test(np.linalg.norm(x-H*y)<epsilon)

def test_mult_col_householder_G_1():
    print("Test * 1 * mult_col_householder_G :")
    print("-----------------------------------")
    n=3
    epsilon=10**(-10)
    x=np.mat([[3],[4],[0]])
    y=np.mat([[0],[0],[5]])
    u=u_householder(x,y)
    c=np.random.randint(12,size=(n,1))
    R=mult_col_householder_G(u,c)
    H=np.matrix(np.eye(n)-2*u*(u.T))
    R_th=H*c
    print_test(np.linalg.norm(R-R_th)<epsilon)

def test_mult_col_householder_G_2():
    nb_test=100
    epsilon=10**(-10)
    taille_max=1000
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
        x=np.matrix(np.random.randint(-valeur,valeur,size=(n,1)))
        y=np.matrix(np.random.randint(-valeur,valeur,size=(n,1)))
        u=u_householder(x,y)
        c=np.matrix(np.random.randint(-valeur,valeur,size=(n,1)))
        R=mult_col_householder_G(u,c)
        H=np.matrix(np.eye(n)-2*u*(u.T))
        R_th=H*c
        if(np.linalg.norm(R-R_th)<epsilon):
            j=j+1
        #print(n,valeur,np.linalg.norm(R-R_th))
    #print(str(j/float(nb_test)*100)+"% des tests passes\n")
    print_test(j==nb_test)

def test_mult_mat_householder_G_1():
    print("Test * 1 * mult_mat_householder_G :")
    print("-----------------------------------")
    n=3
    epsilon=10**(-10)
    x=np.mat([[3],[4],[0]])
    y=np.mat([[0],[0],[5]])
    u=u_householder(x,y)
    M=np.matrix(np.random.randint(12,size=(n,n)))
    R=mult_mat_householder_G(u,M)
    H=np.matrix(np.eye(n)-2*u*(u.T))
    R_th=H*M
    print_test(np.linalg.norm(R-R_th)<epsilon)
    
def test_mult_mat_householder_G_2():
    nb_test=100
    epsilon=10**(-10)
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
        if(np.linalg.norm(R-R_th)<epsilon):
            j=j+1
        #print(n,m,valeur,np.linalg.norm(R-R_th))
    #print(str(j/float(nb_test)*100)+"% des tests passes\n")
    print_test(j==nb_test)


def test_mult_l_householder_D_1():
    print("Test * 1 * mult_col_householder_D :")
    print("-----------------------------------")
    n=3
    epsilon=10**(-10)
    x=np.mat([[3],[4],[0]])
    y=np.mat([[0],[0],[5]])
    u=u_householder(x,y)
    c=np.random.randint(12,size=(1,n))
    R=mult_l_householder_D(c,u)
    H=np.matrix(np.eye(n)-2*u*(u.T))
    R_th=c*H
    print_test(np.linalg.norm(R-R_th)<epsilon)

def test_mult_l_householder_D_2():
    nb_test=100
    epsilon=10**(-10)
    taille_max=1000
    valeur_borne=100
    print("Test * 2 * mult_col_householder_D :")
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
        c=np.matrix(np.random.randint(-valeur,valeur,size=(1,n)))
        R=mult_l_householder_D(c,u)
        H=np.matrix(np.eye(n)-2*u*(u.T))
        R_th=c*H
        if(np.linalg.norm(R-R_th)<epsilon):
            j=j+1
        #print(n,valeur,np.linalg.norm(R-R_th))
    #print(str(j/float(nb_test)*100)+"% des tests passes\n")
    print_test(j==nb_test)

def test_mult_mat_householder_D_1():
    print("Test * 1 * mult_mat_householder_D :")
    print("-----------------------------------")
    n=3
    epsilon=10**(-10)
    x=np.mat([[3],[4],[0]])
    y=np.mat([[0],[0],[5]])
    u=u_householder(x,y)
    M=np.matrix(np.random.randint(12,size=(n,n)))
    R=mult_mat_householder_D(M,u)
    H=np.matrix(np.eye(n)-2*u*(u.T))
    R_th=M*H
    print_test(np.linalg.norm(R-R_th)<epsilon)
    
def test_mult_mat_householder_D_2():
    nb_test=100
    epsilon=10**(-10)
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
        if(np.linalg.norm(R-R_th)<epsilon):
            j=j+1
        #print(n,m,valeur,np.linalg.norm(R-R_th))
    #print(str(j/float(nb_test)*100)+"% des tests passes\n")
    print_test(j==nb_test)

def est_bidiag(M,epsilon):
    C=np.copy(M)
    (n,m)=M.shape
    for i in range(n):
        C[i,i]=0
        if(i<(m-1)):
            C[i,i+1]=0
    return(np.linalg.norm(C)<epsilon)
        
        
def test_bidiagonale_1():
    print("Test * 1 * bidiagonale :")
    print("------------------------")
    epsilon=10**(-10)
    n=5
    M=np.random.randint(1200,size=(n,n))
    (Qleft,B,Qright)=mat_bidiagonale(M)
    #print(np.around(Qleft,decimals=2))
    #print("")
    #print(np.around(B,decimals=2))
    #print("")
    #print(np.around(Qright,decimals=2))
    #print(M)
    #print(np.around(Qleft*B*Qright,decimals=2))
    #print(np.linalg.norm(M-Qleft*B*Qright))
    #print(np.around(M-Qleft*B*Qright,decimals=2))
    print_test(est_bidiag(B,epsilon)==True)

def test_bidiagonale_2():
    nb_test=10
    epsilon=10**(-10)
    taille_max=100
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
        m=np.random.randint(n,taille_max+1) #M n'est pas forcement carre
        print(i,n,m)
        valeur=np.random.randint(1,valeur_borne)
        M=np.matrix(np.random.randint(-valeur,valeur,size=(n,m)))
        (Qleft,B,Qright)=mat_bidiagonale(M)
        if(est_bidiag(B,epsilon)==True):
            if(np.linalg.norm(M-Qleft*B*Qright)<epsilon):
                j=j+1
        #print(np.linalg.norm(M-Qleft*B*Qright))
    print_test(j==nb_test)

def test():
    test_u_householder()
    test_mult_col_householder_G_1()
    test_mult_col_householder_G_2()
    test_mult_mat_householder_G_1()
    test_mult_mat_householder_G_2()
    test_mult_l_householder_D_1()
    test_mult_l_householder_D_2()
    test_mult_mat_householder_D_1()
    test_mult_mat_householder_D_2()
    test_bidiagonale_1()
    test_bidiagonale_2()

test()



###############################################################################

def extract_colors(img_full):#un array de dim n*p et avec q=3 composantes, RGB(*)
    (n,p,q)=np.shape(img_full)
    return [[[img_full[i][j][k] for j in range(p)] for i in range(n)] for k in range(q)]


def one_color(img_full):#idem(*) + conserve les composantes nulles pour l'affichage via plt.imshow
    (n,p,q)=np.shape(img_full)
    return [[[[0 for u in range(k)]+[img_full[i][j][k]]+[0 for v in range(k+1,3)] for j in range(p)] for i in range(n)] for k in range(q)]






### donne un array
#img_full=mp.image.imread("img_takeoff.png") #width=400*300=height
#img_full=mp.image.imread("peint.png")      #      442*262
#img_full=mp.image.imread("couleurs.png")   #      664*634
#img_full=mp.image.imread("couleurs2.png")  #      230*219
#img_full=mp.image.imread("earth.png")      #      980*979

#img_extract_rgb=extract_colors(img_full)
#img_extract_rgb=one_color(img_full)

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
