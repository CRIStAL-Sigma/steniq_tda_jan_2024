# from alpha_complex.alpha_complex import AlphaComplex
# from alpha_complex.spectrogram import Spectrogram

# import generic libraries
import matplotlib.pyplot as plt
import numpy as np
from numpy import pi as pi
from patch import Polygon as Polygon
import shapely.geometry as sg
import descartes
from scipy.spatial import Voronoi,voronoi_plot_2d
import scipy.signal as scisig
from itertools import combinations

def plot_ac_filtration(points, filtration, r, ax=None, voronoi=True):
    """ Generate plots of the simplicial (alpha) complexes in a filtration
    """
    # if ax is None:
    #     ax = plt.gca()

    # filtration = list(ac._st.get_filtration())
    # points = ac.get_points()
    points_aux = points.tolist()
    
    for i in range(-600,750,150):
        for j in range(-600,750,150):
            points_aux.append([i,j])
        

    vor = Voronoi(points_aux)

#   for i,ax in enumerate(axs.flatten()[1::]):
    # voronoi_plot_2d(vor, 
    #                 ax=ax, 
    #                 show_points=False, 
    #                 show_vertices=False,
    #                 linestyle='--')


    for indp_p,point in enumerate(points):
        # Get index of cell corresponding to point
        indvcv = vor.point_region 
        vcv = vor.regions[indvcv[indp_p]] # Get vertices indexes
        
        if np.any([j==-1 for j in vcv]): # Ignore unbounded cells
            continue
                        
        vcvs = vor.vertices[vcv] # Get VC vertices
        vc = Polygon(vcvs) # VC as a polygon
        disc2 = sg.Point(*point).buffer(np.sqrt(r)) # Discs of current filt.
        middle = vc.intersection(disc2)

        # disc =  plt.Circle(point, 
        #                    r, 
        #                    color='g', 
        #                    alpha=0.1, 
        #                    clip_on=True)
        # ax.add_patch(disc)
        
        # ax.plot(point[0],point[1],'r.')

       
        ax.add_patch(descartes.PolygonPatch(middle, fc='g', ec='k', alpha=0.2))

    # Not the brightest idea to make this...
    getvert = lambda filt, val: [v[0] for v in filt if (v[1]>0.0) and (v[1]<=val)]
    V = getvert(filtration, r)

    if voronoi:
        voronoi_plot_2d(vor, show_vertices=False, line_colors='orange',
                          line_width=2, line_alpha=0.6, point_size=2, ax=ax)

    for v in V:
        if len(v) == 3:
            ax.fill(points[v,0], points[v,1], color="k", alpha=.2)

        if len(v) == 2:   
            ax.plot(points[v,0],points[v,1],'k', linewidth=0.5)


    ax.plot(points[:,0],
            points[:,1],
            'ro',
            ms=2.0)


def find_zeros_of_spectrogram(S):
    """ Find the zeros of the spectrogram S as minima in 3x3 grids.
    Includes first/last rows/columns.
    """
    aux_S = np.zeros((S.shape[0]+2,S.shape[1]+2))+np.Inf
    aux_S[1:-1,1:-1] = S
    S = aux_S
    aux_ceros = ((S <= np.roll(S,  1, 0)) &
            (S <= np.roll(S, -1, 0)) &
            (S <= np.roll(S,  1, 1)) &
            (S <= np.roll(S, -1, 1)) &
            (S <= np.roll(S, [-1, -1], [0,1])) &
            (S <= np.roll(S, [1, 1], [0,1])) &
            (S <= np.roll(S, [-1, 1], [0,1])) &
            (S <= np.roll(S, [1, -1], [0,1])) 
            )
    [y, x] = np.where(aux_ceros==True)
    pos = np.zeros((len(x), 2)) # Position of zeros in norm. coords.
    pos[:, 0] = y-1
    pos[:, 1] = x-1
    return pos


class Chain:
    """
    A simple "Chain" class for playing with the cycles
    (JMM: definitely not the best way of doing this, 
    but it's kind of faster than computing the boundary matrix...)
    """
    def __init__(self, comps):
        """ The parameter comps is a list of 1-simplices
        """
        if type(comps)==list:
            comps = np.array(comps)

        self.comps = comps
        # self.order_components()
            
    def __add__(self,b):
        """ Define addition of chains in the integer modulo 2 field
        """
        if not self.comps.size:
             comps = b.comps
        elif not b.comps.size:
             comps = self.comps
        else:
            comps = np.vstack((self.comps,b.comps))
        
        arr, uniq_cnt = np.unique(comps, axis=0, return_counts=True)
        arr = arr[np.mod(uniq_cnt,2)==1]
        c = Chain(arr)
        # c.order_components()
        return c



import matplotlib.path as mpltPath
def get_mask_from_triangles(S, triangles, vertices, mask=None):
    """ A function to get the mask from the triangles 
    ( JMM: this is the fastest way I found for now, to be improved)

    Args:
        S (ndarray): Spectrogram
        TRI (ndarray): Triangle
        vertices (ndarray): Vertices of the triangles

    Returns:
        ndarray: A boolean mask
    """

    if mask is None:
        mask = np.zeros(S.shape).astype(bool)
    
    triangles = triangles.astype(int)
    # inside2 = np.zeros((points.shape[0],)).astype(bool)
    for tri in triangles:
            min_row, min_col = np.min(vertices[tri,:],axis=0)
            max_row, max_col = np.max(vertices[tri,:],axis=0)   
            points = np.array([[i,j] for i in range(int(max_row+1-min_row)) for j in range(int(max_col+1-min_col))])
            tri_vert = vertices[tri,:] - [int(min_row), int(min_col)]
            path = mpltPath.Path(tri_vert)
            inside2 = path.contains_points(points)
            points = points + [int(min_row), int(min_col)]
            for point in points[inside2,:]:         
                mask[tuple(point)] = True
    return mask


def compute_boundary_operators(filtration, birth=-np.inf,death=np.inf,return_simplex=False):
    simplex0 = np.array([simplex[0] for simplex in filtration if len(simplex[0])==1 and simplex[1]>=birth and simplex[1]<=death]).astype(int)
    simplex1 = np.array([simplex[0] for simplex in filtration if len(simplex[0])==2 and simplex[1]>=birth and simplex[1]<=death]).astype(int)
    simplex2 = np.array([simplex[0] for simplex in filtration if len(simplex[0])==3 and simplex[1]>=birth and simplex[1]<=death]).astype(int)

    n0 = simplex0.shape[0]
    n1 = simplex1.shape[0]
    n2 = simplex2.shape[0]

    B1 = np.zeros((n1,n0),dtype=int)
    B2 = np.zeros((n2,n1),dtype=int)

    for i,s2 in enumerate(simplex2):
        for s1 in combinations(s2, 2):
            B2[i,(simplex1 == s1).all(axis=1)] = 1
    B2 = B2.T

    for i in range(B2.shape[0]):
        row = B2[i]
        if np.sum(row)>1:
            for ind,j in enumerate(np.where(row==1)[0]):
                B2[i,j]=(-1)**ind

    for i,s1 in enumerate(simplex1):
        for s0 in s1:
            B1[i,np.where(simplex0==s0)] = 1
    B1 = B1.T

    if return_simplex:
        return B1,B2,simplex1,simplex2
    else:
        return B1,B2
    

def get_chain(simplices,simplex_list):
    aux = np.zeros((simplex_list.shape[0]))
    for simplex in simplices:
        aux[(simplex_list == simplex).all(axis=1)] = 1
    
    return aux


def low(a):
    j=-1
    for i in range(len(a)):
        if a[i] != 0:
            j = i
    return j

def reduction(A):
    reduct_list = []
    flag = True
    while flag:
        flag = False
        for i in range(A.shape[1]):
            li = low(A[:,i])
            for j in reversed(range(i+1,A.shape[1])):    
                lj = low(A[:,j])
                if li == lj and lj != -1:
                    reduct_list.append([j,i])
                    flag = True
                    alpha = A[li,j]//A[lj,i]
                    A[:,j] -= alpha*A[:,i]
                    #print(A)

    return A, reduct_list

def find_representer(filtration,dj,return_simplex=False):
    """Finds a representer of a given (b,d) pair in a persistence diagram.

    Args:
        filtration (_type_): _description_
        dj (_type_): _description_
        return_simplex (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: _description_
    """
    simplex1 = np.array([simplex[0] for simplex in filtration if len(simplex[0])==2 and simplex[1]<=dj])
    simplex2 = np.array([simplex[0] for simplex in filtration if len(simplex[0])==3 and simplex[1]<=dj])
    B1,B2 = compute_boundary_operators(filtration, death=dj)
    B2red = reduction(B2.copy())[0]
    #print(B2red)
    #np.sum(B2red,axis=0)
    if return_simplex:
        return simplex1, simplex2, B2red[:,-1].astype(bool)
    else:
        return B2red[:,-1].astype(bool)
    

# Rossler
def rossler_sys(x):
    a = 0.398
    b = 2
    c = 4   
    y = np.zeros((3,))
    y[0] = -x[1]-x[2]
    y[1] = x[0]+a*x[1]
    y[2] = b+x[2]*(x[0]-c)

    return y

# Lorenz
def lorenz_sys(x):
    s = 10
    r = 28
    b = 8/3   
    y = np.zeros((3,))
    y[0] = s*(x[1]-x[0])
    y[1] = r*x[0]-x[1]-x[0]*x[2]
    y[2] = x[0]*x[1] - b*x[2]

    return y

def get_series_lorenz(N,n=1):
    h = 0.01
    t = np.arange(0,int((N+2000)*h),h)
    y0 = np.random.randn(3,)*h
    Y = np.zeros((len(t),3))
    Y[0] = lorenz_sys(y0)
    for i in range(1,len(t)):
        Y[i] = h*lorenz_sys(Y[i-1]) + Y[i-1]

    if n==1:    
        return Y[2000::]
    else:
        output = []
        for i in range(n):
            output.append(get_series_lorenz(N,n=1)[:,0])
        return np.array(output)      

def get_series_rossler(N,n=1):
    h = 0.01
    t = np.arange(0,int((N+2000)*h),h)
    y0 = np.random.randn(3,)
    Y = np.zeros((len(t),3))
    Y[0] = rossler_sys(y0)
    for i in range(1,len(t)):
        Y[i] = h*rossler_sys(Y[i-1]) + Y[i-1]

    if n==1:    
        return Y[2000::]
    else:
        output = []
        for i in range(n):
            output.append(get_series_rossler(N,n=1)[:,0])
        return np.array(output) 

import pandas as pd
def get_embedding(d,tau,x):
    x_emb = {'x1':[],'x2':[],'x3':[]}
    i = 0

    while i < len(x)-d*tau:
        for j,key in enumerate(x_emb):  
            x_emb[key].append(x[i+j*tau])
        i+=1

    x_emb = pd.DataFrame(x_emb)
    return x_emb
            
import gudhi as gd

def get_features_from_PD(points, sort=True):
    # Normalize points
    for dim in range(points.shape[1]):
        points[:,dim] -= np.min(points[:,dim])
        points[:,dim] /= np.max(points[:,dim])        
    st = gd.AlphaComplex(points).create_simplex_tree()
    persistence = st.persistence()
    
    # Get features from the PD    
    pairs = np.array([list(p[1]) for p in persistence if p[0]==1])
    pairs_lives = np.diff(pairs,axis=1)
    if sort:
        pairs_lives = np.sort(pairs_lives)#.reshape((len(pairs),))
    
    pairs_lives = pairs_lives.reshape((len(pairs),))
    return pairs_lives, persistence


def min_cycle_linprog_constraints(filtration, bdpair, perst_diagram):

    birth, death = bdpair
    alive_pairs = [pair[1] for pair in perst_diagram if pair[0]==1 and pair[1][0]<birth and pair[1][1]>birth]
    Z = []
    for pair in alive_pairs:
        bj,dj = pair
        simplex1, simplex2, zi = find_representer(filtration, dj, return_simplex=True)
        Z.append(zi)

    B1,B2 = compute_boundary_operators(filtration, death=birth, return_simplex=False)
    simplex0 = np.array([simplex[0] for simplex in filtration if len(simplex[0])==1 and simplex[1]<=birth])
    simplex1 = np.array([simplex[0] for simplex in filtration if len(simplex[0])==2 and simplex[1]<=birth])
    simplex2 = np.array([simplex[0] for simplex in filtration if len(simplex[0])==3 and simplex[1]<=birth])
    n0 = simplex0.shape[0]
    n1 = simplex1.shape[0]
    n2 = simplex2.shape[0]
    I = np.identity(n1)
    A = np.concatenate((I,-B2), axis=1)

    # Add other cycles to the restrictions
    def auxfun(z):
        L = simplex1.shape[0]
        a = np.zeros((L,))
        if len(z)<L:
            a[:len(z)] = z
            
        if len(z) >= L:
            a = z[:L]
        
        return a[:]

    Z = np.array([auxfun(z) for z in Z]).T
    A = np.concatenate((A,Z), axis=1)

    # Find initial cycle for the signficant bdpair
    zi = find_representer(filtration, death)
    zi = zi[:simplex1.shape[0]]

    # Define the linear program    
    nvars = n1 + n2 + Z.shape[1]
    c1 = np.zeros((nvars,))
    c1[:n1]=1
    c2 = np.zeros((nvars,))
    c2[n1::] = 1 
    
    return A, nvars, n1, zi, c1


def min_volume_linprog_constraints(filtration, bdpair):
    birth, death = bdpair
    source = [simplex[0] for simplex in filtration if simplex[1]==death][0]

    simplex1 = np.array([simplex[0] for simplex in filtration if len(simplex[0])==2 and simplex[1]<=death])
    n1 = simplex1.shape[0]

    simplex2 = np.array([simplex[0] for simplex in filtration if len(simplex[0])==3 and simplex[1]<=death])
    n2 = simplex2.shape[0]

    pre_simplex1 = np.array([simplex[0] for simplex in filtration if len(simplex[0])==2 and simplex[1]<birth])

    taus = np.array([simplex[0] for simplex in filtration if len(simplex[0])==2 
                                                            and simplex[1]>birth and simplex[1]<death])
    
    sigmas = np.array([simplex[0] for simplex in filtration if len(simplex[0])==3 
                                                            and simplex[1]>birth and simplex[1]<death])
    

    # Define the 1-simpleces living along the persistent volume
    T = np.zeros((n1,))
    for tau in taus:
        T[(simplex1 == tau).all(axis=1)] = 1
        
    simp_birth = np.array([simplex[0] for simplex in filtration if len(simplex[0])==2 and simplex[1]==birth])
    taui = np.zeros((n1,))
    taui[(simplex1 == simp_birth).all(axis=1)] = 1 

    # Define the initial 2-simplex
    zi = np.zeros((n2,))
    zi[(simplex2 == source).all(axis=1)] = 1 

    # Define the 2-simplices living along the persistent volume    
    # Z = []
    # for sigma in sigmas:
    #     z = np.zeros((n2,))
    #     z[(simplex2 == sigma).all(axis=1)] = 1 
    #     Z.append(z)

    # Z = np.array(Z).T

    # Z = np.concatenate((np.zeros((Z.shape[0],n2-Z.shape[1])),-Z), axis=1)
    
    # First restriction matrix
    A1 = np.identity(n2)
    # A1 = np.concatenate((I,-Z), axis=1)
    A1[:,n2-sigmas.shape[0]:-1] = 0


    # fig,ax = plt.subplots(1,1)
    # ax.imshow(A1)

    # Second restriction matrix
    _,B2 = compute_boundary_operators(filtration, death=death)

    # A2 = np.matmul(T,B2)
    # A2.resize(1,len(A2))

    # A3 = np.matmul(taui,B2)
    # Define the linear program    
    nvars = n2
    c1 = np.ones((nvars,))
    c2 = np.zeros((n1,))
 

    A3 = np.matmul(np.diag(T),B2)

    A = np.vstack((A1,A3))
    b = np.concatenate((zi,c2))
    
    return A, B2, nvars, b, c1, simplex2



def plot_ac_filtration_2(st, points, r, enum=True , ax=None):
    if ax is None:
        ax = plt.gca()

    filtration = list(st.get_filtration())
    vor = Voronoi(points)

    # Not the brightest idea to make this...
    getvert = lambda filt,val: [v[0] for v in filt if (v[1]>0.0) and (v[1]<=val)]

#   for i,ax in enumerate(axs.flatten()[1::]):
    # voronoi_plot_2d(vor, 
    #                 ax=ax, 
    #                 show_points=False, 
    #                 show_vertices=False,
    #                 linestyle='--')


    for indp_p,point in enumerate(points):
        # Get index of cell corresponding to point
        indvcv = vor.point_region 
        vcv = vor.regions[indvcv[indp_p]] # Get vertices indexes
        
        if np.any([j==-1 for j in vcv]): # Ignore unbounded cells
            # HalfspaceIntersection(halfspaces, interior_point, incremental=False, qhull_options=None)
            continue
                        
        vcvs = vor.vertices[vcv] # Get VC vertices
        vc = Polygon(vcvs) # VC as a polygon
        disc2 = sg.Point(*point).buffer(np.sqrt(r)) # Discs of current filt.
        middle = vc.intersection(disc2)

        # disc =  plt.Circle(point, 
        #                    r, 
        #                    color='g', 
        #                    alpha=0.1, 
        #                    clip_on=True)
        # ax.add_patch(disc)
        
        # ax.plot(point[0],point[1],'r.')

       
        # ax.add_patch(descartes.PolygonPatch(middle, fc='g', ec='k', alpha=0.2))


    V = getvert(filtration, r)
    #TODO Acá hay que ver si hay un triangulo o un lado.

    for v in V:
        if len(v) == 3:
            ax.fill(points[v,0], points[v,1], color="k", alpha=.2)

        if len(v) == 2:   
            ax.plot(points[v,0],points[v,1],'k', linewidth=0.5)

    ax.plot(points[:,0],
            points[:,1],
            'ro',
            ms=5.0)
    if enum:
        [ax.text(point[0],point[1], str(i), fontsize=10.0) for i,point in enumerate(points)]

    ax.set_axis_off()    