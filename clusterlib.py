import csv
import numpy as np
import sys
from collections import defaultdict
from matplotlib.colors import *
from pylab import *
from matplotlib.widgets import Slider, Button, RadioButtons
import matplotlib.pyplot as plt
from collections import defaultdict,OrderedDict
from math import pow,pi,exp,sqrt
from itertools import product, combinations
import ROOT
barrel_phi_min=0#inclusive
barrel_phi_max=120#exclusive
barrel_theta_min=20#inclusive
barrel_theta_max=68#exclusive

def gauss_peak_norm_functor(mean,sigma,peak):
    my_mean = mean
    my_sigma = sigma
    my_peak = peak
    my_norm = peak/exp(-1*(0.0-mean)**2/(2.0*sigma**2))
    def ret(x):
        return exp(-1*(x-my_mean)**2/(2.0*my_sigma**2))*my_norm
    return ret

def distance(x,y):
    return sqrt(x**2+y**2)
def dis(xy1,xy2):
    return sqrt((xy1[0]-xy2[0])**2+(xy1[1]-xy2[1])**2)
#from Bertrand's Geant4 simulation
class RootData:
    def __init__(self,dat):
        self.row = dat.row
        self.eventId = dat.eventId
        self.column = dat.column
        self.energy = dat.energy
  
class HitMapFile:
    fname = None
    def __init__(self,fname=None):
        self.fname = fname
    def hitmaps(self):
        f = ROOT.TFile(self.fname)
        tree = f.Get('MyTuple')
        dd = defaultdict(HitMap.creator(12,44))
        for dat in tree:
            dd[dat.eventId].hits[dat.row,dat.column]+=dat.energy
        #f.Close()
        return dd.values()

class HitMap:
    hot_regions = [
        (0,0),(0,1),(0,2),(0,3),(0,4),(0,5),(0,6),(0,7),(0,8),(0,9),(0,10),(0,11),(0,12),(0,13),(0,14),(0,15),(0,16),(0,17),(0,18),(0,19),
        (1,0),(1,1),(1,2),(1,3),(1,4),(1,5),(1,6),(1,7),(1,8),(1,9),(1,10),(1,11),(1,12),
        (2,0),(2,1),(2,2),(2,3),(2,4)
        ]
    small_hot_regions = [
        (0,0),(0,1),(0,2),(0,3),(0,4),(0,5),(0,6),(0,7),(0,8),(0,9),(0,10),(0,11),(0,12),
        (1,0),(1,1),(1,2),(1,3),(1,4),(1,5),(1,6),
        (2,0),(2,1)
    ]
    def __init__(self,num_phi=12,num_theta=44):
        #hits is structred as theta,phi
        self.hits = np.zeros((num_phi,num_theta))
    
    def num_row(self):
        return self.hits.shape[0]
    
    def num_col(self):
        return self.hits.shape[1]
    
    def acc(self,phi,theta,energy): #x are array(eventno thetaindex, phiindex, energy) from v
        self.hits[phi,theta]+=energy #imshow maps first index to y axis and second to x axis
        
    def compute_laplacian(self):
        self.lpc = laplacian(self.hits)
        
    def sumE(self,cluster):
        return self.sumE_from_hits(self.hits,cluster)
    
    def mix_in(self,another_one):
        self.hits+=another_one.hits
    @classmethod
    def in_barrel(self,phi,theta):
        return 0<= phi< 12 and 0<= theta < 44
    
    @classmethod  
    def in_hot_region(self,phi,theta):
        return (phi,theta) in HitMap.small_hot_regions
    
    @classmethod
    def sumE_from_hits(self,hits,cluster):
        xlist,ylist = zip(*cluster)
        return np.sum(hits[xlist,ylist])

      
    @classmethod
    def creator(self,num_phi,num_theta):
        return lambda : HitMap(num_phi,num_theta)
        
    
"""
    Cluster class contains cluster as set of tuple (each one represent the point on the hitmap)
    and the seed is the original seed used in making this cluster
"""
class Cluster:
    def __init__(self,seed,cluster):
        self.cluster = cluster
        self.seed = seed

class Clustering:
    seed_low_cutoff = 20 #50MeV default
    seed_high_cutoff = 100 # not more than 100MeV
    directions = [[-1,0],[0,-1],[1,0],[0,1]] #how cluster look around uldr
    moire_r = 3.6/5.0 #in the unit of the crystal face length
    
    #the allow region is enveloped by two gaussian
    #upper is a very wide one with peak normalized to the seed energy
    #lower one is a narrow one with peak normalized to half the seed energy
    #there is also an absolute cutoff at expand_cutoff at 1 MeV
    
    cutoff_factor = 0.01
    #lower_sigma_factor_cutoff = 0.5
    #lower_norm_cutoff = 0.5
    
    def __init__(self):
        pass
        
    def use9x9dir(self):
        self.directions = [[-1,0],[0,-1],[1,0],[0,1],[-1,-1],[1,1],[-1,1],[1,-1]]
    
    def use25x25dir(self):
        x = range(-2,3)
        self.directions = [ (y,z) for y in x for z in x]
    
    def useDiamondDir(self):
        #9x9 plus the 2 straight in each direction
        self.directions = [[-1,0],[0,-1],[1,0],[0,1],[-1,-1],[1,1],[-1,1],[1,-1],[2,0],[-2,0],[0,2],[0,-2]]
        
    #return ordereddict of seed
    def find_seed(self,hitmap):
        hm=hitmap
        hot_region = zip(*HitMap.small_hot_regions)
        hm[hot_region[0],hot_region[1]]=-1.0 #cut off lower corner
        seedlist = zip(*np.where( (hm>self.seed_low_cutoff) & (hm<self.seed_high_cutoff)))
        seedlist.sort(key=lambda x: hm[x])        
        od = OrderedDict()
        for p in seedlist: 
            od[p] = hm[p]
        return od
    #return list of cluster object
    #also note that seedod is passed by reference and will have it value change(empty upon return)
    def find_clusters(self,hitmap,seedod=None):
        hm = hitmap.hits
        #making seedlist put in ordered dict
        if seedod is None: seedod = self.find_seed(hm)
        clusters = []
        seeds_ret = [] #list of seed used for each seed used by cluster
        while len(seedod)!=0:
            seed,E = seedod.popitem()
            if not hitmap.in_barrel(*seed): continue; #skip endcaps
            #print seed,E
            cluster_so_far = set()
            cluster_so_far.update([seed])
            this_cluster = self.expand_cluster(seed,hm,cluster_so_far,seed)
            #remove seed if seed in this cluster
            for hit_pos in this_cluster:
                if hit_pos in seedod: del seedod[hit_pos]
            clusters.append(Cluster(seed,this_cluster))
        return clusters

    #recursively expand cluster from given seed
    #note that cluster_so_far is passed by reference and it acts as accumulator
    #last argument is the original seed from this cluster(for calculating upper cutoff)
    def expand_cluster(self,seed,hits,cluster_so_far,org_seed):
        #look in 4 direction
        for direction in self.directions:
            neighbor = self.add_direction(seed,direction)
            if neighbor not in cluster_so_far and \
                HitMap.in_barrel(*neighbor) and not HitMap.in_hot_region(*neighbor) and\
                self.passcut(neighbor,hits,org_seed):
              
                cluster_so_far.update([neighbor])
                self.expand_cluster(neighbor,hits,cluster_so_far,org_seed)
        return cluster_so_far

    def add_direction(self,org,direc):
        #the modulo is for wrapping
        return tuple([org[0]+direc[0],org[1]+direc[1]])

    def passcut(self,pos,hits,org_seed):
        #dis = distance((pos[0]-org_seed[0])%barrel_phi_max,pos[1]-org_seed[1])
        
        low_cutoff = hits[org_seed]*self.cutoff_factor

        high_cutoff = hits[org_seed]
        return high_cutoff > hits[pos] > low_cutoff
    
    #note that this is pass by reference clusters will be changed
    def reduce_clusters(self,clusters,hits):
        #taking care of overlapping clusters
        #it first find all the crystals that is in two or more clusters
        #then for each one of them calculate the figure of merit based on seed energy and distance for the seed
        #the one with the highest figure of merits gets the whole crystal
        
        #first put them all in a map by seed
        cl_map = {c.seed : c.cluster for c in clusters}
        
        #dupe_map is map from crystal_pos to list of seed that has this crystal in it
        dupe_map = defaultdict(set)
        #build dupe_map
        for lhs_cl, rhs_cl in combinations(clusters, 2):
            #find intersection for all the intersection
            intersection = lhs_cl.cluster.intersection(rhs_cl.cluster)
            for overlap in intersection:
                dupe_map[overlap].update([lhs_cl.seed,rhs_cl.seed])
        #for each overlapped crystal compute the expected E and remove it from the low expected E clusters
        for crystal_pos, seed_list in dupe_map.items():
            expected_E = [(seed,gauss_peak_norm_functor(0,self.moire_r,hits[seed])(dis(crystal_pos,seed))) for seed in seed_list]
            best_seed = max(expected_E,key=lambda x:x[1])
            #now that we get the seed remove it from other list
            for seed in seed_list:
                if(seed!=best_seed[0]): 
                    cl_map[seed]-=set([crystal_pos])
        #now clusters should have no overlap
        #check if there is a seed with empty cluster
        clusters = [cl for cl in clusters if len(cl.cluster)!=0]
        return clusters
    
    def draw_cutoff(self,E=0.100,ax=None):
        if ax is None: ax = gca()
        #E=0.100
        #high_cutoff_g = gauss_peak_norm_functor(0,self.moire_r*self.upper_sigma_factor_cutoff,self.upper_norm_cutoff*E)
        #vhg = np.vectorize(high_cutoff_g)
        #low_cutoff_g =  gauss_peak_norm_functor(0,self.moire_r*self.lower_sigma_factor_cutoff,self.lower_norm_cutoff*E)
        #lhg = np.vectorize(low_cutoff_g)
        high_f = lambda x: self.calculate_upper_cutoff(E,x)
        low_f = lambda x: self.calculate_lower_cutoff(E,x)
        lv = np.vectorize(low_f)
        hv = np.vectorize(high_f)
        x = np.linspace(-20,20,1000)
        ax.plot(x,lv(x),label='low cutoff')
        ax.plot(x,hv(x),label='high cutoff')
        #ax.set_ylim(ymin=0,ymax=0.01)
        #ax.plot(x,[self.expand_cutoff]*len(x),label='absolute low')
        #ax.plot(x,[E]*len(x),label='absolute high') 
        ax.minorticks_on()
        ax.grid(True,which='both')
        ax.set_xlabel('distance (#of crystal)')
        ax.set_ylabel('E(GeV)')
        ax.legend()
        return ax       
        
        

#assume seed list is sorted
#these operations return array the same size as a
#note that user is responsible for selecting only the region that makes sense
#usually it's a[1:-1,1:-1]
#or your could filter out the deges by 
# a[0:,:]=0;a[:-1,:]=0;a[:,0:]=0;a[:,:-1]=0 etc.
#0,0 is defined as top left first index is row and second index is column
#what this does is the following
#  |ul | u | ur |
#  | l | c | r |
#  | bl| b | br|
# forevery element center at c new value = c_arg*c + ul*ul_arg .... br*br_arg
def nine_op(a,ul=0,u=0,ur=0,l=0,c=0,r=0,bl=0,b=0,br=0):

   ret = np.copy(a)
   ret*=c

   ret[1:,1:] += ul*a[:-1,:-1]
   ret[1:,:] += u*a[:-1,:]
   ret[1:,:-1] += ur*a[:-1,1:]

   ret[:,1:] += l*a[:,:-1]
   ret[:,:-1] += r*a[:,1:]

   ret[:-1,1:] += bl*a[1:,:-1]
   ret[:-1,:] += b*a[1:,:]
   ret[:-1,:-1] += br*a[1:,1:]
   return ret

def nine_opa(a,c):
   return nine_op(a,c[0,0],c[0,1],c[0,2],c[1,0],c[1,1],c[1,2],c[2,0],c[2,1],c[2,2])

#perform cross operation this is just to save some time
def cross_op(a,u=0,l=0,c=0,r=0,b=0):
   ret = np.copy(a)
   ret*=c
   ret[1:,:] += u*a[:-1,:]
   ret[:,1:] += l*a[:,:-1]
   ret[:,:-1] += r*a[:,1:]
   ret[:-1,:] += b*a[1:,:]
   return ret

#compute laplacian; poor man edge detection
def laplacian(a):
   return cross_op(a,c=4,r=-1,l=-1,u=-1,b=-1)

#return magnitude of gradient
def grad2(a):
    g_x = cross_op(a,l=1,r=-1)
    g_y = cross_op(a,u=1,b=-1)
    return np.sqrt(g_x*g_x+g_y*g_y)

#3x3 normalized gaussian blur
def gaussian_blur(a,sigma=1):
   x = np.array([[-1,0,1],[-1,0,1],[-1,0,1]])
   y = np.array([[-1,-1,-1],[0,0,0],[1,1,1]])
   unnorm_gau = np.exp(-(x**x+y**y)/(2*sigma));
   norm_gau = unnorm_gau/np.sum(unnorm_gau)
   return nine_opa(a,norm_gau)

def test_op():
   a = np.zeros((5,5))
   a[:,:]=0.5
   print a
   print laplacian(a)
   print gaussian_blur(a)
#test_op()

class Visualizer:
    @classmethod
    def show_hits(self,hits, ax=None,cutoff=0.0005,vmin=None,vmax=None):
        if ax is None: ax = gca()
        toshow = np.copy(hits)
        if(cutoff is not None):
            toshow[toshow<cutoff] = None
        img = ax.imshow(toshow,interpolation='nearest',origin='lower',vmin = vmin,vmax = vmax)
        ax.grid(True,which='both')
        return ax,img
    
    @classmethod
    def show_cluster(self,clusters,ax=None,hits=None,xlim=(0,44),ylim=(0,12)):
        if ax is None: ax = gca()
        p=[]
        
        for cluster in clusters:
            
            cx,cy = zip(*cluster.cluster)
            #print cx,cy
            #yep y then x it's the way imshow works
            q = ax.plot(cy,cx,'s',ms=12.5,alpha=0.7)
            p.append(q)
            if hits is not None:
                Ecl = HitMap.sumE_from_hits(hits,cluster.cluster)
                #print Ecl,cluster
                ax.annotate('%5.4f'%Ecl,xy=(cy[0],cx[0]))
        ax.grid(True)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        return ax,p
    
    @classmethod
    def show_seeds(self,seeds,hits, ax=None,cutoff=None,xlim=(0,44),ylim=(0,12)):
        if ax is None: ax = gca()
        p=[]
        for seed in seeds:
            #print seed
            q = ax.plot(seed[1],seed[0],'x')
            p.append(q)
        ax.grid(True,which='both')
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        return ax,p
