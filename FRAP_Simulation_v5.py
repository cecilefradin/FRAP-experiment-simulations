# Simulation of FRAP experiments in a finite compartment
#
# v0: Written by Lili Zhang (2022)
# v1: Modified by Cécile Fradin to include a finite photobleaching step (April 2024)
# v2: Improved to include two molecular species (Cécile, May 2024)
# v3: Adapted to the geometry of a bacteria (Cécile, May 2024)
# v4: Added nucleocytoplasmic transport (Cécile, May 2024)
# v5: Used to perform FRAP simulations in Zhang et al.
# 

# In[1]: Initialization

### Import packages
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.patches import Rectangle
import time
from scipy.optimize import curve_fit
import random

### Start the clock
time0 = time.time()

### Physical constants
Avonano = 6.02e14 # particles in a nanomole (particles in a L if the concentration is 1 nM)
Avomicronano = Avonano * 1e-15 # particles in a micron^3 if the concentration is 1 nM


# In[2]: Define simulation parameters
     
### Variables related to the shape of the compartment, photobleaching area and simulation window
shape = 0 # Enter 0 for sphere (nucleus) and 1 for rod (E. coli)

# If sphere:
radius =  3 # um (radius of nucleus)   #radius = rf
if shape == 0: 
    boundary = 2 * 1.2 * radius     # width of simulation window
    V_in = 4/3*np.pi*radius**3     # Volume of compartment
    V_out = boundary**3 - V_in  # Volume of simulation window now in compartment
    # Photobleaching area
    xB_min = -radius
    xB_max = radius
    yB_min = 0
    yB_max = radius
    f_PB = 1/2           # Fraction of compartment volume that's photobleached

# If rod:
radiusR = 0.5 # um (radius of rod)
lengthR = 3 # um (length of rod)    
if shape ==1:  # Photobleaching area
    boundary = 1.2*(lengthR + 2* radiusR)    # width of simulation window
    V_in = 4/3*np.pi*radiusR**3 + lengthR*np.pi*radiusR**2    # Volume of compartment
    V_out = boundary**3 - V_in  # Volume of simulation window now in compartment
    L_PB = radiusR           #  Distance passed the cap in the photobleaching area (should be between 0 and lengthR)
    xB_max = lengthR/2 + radiusR
    xB_min = lengthR/2 - L_PB
    yB_min = -radiusR
    yB_max = radiusR
    f_PB = (2*radiusR + 3*L_PB)/(4*radiusR + 3*lengthR)    # Fraction of compartment volume that's photobleached

 
### Variables related to the size of the simulation window
pixel_size = 0.1 # um (pixel size in micron)
image_size = int(boundary / pixel_size) # px (width of images in pixel)
w_rad = image_size * pixel_size / 2 # um (radius of the simulation window)
r_c = [w_rad, w_rad, w_rad] # (coordinates of the simulation window center)


### Variable related to the mobility and brightness of the particles
diff_const = 0.5 # um^2/s (diffusion coefficient of slow particles)   #diff_const = Df # um^2/s 
diff_const2 = 15 # um^2/s (diffusion coefficient of fast particles)
step_time = 0.001 # s  (step duration)
B = 1e4 # Hz  (molecular brightness)  


### Number of particles
Nparticles = 2500 + 1250   # Initial number of particles (before photobleaching step)
f2 = 0.6666      # Fraction of fast particles     #f2 = f2f
N2 = int(f2 * Nparticles)   # Total number of fast particles inside the compartment
N1 = Nparticles - N2      # Total number of slow particles inside the compartment
print('Simulation run for ' + str(Nparticles) + ' particles in total inside the compartment')
print(str(N1) + ' slow particles and ' + str(N2) + ' fast particles')
Cin1 = N1 / Avomicronano / V_in 
Cin2 = N2 / Avomicronano / V_in 
print('The concentration of slow particles inside the compartment is: ' + str(Cin1) + ' (nM)')
print('The concentration of fast particles inside the compartment is: ' + str(Cin2) + ' (nM)')

### Variables related to nucleo-cytoplasmic transport
NCT = True     # Set to True if the particles in the cytoplasm should be simulated
N_C_ratio_true = 15 # (nuclear to cytoplasmic ratio)
pin = 0.02   #  sets the probability for nuclear import   #pin = pinf
if NCT:
    Cout2 = (Cin1+Cin2) / N_C_ratio_true
    N_C_ratio = Cin2/Cout2
    pout = pin/N_C_ratio # (probability of nuclear export)
else:
    Cout2 = 0
    pout = 0
print('The concentration of fast particles outside the compartment is: '  + str(Cout2) + ' (nM)')
Nout2 = int(Cout2 * V_out * Avomicronano)
N2T = N2 + Nout2
NparticlesTot = Nparticles + Nout2 
print('The number of fast particles outside the compartment is: ' + str(Nout2))

### Particle flux outside the simulation box (has been validated analytically and "experimentally")
FluxBox = Cout2 * Avomicronano * 6 * boundary**2 * np.sqrt(diff_const2 * step_time / np.pi)  # Average number of particles coming out of the box at each step

print('The total number of simulated particles is: ' + str(NparticlesTot))
      
### Variables related to photobleaching
t_B = 0.6       # Time spent photobleaching         #t_B = t_Bf
#p_lifetime = 0.2    # s (lifetime of a fluorophore at the photobleaching intensity)  #p_lifetime = pf
p_lifetime = t_B/3
tinterval_B = max(min(0.1, t_B/5),step_time)   # Time interval between frames during photobleaching
npoints_B = round(t_B/tinterval_B)      # Number of frames pre-bleach
ns_B = round(tinterval_B/step_time)     # number of steps between two frames


### Variables related to the pre-bleach step
t_preB = 2     # Time spent pre-bleach
tinterval_preB = 0.1      # Time interval between frames pre-bleach
npoints_preB = round(t_preB/tinterval_preB)      # Number of frames pre-bleach
ns_preB = round(tinterval_preB/step_time) # number of steps between two frames


### Variables related to the post-bleach step
t_postB = 20   #20
#if shape == 0:
#    t_postB = min(t_postB,radius**2/diff_const)     # Time spent post-bleach
tinterval_postB = 0.1      # Time interval between frames post-bleach
npoints_postB = round(t_postB/tinterval_postB)      # Number of frames post-bleach
ns = round(tinterval_postB/step_time) # number of steps between two frames


### Variables related to image production
dwell_time = 0.001 # s  (pixel dwell time)
psf_width = 0.3 # um (width of the point spread function)
psf_height = 1.5 # um (height of point spread function)


### Variable related to plots
bin_size = 0.1
plt.rcParams['figure.dpi'] = 300  # Adjust the DPI to desired value


# In[3]: Useful functions

def GaussianBeam( start_pos, beam_pos, psf_width, psf_height):
    if start_pos.shape[0] == 2:
        GB = B*step_time*np.exp(- 2* ((start_pos - beam_pos)**2).sum()/ psf_width**2)
    else:
        GB = B*step_time*np.exp(- 2* ((start_pos[0:2] - beam_pos[0:2])**2).sum()/ psf_width**2) * np.exp(-2*((start_pos[2]-beam_pos[2])**2/psf_height**2))
        
    return GB

# In[4]: Useful procedures

### Count number of particles in the bleached and unbleached regions of the compartment
def Signal(r_pos):
    Ib = 0
    Iu = 0
    Ic = 0
    for n in range(len(r_pos)):
        d_pos = r_pos[n] - r_c
        if InCompartment(r_pos[n],shape):
            if (d_pos[0] >= xB_min and d_pos[0] <= xB_max and d_pos[1] >= yB_min and d_pos[1] <= yB_max ):
                Ib += 1
            else:
                Iu += 1
        else:
            Ic += 1
    return Ib, Iu, Ic

### Test if a particle is inside the compartment
def InCompartment(n_pos,shapeV):
    if shapeV == 0:
        return InSphere(n_pos,r_c,radius)
    else:
        return InRod(n_pos,r_c,lengthR,radiusR)

### Test if a certain point is inside a sphere
def InSphere(n_pos,r_center,rad): 
    radial_pos = np.sqrt(((n_pos - r_center)**2).sum())
    if radial_pos <= rad:
        return True
    else:
        return False
    
### Test if a certain point is inside a rod aligned with the x-axis
def InRod(n_pos,r_center,length_rod,radius_rod): 
    radial_pos_yz = np.sqrt(((n_pos[1:] - r_center[1:])**2).sum())
    r_m = r_center - np.array([length_rod/2,0,0])
    r_M = r_center + np.array([length_rod/2,0,0])
    if ((n_pos[0] >= r_m[0] and n_pos[0] <= r_M[0] and radial_pos_yz <= radius_rod) or (InSphere(n_pos,r_m,radius_rod)) or (InSphere(n_pos,r_M,radius_rod))):
# or (InSphere(n_pos,r_m,radius_rd)) or (InSphere(n_pos,r_M,radius_rd))
        return True
    else:
        return False
    

### Update particle position for particles confined to the compartment
def UpdatePositions(r_pos, deltaT, d_c, shapeV):
    displacement = np.random.normal(loc=0,scale=np.sqrt(2*d_c*deltaT),size=(len(r_pos),3))
    for n in range(len(r_pos)):            # Loop on all particles whose positions are in array
        new_position = r_pos[n,:] + displacement[n,:]
        if InCompartment(new_position,shape):  # Accept all moves that don't cross membrane compartment
                r_pos[n,:] = new_position
    return r_pos

### Update particle position for particles that might be inside or outside of the compartment
def UpdatePositionsOut(r_pos, deltaT, d_c, shapeV, pinV, poutV, Flux):

    displacement = np.random.normal(loc=0,scale=np.sqrt(2*d_c*deltaT),size=(len(r_pos),3))
    counter = 0   ### This counter is not currently needed
    n=0

    while n < len(r_pos):          # Loop on all particles whose positions are in array
        new_position = r_pos[n,:] + displacement[n,:]

        # Case of particles that are initially inside the compartment:
        if InCompartment(r_pos[n,:],shape):     
            if InCompartment(new_position,shape):   # Accept all moves that don't cross membrane compartment
                r_pos[n,:] = new_position
            else:
                if np.random.rand() < poutV:  # Accept moves that cross membrane compartment with proba pin
                    r_pos[n,:] = new_position

        # Case of particles that are initially outside the compartment:
        else:
            if not InCompartment(new_position,shape):   # Accept all moves that don't cross membrane compartment
                r_pos[n,:] = new_position
            else:
                if np.random.rand() < pinV:  # Accept moves that cross membrane compartment with proba pin
                    r_pos[n,:] = new_position  
                    
        # Remove particles that leave the simulation window
        if max(r_pos[n,:]) > 2*w_rad or min(r_pos[n,:]) < 0:
            r_pos = np.delete(r_pos, n, axis = 0) 
        else:
            n += 1
#        n+=1
    # Inject back the right number of particles back in
    r_pos = InjectParticles(r_pos,deltaT,d_c,Flux)
    return r_pos, counter

### Inject particles at the border of the simulation window
def InjectParticles(r_pos,deltaT,d_c,Flux):     # Maybe the flux should be calculated here as it depends on deltaT
    Np = np.random.poisson(Flux)
    n = 0
    while n < Np:
        k = random.randint(0,5)
        x = np.random.rand() * boundary
        y = np.random.rand() * boundary
        z = abs(np.random.normal(loc=0,scale=np.sqrt(2*d_c*deltaT),size=1))[0]
        if k == 0:
            new_pos = np.array([[x,y,z]])
        elif k == 1:
            new_pos = np.array([[x,y,boundary-z]])
        elif k == 2:
            new_pos = np.array([[x,z,y]])
        elif k == 3:
            new_pos = np.array([[x,boundary-z,y]])
        elif k == 4:
            new_pos = np.array([[z,x,y]])
        else:
            new_pos = np.array([[boundary-z,x,y]])   
        if not InCompartment(new_pos,shape):
            r_pos = np.append(r_pos,new_pos, axis=0)
            n += 1
    return r_pos

### Perform photobleaching
def UpdateFluorescence(r_pos,deltaT):
    Pb = deltaT / p_lifetime
    n = 0
    while n < len(r_pos):
        d_pos = r_pos[n] - r_c
        if (d_pos[1] >= yB_min) and (d_pos[1] < yB_max) and (d_pos[0] < xB_max) and (d_pos[0] > xB_min):
            proba = np.random.rand()
            if proba < Pb: # photobleach the fluorophore with a probability depending on the exposure time
                r_pos = np.delete(r_pos, n, axis = 0)
            else:
                n += 1
        else:
            n += 1
    return r_pos

### Visualize the position of the particles
def P_plot(r_pos1,r_pos2,k):
    r_pos1 = r_pos1.T
    r_pos2 = r_pos2.T
    x_coords = [r_pos1[0] for point in r_pos1]
    y_coords = [r_pos1[1] for point in r_pos1]
    x_coordf = [r_pos2[0] for point in r_pos2]
    y_coordf = [r_pos2[1] for point in r_pos2]
    plt.figure(figsize=(6, 6))
    pads = 0.2
    plt.xlim(-pads,boundary+pads)
    plt.ylim(-pads,boundary+pads)
    if k == 1:
        # Highlight photobleaching area
        photoB_patch = Rectangle((r_c[0]+xB_min, r_c[1]+yB_min), xB_max-xB_min, yB_max-yB_min, color='yellow', alpha=0.3)  # Semi-transparent rectangle
        plt.gca().add_patch(photoB_patch)
    ms1 = 2
    ms2 = 2
    plt.scatter(x_coords, y_coords, c='blue', marker='o', s = ms1)
    plt.scatter(x_coordf, y_coordf, c='green', marker='o', s = ms2)
    plt.xlabel('X ($\mu$m)')
    plt.ylabel('Y ($\mu$m)')
#    plt.title('Projection of particle positions onto the (x, y) plane')
    plt.show()
    
### Generate and return an intensity profile
def Prof_plot(r_pos,bsize):
    x_pos = r_pos[:,0]
    bins = np.arange(0,boundary+bsize,bsize)
    hist, bin_edges = np.histogram(x_pos, bins=bins)
    return hist
    

# In[5]: Generate initial particle positions


### Generate initial particle positions for slow particles
start_pos_1 = np.zeros((N1,3))
for n in range(N1):  
#for n in range(Nnuclear):   
    r = start_pos_1[n,:]
    if shape == 0:
        while not InSphere(r,r_c,radius):
            r = np.random.rand(3) * w_rad * 2
    if shape == 1:
        while not InRod(r,r_c,lengthR,radiusR):
            r = np.random.rand(3) * w_rad * 2            
    start_pos_1[n,:] = r  
    

### Generate initial particle positions for fast particles

start_pos_2 = np.zeros((N2T,3))

for n in range(N2):            # Particles in the compartment
    r = start_pos_2[n,:]
    while not InCompartment(r,shape):
        r = np.random.rand(3) * w_rad * 2
    start_pos_2[n,:] = r  

for n in range(N2,N2T):          # Particles outside of compartment
    r = start_pos_2[n,:]+r_c
    while InCompartment(r,shape):
        r = np.random.rand(3) * w_rad * 2
    start_pos_2[n,:] = r                       
                                  

### Assign a specific brightness to each particle (in monomer units)
# start_B = np.ones((Nparticles,1))

### Initialize timer and intensities
timer = 0
TimelineFRAP = np.array([timer])
Ib1, Iu1, Ic1 = Signal(start_pos_1)
Ib2, Iu2, Ic2 = Signal(start_pos_2)
Intensityb = np.array([Ib1+Ib2])
Intensityu = np.array([Iu1+Iu2])
Intensityc = np.array([Ic2])

P_plot(start_pos_1,start_pos_2,0)

bins = np.arange(0,boundary+bin_size,bin_size)
bins_pos = bins[:-1] + bin_size/2
hist1 = Prof_plot(start_pos_1,bin_size)
hist2 = Prof_plot(start_pos_2,bin_size)
hist = hist1 + hist2
hist = np.stack((bins_pos,hist)).T

time1 = time.time()
print('Initialization took ' + str(round((time1-time0),2)) + ' s')


# In[7]: Run the system pre-bleach

for n in range(npoints_preB):
    for i in range(ns_preB):
        start_pos_1 = UpdatePositions(start_pos_1,step_time,diff_const,shape)
        start_pos_2, counter = UpdatePositionsOut(start_pos_2,step_time,diff_const2,shape,pin,pout,FluxBox)
    timer += tinterval_preB
    TimelineFRAP = np.append(TimelineFRAP,timer)
    Ib1, Iu1, Ic1 = Signal(start_pos_1)
    Ib2, Iu2, Ic2 = Signal(start_pos_2)
    Intensityb = np.append(Intensityb,Ib1+Ib2)
    Intensityu = np.append(Intensityu,Iu1+Iu2)
    Intensityc = np.append(Intensityc,Ic2)    
    temp_hist = Prof_plot(start_pos_1,bin_size) + Prof_plot(start_pos_2,bin_size)
    temp_hist = temp_hist.reshape(len(temp_hist),1)
    hist = np.hstack((hist,temp_hist))
    
P_plot(start_pos_1,start_pos_2,0)
    
time2 = time.time()
print('Pre-bleach diffusion took ' + str(round((time2 - time1),2)) + ' s')


# In[4]: Photobleaching step


for n in range(npoints_B):
    for i in range(ns_B):
        start_pos_1 = UpdatePositions(start_pos_1,step_time,diff_const,shape)
        start_pos_2, counter = UpdatePositionsOut(start_pos_2,step_time,diff_const2,shape,pin,pout,FluxBox)
        start_pos_1 = UpdateFluorescence(start_pos_1,step_time)
        start_pos_2 = UpdateFluorescence(start_pos_2,step_time)
    timer += tinterval_B
    TimelineFRAP = np.append(TimelineFRAP,timer)
    Ib1, Iu1, Ic1 = Signal(start_pos_1)
    Ib2, Iu2, Ic2 = Signal(start_pos_2)
    Intensityb = np.append(Intensityb,Ib1+Ib2)
    Intensityu = np.append(Intensityu,Iu1+Iu2)
    Intensityc = np.append(Intensityc,Ic2)
    temp_hist = Prof_plot(start_pos_1,bin_size) + Prof_plot(start_pos_2,bin_size)
    temp_hist = temp_hist.reshape(len(temp_hist),1)
    hist = np.hstack((hist,temp_hist))
    
### Reset the timer to 0    
TimelineFRAP = TimelineFRAP - timer
timer = 0

### Calculate unbleached fraction
if N1>0:
    ufrac1 = (Iu1 + Ib1)/N1
else:
    ufrac1 = 0
if N2>0:
    ufrac2 = (Iu2 + Ib2)/N2
else:
    ufrac2 = 0
ufrac = (Iu1 + Iu2 + Ib1 + Ib2)/Nparticles
print('The fraction of slow molecules that are still fluorescent is: ' + str(round(ufrac1,2)))
print('The fraction of fast molecules that are still fluorescent is: ' + str(round(ufrac2,2)))

P_plot(start_pos_1,start_pos_2,1)


time3 = time.time()
print('Photobleaching took ' + str(round((time3 - time2),2)) + ' s')


# In[7]: Run the system post-bleach

for n in range(npoints_postB):
    for i in range(ns):
        start_pos_1 = UpdatePositions(start_pos_1,step_time,diff_const,shape)
        start_pos_2, counter = UpdatePositionsOut(start_pos_2,step_time,diff_const2,shape,pin,pout,FluxBox)       
    timer += tinterval_postB
    TimelineFRAP = np.append(TimelineFRAP,timer)
    Ib1, Iu1, Ic1 = Signal(start_pos_1)
    Ib2, Iu2, Ic2 = Signal(start_pos_2)
    Intensityb = np.append(Intensityb,Ib1+Ib2)
    Intensityu = np.append(Intensityu,Iu1+Iu2)
    Intensityc = np.append(Intensityc,Ic2)
    temp_hist = Prof_plot(start_pos_1,bin_size) + Prof_plot(start_pos_2,bin_size)
    temp_hist = temp_hist.reshape(len(temp_hist),1)
    hist = np.hstack((hist,temp_hist))

    
P_plot(start_pos_1,start_pos_2,0)
    
time4 = time.time()
print('Post-bleach diffusion took ' + str(round((time4 - time3),2)) + ' s')

# In[10]: Plot the intensity profiles

if shape == 1:

    plt.figure(figsize=(10, 6))

    histT = hist.T
    bins_pos = histT[0]

    max_int = Nparticles / (4/3*np.pi*radiusR**3 + lengthR*radiusR**2*np.pi) * (np.pi * radiusR**2 * bin_size)

    hist_temp = np.zeros(len(histT[0]))
    for i in range(1,npoints_preB+1):
        hist_temp = hist_temp + histT[i]
        hist_preB = hist_temp / npoints_preB / max_int
        plt.plot(bins_pos, hist_preB, c='black', marker='o')

    s_factor = 5 # average every 10 curves
    num_curves = min(20,math.floor(npoints_postB / 10))  # number of profiles to return
    colormap = cm.viridis  # You can choose different colormaps like 'plasma', 'inferno', 'magma', etc.
    colors = colormap(np.linspace(0, 1, num_curves))

    for j in range(0,num_curves):
        hist_temp = np.zeros(len(histT[0]))
        for i in range(0,s_factor):
            hist_temp = hist_temp + histT[npoints_preB+npoints_B+1+j*s_factor+i]
            hist_postB = hist_temp / s_factor / max_int
            plt.plot(bins_pos, hist_postB, c=colors[j], marker='s', linestyle='--')

    photoB = Rectangle((r_c[0]+xB_min, 0), xB_max - xB_min, 1.2, color='yellow', alpha=0.5)  # Semi-transparent rectangle
    plt.gca().add_patch(photoB)

    plt.xlabel('X ($\mu$m)')
    plt.ylabel('Number of Particles')
    plt.title('Intensity profiles')

    plt.show()
    
# In[20]: Fit the FRAP recovery curves

### Fit the recovery curves
def funcb(t, Ieq, tauf, dI):
    return Ieq - dI * 2 * (1-f_PB) * np.exp(-t / tauf)

def funcu(t, Ieq, tauf, dI):
    return Ieq + dI * 2 * f_PB * np.exp(-t / tauf)


### Extract the curves post-bleach

startP = int(npoints_preB+npoints_B)
TimeLine_postB = TimelineFRAP[startP:len(TimelineFRAP)]
IbpostB = Intensityb[startP:len(TimelineFRAP)]/(Nparticles*f_PB)
IupostB = Intensityu[startP:len(TimelineFRAP)]/(Nparticles*(1-f_PB))

xdata = np.concatenate([TimeLine_postB,TimeLine_postB])
ydata = np.concatenate([IbpostB,IupostB])

Iu0 = IupostB[0]
Ib0 = IbpostB[0]
IuL = IupostB[-1]
IbL = IbpostB[-1]

initial_guesses = [(Iu0+Ib0)/2,t_B,0.3]
try:
    param_b, cov_b = curve_fit(funcb, TimeLine_postB, IbpostB, p0=initial_guesses)
    param_u, cov_u = curve_fit(funcu, TimeLine_postB, IupostB, p0=initial_guesses)
    print("Fit result for bleached region: ", param_b)
    print("Fit result for unbleached region: ", param_u)
except RuntimeError as e:
    print("Fit did not converge:", e)
    param_b = initial_guesses
    param_u = initial_guesses
    
# def funcglobal(t, fp, tauf, dI, c, fi, empty):       # Linearized version 
#     midpoint = len(t) // 2
#     tb = t[:midpoint]
#     tu = t[midpoint:]
#     yb = fp - dI * (2 * (1-f_PB)) * np.exp(-tb / tauf) + (1 - fp) * tb / c - fi/2
#     yu = fp + dI * (2 * f_PB) * np.exp(-tu / tauf) + (1 - fp) * tu / c + fi/2
#     return np.concatenate([yb, yu])


def funcglobal(t, feq, tauf, dfs, taub, fi, g):       # Full version
     midpoint = len(t) // 2
     tb = t[:midpoint]
     tu = t[midpoint:]
     yb = feq + abs(g) - fi/2 - abs(dfs) * (2 * (1-f_PB)) * np.exp(-tb / tauf)  - abs(g) * np.exp(- tb / taub) 
     yu = feq + abs(g) + fi/2 + abs(dfs) * (2 * f_PB) * np.exp(-tu / tauf) - abs(g) * np.exp(- tu / taub) 
     return np.concatenate([yb, yu])    

initial_guesses_global = [(Iu0+Ib0)/2, 0.21*radius**2/diff_const, param_b[2], 3, 0, 1-(IuL+IbL)/2]
try:
    param_global, cov_global = curve_fit(funcglobal, xdata, ydata, p0=initial_guesses_global)
    print("Global fit result: ", param_global)
except RuntimeError as e:
    print("Fit did not converge:", e)
    param_global = initial_guesses_global


# In[20]: Plot the FRAP recovery curves

### Fix the axes size and position
Imax = 1.1
fig = plt.figure(figsize=(4.27,3.2))
ax = plt.gca()
pos1 = ax.get_position() # get the original position 
pos2 = [pos1.x0 + 0.02, pos1.y0 + 0.035,  pos1.width , pos1.height ] 
ax.set_position(pos2) # set a new position
plt.xlim(min(TimelineFRAP)-0.1,max(TimelineFRAP)+0.1)
plt.ylim(0,Imax)

### Plot the curves
ms = 10
lw = 1

npoints_preB+npoints_B

plt.scatter( TimelineFRAP[:npoints_preB], Intensityb[:npoints_preB]/(Nparticles*f_PB), marker = 'o', color = 'darkorange',  s = ms)
plt.scatter( TimelineFRAP[:npoints_preB], Intensityu[:npoints_preB]/(Nparticles*(1-f_PB)), marker = 'o', color = 'green',  s = ms)

plt.scatter( TimelineFRAP[npoints_preB:startP], Intensityb[npoints_preB:startP]/(Nparticles*f_PB), marker = 'o', edgecolors='darkorange', facecolors='none',  s = ms)
plt.scatter( TimelineFRAP[npoints_preB:startP], Intensityu[npoints_preB:startP]/(Nparticles*(1-f_PB)), marker = 'o', edgecolors='green', facecolors='none',  s = ms)

plt.scatter( TimelineFRAP[startP:], Intensityb[startP:]/(Nparticles*f_PB), marker = 'o', color = 'darkorange',  s = ms)
plt.scatter( TimelineFRAP[startP:], Intensityu[startP:]/(Nparticles*(1-f_PB)), marker = 'o', color = 'green',  s = ms)

if NCT:
    plt.scatter( TimelineFRAP[:npoints_preB], Intensityc[:npoints_preB]/Nout2/N_C_ratio_true, marker = 'o', color = 'darkblue',  s = ms)
    plt.scatter( TimelineFRAP[npoints_preB:startP], Intensityc[npoints_preB:startP]/Nout2/N_C_ratio_true, marker = 'o', edgecolors='darkblue', facecolors='none',  s = ms)
    plt.scatter( TimelineFRAP[startP:], Intensityc[startP:]/Nout2/N_C_ratio_true, marker = 'o', color = 'darkblue',  s = ms)

plt.plot( TimeLine_postB, funcglobal(xdata, *param_global)[:len(TimeLine_postB)], '-', color = 'indigo', linewidth = lw)
plt.plot( TimeLine_postB, funcglobal(xdata, *param_global)[len(TimeLine_postB):], '-', color = 'indigo', linewidth = lw)
plt.xlabel('Time (s)', fontsize=12)
plt.ylabel('Normalized intensity', fontsize=12)
#plt.legend(['$I_{b}$ ', ' $I_{u}$', 'fit'], loc='upper right', fontsize=12)

plt.text(13.9, 1, '$p_{in} =$'  + str(round(pin,3)), fontsize=12)

# Highlight photobleaching duration
photoB = Rectangle((-t_B, 0), t_B, Imax, color='yellow', alpha=0.5)  # Semi-transparent rectangle
plt.gca().add_patch(photoB)


fig.tight_layout()
#filename = 'Recovery.png'
#plt.savefig(filename, dpi=300)
plt.show() 

time5 = time.time()
print('Fitting and plotting took ' + str(round((time5 - time4),2)) + ' s')

# In[21]: Plot the residual curve

### Fix the axes size and position
DImax = 0.05
fig = plt.figure(figsize=(4.27,1.5))
ax = plt.gca()
pos1 = ax.get_position() # get the original position 
pos2 = [pos1.x0 + 0.02, pos1.y0 + 0.035,  pos1.width , pos1.height ] 
ax.set_position(pos2) # set a new position
plt.xlim(min(TimelineFRAP)-0.1,max(TimelineFRAP)+0.1)
plt.ylim(-DImax,DImax)

### Plot the curves
ms = 5
plt.scatter(TimeLine_postB, Intensityb[startP:]/(Nparticles*f_PB)-funcglobal(xdata, *param_global)[:len(TimeLine_postB)], marker = 'o', color = 'darkorange', s = ms)
linedata = np.zeros(len(TimeLine_postB))
plt.plot(TimeLine_postB, linedata, '-', color = 'indigo', linewidth = lw)
plt.xlabel('Time (s)', fontsize=12)
plt.ylabel('Residuals', fontsize=12)

fig.tight_layout()
#filename = 'Residuals.png'
#plt.savefig(filename, dpi=300)
plt.show() 


# In[100]: Conclusion

dose = t_B/p_lifetime
diff_const_ballpark = radius**2/param_global[1]
alpha = diff_const/diff_const_ballpark
f_p = param_global[0]
Delta_f = 2 * param_global[2] + param_global[4]
InputFRAP = np.array([Nparticles,radius,p_lifetime,step_time,t_preB,t_B,t_postB,tinterval_preB,tinterval_B,tinterval_postB,diff_const,diff_const2,pin,pout,dose,diff_const_ballpark,alpha,ufrac,f_p,Delta_f])
ResultFRAP = np.append(np.append(param_b,param_u),param_global)
OutputFRAP = np.append(InputFRAP,ResultFRAP)

time6 = time.time()
print('The total simulation time was ' + str(round((time6 - time0)/60,2)) + ' min')


  
    