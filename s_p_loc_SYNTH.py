import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as col
from scipy.interpolate import interp1d


#---------------------------------------
# function to compute misfit (cost function)
#---------------------------------------
def cost_f(x0,y0,xsta,ysta, s_p_obs, sdiff, sigma):
    dist = np.sqrt((xsta-x0)**2 + (ysta-y0)**2)
    s_p = dist*sdiff
    ns = s_p_obs.size
    cf1 = np.sum(((s_p-s_p_obs)**2))/(ns*sigma**2)
    cf2 = np.exp(-1/2.*np.sum(((s_p-s_p_obs)**2))/(ns*sigma**2))
    cf3 = np.sum(np.exp(-1/2.*((s_p-s_p_obs)**2)/(sigma**2)))/ns
    return (cf1,cf2,cf3)
#---------------------------------------


#-----------------------------------------------
# function to remove outlier number i from array X
#-----------------------------------------------
def remove_outlier(X, i):
    n = np.size(X)
    num = np.arange(n)
    numi = num != i
    out = X[numi]
    return out


#---------------------------------------
# defining parameters and computing synthetic data
#---------------------------------------

#------------------ travel time "model" used for the prediction of synthetic S-P time differences
sdiff0 = 10     # difference in slownesses between S and P waves
sigmat0 = 2     # standars deviation for travel time random errors

#------------------ travel time "model" used for inversion
sdiff = 10.
sigmat = 2.

#----------- number of stations
NSTA = 10


xpmin = -10     # area for the grid search
xpmax = 10
ypmin = -10
ypmax = 10


xmax = 10       # area where stations are distributed
xmin = 0
ymax = 10
ymin = 0

dx = .1         # steps for the grid search
dy = .1

x0 = 5.         # true epicenter position 
y0 = 5.


#---------------------------------------
# computing station positions and trave times
#---------------------------------------

xsta = xmin + (xmax-xmin)*np.random.rand(NSTA)
ysta = ymin + (ymax-ymin)*np.random.rand(NSTA) 

dt = sigmat0*np.random.randn(NSTA)

dist = np.sqrt((xsta-x0)**2 + (ysta-y0)**2)

s_p_obs = dist*sdiff0 + dt


#------------------- manipulating data 

#s_p_obs[4] += 20                    # introducting "strong" time errors

#-----------------------------------------------
# removing outlier number i
#-----------------------------------------------
#i = 4
#s_p_obs = remove_outlier(s_p_obs, i)
#xsta = remove_outlier(xsta, i)
#ysta = remove_outlier(ysta, i)



#---------------------------------------
# inverting "data"
#---------------------------------------

nx = int((xpmax-xpmin)/dx) + 1      # defining grid for search
ny = int((ypmax-ypmin)/dy) + 1

xpmax = xpmin + (nx-1)*dx
ypmax = ypmin + (ny-1)*dy

mf1 = np.zeros((nx, ny))
mf2 = np.zeros((nx, ny))
mf3 = np.zeros((nx, ny))
xm = np.zeros((nx, ny))
ym = np.zeros((nx, ny))


for ix in range(0, nx):                     # grid search
    for iy in range(0, ny):
        xm[ix,iy] = xpmin + ix*dx
        ym[ix,iy] = ypmin + iy*dy
        (mf1[ix,iy],mf2[ix,iy],mf3[ix,iy]) = cost_f(xm[ix,iy],ym[ix,iy],xsta,ysta, s_p_obs, sdiff, sigmat)
        
        
mf = mf2                                    # selecting final type of the cost function


epicenter = np.where(mf == mf.max())        # finding "best-fit" epicenter
xE = xm[epicenter]
yE = ym[epicenter]
mfE = mf[epicenter]


#---------------------------------------
# plotting results
#---------------------------------------
fig1 = plt.figure(1, figsize=(8,8))
ax = fig1.add_axes([0.1,0.1,0.8,0.8])

my_cmap = plt.cm.gist_stern_r
cs = plt.pcolor(xm, ym, mf, cmap=my_cmap)
cbar = plt.colorbar(orientation='horizontal', pad=0.05)
cbar.set_label('cost function')

plt.plot(xsta,ysta,'.')
plt.plot(x0,y0,'y*')

plt.show()


#-----------------------------------------------
# plotting residuals at the best fit position
#-----------------------------------------------
d = np.sqrt((xsta-xE)**2 + (ysta-yE)**2)
t_s_p = d*sdiff

resid = t_s_p - s_p_obs
mean_resid = np.mean(resid)
std_resid = np.std(resid)

midline = resid-resid + mean_resid
maxline1 = midline + std_resid
maxline2 = midline + 2*std_resid
minline1 = midline - std_resid
minline2 = midline - 2*std_resid

#---------------------------------------
fig2 = plt.figure(2, figsize=(8,8))
ax = fig2.add_axes([0.1,0.1,0.8,0.6])

plt.plot(resid,'ok', mfc='none')
plt.plot(midline,'g')
plt.plot(maxline1,'g--')
plt.plot(maxline2,'r--')
plt.plot(minline1,'g--')
plt.plot(minline2,'r--')
plt.xlim(-1,np.size(resid))
plt.xlabel('station N')
plt.ylabel('residual (s)')
plt.title('Summary of misfit and residuals at the best-fit location\n\
best-fit X %.2f\n\
best-fit Y %.2f\n\
cost function %.7f\n\
mean time residual (s) %.2f\n\
std of the time residuals (s) %.2f' % (xE, yE, mfE, mean_resid, std_resid) )

plt.show()


#-----------------------------------------------
# analyzing residuals for outliers
#-----------------------------------------------
nr = np.size(resid)

out = np.zeros(nr)

num = np.arange(nr)

for i in range(0, nr):
    numi = num != i
    m = np.mean(resid[numi])
    s = np.std(resid[numi])
    out[i] = np.fabs(resid[i]-m)/s

maxline = resid-resid + 3.
ymax = np.max((np.max(out),3)) +.25

#-----------------------------------------------
# plotting residuals norimalized with starndard deviations (of the  dataset not including them)
fig3 = plt.figure(3, figsize=(8,8))
ax = fig3.add_axes([0.1,0.1,0.8,0.8])

plt.plot(out,'ok', mfc='none')
plt.plot(maxline,'r--')
plt.xlim(-1,nr)
plt.ylim(0,ymax)
plt.xlabel('station N')
plt.ylabel('residual in standard deviations')

plt.show()


#-----------------------------------------------
# histogram of residuals
fig4 = plt.figure(4)
plt.hist(resid)
plt.ylabel('histogram')
plt.xlabel('residual(s)')

plt.show()


