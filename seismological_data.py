#------------------------ importing basic packages
import matplotlib.pyplot as plt
import numpy as np
#------------------------ importing ObsPy functions
from obspy import read
from obspy.clients.fdsn import Client
from obspy import UTCDateTime

plt.close("all")

#------------------ plotting mode
#%matplotlib widget
#----------------------------


#------------------------ selecting an FDSN datacenter
client = Client('RESIF')

#---------------------------------------------------- defining start time for the data
tstart = UTCDateTime("2020-09-26T22:43:10.000")

#-------------------- defining duration of the downloaded time series in sec
t_duration = 30*60


#--------------- selecting network: FR
#--------------- selecting station: OGCN
#--------------- selecting component: HHZ
#--------------- downloading data
st1 = client.get_waveforms("FR", "OGCN", "*", "HHZ", tstart, tstart + t_duration, attach_response=True)



#-------------- extracting a trace from the stream
s1 = st1[0]

#------------ detrending time series
s1.detrend()

#----------- finding information in the header (trace.stats)

print("network: ", s1.stats.network)
print("station: ", s1.stats.station)
print("component: ", s1.stats.channel)
print("discretization time step: ", s1.stats.delta)
print("number of samples: ", s1.stats.npts)


#-------- plotting raw data
dt = s1.stats.delta
npts = s1.stats.npts
time = dt*(np.linspace(1,npts,npts)-1)

plt.figure()
plt.plot(time,s1.data)
plt.title(s1.stats.station)
plt.xlabel('time (s)')
plt.ylabel('counts')
plt.show()

#------------------------ writing data in local file
s1.write("seismogram.sac","SAC")


#------------------------ reading data from local file
stlocal = read("seismogram.sac")

st1[0].data = stlocal[0].data

s1 = st1[0]


#------------------ correcting for instrument response
pre_filt = (0.003, 0.005, 30.0, 35.0)                   # defining spectral band
st1.remove_response(output='DISP', pre_filt=pre_filt)

#-------- plotting corrected displacement seismogram
dt = s1.stats.delta
npts = s1.stats.npts
time = dt*(np.linspace(1,npts,npts)-1)

plt.figure()
plt.plot(time,s1.data)
plt.title(s1.stats.station)
plt.xlabel('time (s)')
plt.ylabel('ground displacement (m)')
plt.show()


#-------- plotting velocity seismogram
st1.differentiate()

dt = s1.stats.delta
npts = s1.stats.npts
time = dt*(np.linspace(1,npts,npts)-1)

plt.figure()
plt.plot(time,s1.data)
plt.title(s1.stats.station)
plt.xlabel('time (s)')
plt.ylabel('ground velocity (m/s)')
plt.show()


#-----------------------------------------------
# function to compute Fourier spectra
#-----------------------------------------------
def signal_fft1d(sig,dt):
    npt = np.size(sig)
    spe = np.fft.fft(sig)
    freq = np.fft.fftfreq(npt,dt)
    sp_amp = np.sqrt(spe.real**2+spe.imag**2)
    sp_pha = np.arctan2(spe.imag, spe.real)
    npt_spe = int(npt/2)
    return npt_spe, sp_amp[0:npt_spe],sp_pha[0:npt_spe],freq[0:npt_spe]

nspe, spamp, sppha, fr = signal_fft1d(s1.data,s1.stats.delta)

plt.figure()
plt.loglog(fr,spamp)
plt.xlim(.005,20)
plt.xlabel('frequency(Hz)')
plt.show()


#-------- filtering seismogram
st1.filter("bandpass", freqmin=0.02, freqmax=.1, corners=4, zerophase=True)

dt = s1.stats.delta
npts = s1.stats.npts
time = dt*(np.linspace(1,npts,npts)-1)

plt.figure()
plt.plot(time,s1.data)
plt.title(s1.stats.station)
plt.xlabel('time (s)')
plt.ylabel('filtered signal')
plt.show()