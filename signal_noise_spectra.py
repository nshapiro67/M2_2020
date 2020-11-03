#------------------------ importing basic packages
import matplotlib.pyplot as plt
import numpy as np
#------------------------ importing ObsPy functions
from obspy import read
from obspy.clients.fdsn import Client
from obspy import UTCDateTime

#------------------ plotting mode
%matplotlib widget
#----------------------------


#------------------------ selecting an FDSN datacenter
client = Client('RESIF')


#-------------------- defining duration of the downloaded time series in sec
t_duration =10*60


#--------------- downloading noise window
tstart = UTCDateTime("2020-09-26T22:43:10.000")
st1 = client.get_waveforms("FR", "OGCN", "*", "HHZ", tstart, tstart + t_duration, attach_response=True)
noise = st1[0]
noise.detrend()

#--------------- downloading signal window
tstart = UTCDateTime("2020-09-26T22:53:10.000")
st1 = client.get_waveforms("FR", "OGCN", "*", "HHZ", tstart, tstart + t_duration, attach_response=True)
signal = st1[0]
signal.detrend()


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

#-----------------------------------------------
# function to smooth an array
#-----------------------------------------------
def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

nspe, n_spamp, sppha, fr = signal_fft1d(noise.data,noise.stats.delta)
nspe, s_spamp, sppha, fr = signal_fft1d(signal.data,signal.stats.delta)


plt.figure()
plt.loglog(fr,smooth(n_spamp,20),'b')
plt.loglog(fr,smooth(s_spamp,20),'r')
plt.xlim(.005,20)
plt.xlabel('frequency(Hz)')
plt.show()


