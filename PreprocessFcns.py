#Helper fcns for Data Preprocessing
import numpy as np
import pandas as pd
import pywt
import pathlib
import pickle #to save files
from itertools import product
from scipy.stats import skew, kurtosis, entropy
from scipy.signal import butter, welch, filtfilt, resample, decimate, resample_poly
from scipy.interpolate import interp1d
import math
import nolds
import time

#extract clips for accelerometer and gyro data (allows selecting start and end fraction)
#lentol is the % of the intended clipsize below which clip is not used
def gen_clips(act_dict,task,location,clipsize=5000,overlap=0,verbose=False,startTS=0,endTS=1,len_tol=0.8,resample=False):

    clip_data = {} #the dictionary with clips
    for trial in list(act_dict[task].keys()):
    
        for s in ['accel','gyro','elec']:

            if verbose:
                print(task,' sensortype = %s - trial %d'%(s,trial))
            #create clips and store in a list
            rawdata = act_dict[task][trial][location][s]
            if rawdata.empty is True: #skip if no data for current sensor
                continue
            #reindex time (relative to start)
            idx = rawdata.index
            idx = idx-idx[0]
            rawdata.index = idx
            #choose to create clips only on a fraction of the data (0<[startTS,endTS]<1)
            if (startTS > 0) | (endTS < 1):
                rawdata = rawdata.iloc[round(startTS*len(rawdata)):round(endTS*len(rawdata)),:]
                #reindex time (relative to start)
                idx = rawdata.index
                idx = idx-idx[0]
                rawdata.index = idx
            #create clips data
            deltat = np.median(np.diff(rawdata.index))
            clips = []
            #use entire recording
            if clipsize == 0:
                clips.append(rawdata)
            #take clips
            else:
                idx = np.arange(0,rawdata.index[-1],clipsize*(1-overlap))
                for i in idx:
                    c = rawdata[(rawdata.index>=i) & (rawdata.index<i+clipsize)]
                    if len(c) > len_tol*int(clipsize/deltat): #discard clips whose length is less than len_tol% of the window size
                        clips.append(c)

            #store clip length
            clip_len = [clips[c].index[-1]-clips[c].index[0] for c in range(len(clips))] #store the length of each clip
            #assemble in dict
            clip_data[trial][s] = {'data':clips, 'clip_len':clip_len}

    return clip_data


#store clips from all locations in a dataframe, indexed by the visit - NEED TO REFINE, do NOT USE
def gen_clips_alllocs(act_dict,task,clipsize=5000,overlap=0,verbose=False,len_tol=0.95):

    clip_data = pd.DataFrame()

    for visit in act_dict[task].keys():

        clip_data_visit=pd.DataFrame()

        #loop through locations
        for location in act_dict[task][visit].keys():

            for s in ['accel','gyro']:

                if verbose:
                    print(task,' sensortype = %s - visit %d'%(s,visit))
                #create clips and store in a list
                rawdata = act_dict[task][visit][location][s]
                if rawdata.empty is True: #skip if no data for current sensor
                    continue
                #reindex time (relative to start)
                idx = rawdata.index
                idx = idx-idx[0]
                rawdata.index = idx
                #create clips data
                deltat = np.median(np.diff(rawdata.index))
                clips = pd.DataFrame()
                #take clips
                idx = np.arange(0,rawdata.index[-1],clipsize*(1-overlap))
                for i in idx:
                    c = rawdata[(rawdata.index>=i) & (rawdata.index<i+clipsize)]
                    if len(c) > len_tol*int(clipsize/deltat): #discard clips whose length is less than len_tol% of the window size
                        df = pd.DataFrame({location+'_'+s:[c.values]},index=[visit])
                        clips=pd.concat((clips,df))

                clip_data_visit=pd.concat((clip_data_visit,clips),axis=1) #all clips from all locs for current visit

        clip_data = pd.concat((clip_data_visit,clip_data)) #contains clips from all visits (index) and sensors (cols)

    return clip_data



#PSD on magnitude using Welch method
def power_spectra_welch(rawdata,fm,fM):
    #compute PSD on signal magnitude
    x = rawdata.iloc[:,-1]
    n = len(x) #number of samples in clip
    Fs = np.mean(1/(np.diff(x.index)/1000)) #sampling rate in clip
    f,Pxx_den = welch(x,Fs,nperseg=min(256,n))
    #return PSD in desired interval of freq
    inds = (f<=fM)&(f>=fm)
    f=f[inds]
    Pxx_den=Pxx_den[inds]
    Pxxdf = pd.DataFrame(data=Pxx_den,index=f,columns=['PSD_magnitude'])

    return Pxxdf



def HPfilter(act_dict,task,loc,cutoff=0.75,ftype='highpass'):
#highpass (or lowpass) filter data. HP to remove gravity (offset - limb orientation) from accelerometer data from each visit (trial)
#input: Activity dictionary, cutoff freq [Hz], task, sensor location and type of filter (highpass or lowpass).

    sensor = 'accel'
    for trial in act_dict[task].keys():
        rawdata = act_dict[task][trial][loc][sensor]
        if rawdata.empty is True: #skip if no data for current sensor
            continue
        idx = rawdata.index
        idx = idx-idx[0]
        rawdata.index = idx
        x = rawdata.values
        Fs = np.mean(1/(np.diff(rawdata.index)/1000)) #sampling rate
        #filter design
        cutoff_norm = cutoff/(0.5*Fs)
        b,a = butter(4,cutoff_norm,btype=ftype,analog=False)
        #filter data
        xfilt = filtfilt(b,a,x,axis=0)
        rawdatafilt = pd.DataFrame(data=xfilt,index=rawdata.index,columns=rawdata.columns)
        act_dict[task][trial][loc][sensor] = rawdatafilt


#bandpass filter data (analysis of Tremor)
#input: Activity dictionary, min,max freq [Hz], task and sensor location to filter
def BPfilter(act_dict,task,loc,cutoff_low=3,cutoff_high=8,order=4):

    sensor = 'accel'
    for trial in act_dict[task].keys():
        rawdata = act_dict[task][trial][loc][sensor]
        idx = rawdata.index
        idx = idx-idx[0]
        rawdata.index = idx
        x = rawdata.values
        Fs = np.mean(1/(np.diff(rawdata.index)/1000)) #sampling rate
        #filter design
        cutoff_low_norm = cutoff_low/(0.5*Fs)
        cutoff_high_norm = cutoff_high/(0.5*Fs)
        b,a = butter(order,[cutoff_low_norm,cutoff_high_norm],btype='bandpass',analog=False)
        #filter data
        xfilt = filtfilt(b,a,x,axis=0)
        rawdatafilt = pd.DataFrame(data=xfilt,index=rawdata.index,columns=rawdata.columns)
        act_dict[task][trial][loc][sensor] = rawdatafilt

#filters data in all recordings (visits)
def filterdata(act_dict,task,loc,trial,sensor='accel',ftype='highpass',cutoff=0.5,cutoff_bp=[3,8],order=4):

    rawdata = act_dict[task][trial][loc][sensor]
    rawdata 
    if not rawdata.empty:
        idx = rawdata.index
        idx = idx-idx[0]
        rawdata.index = idx
        x = rawdata.values
        #print(np.unique(np.diff(rawdata.index)))
        Fs = np.mean(1/(np.diff(rawdata.index)/1000)) #sampling rate
        if ftype != 'bandpass':
            #filter design
            cutoff_norm = cutoff/(0.5*Fs)
            b,a = butter(4,cutoff_norm,btype=ftype,analog=False)
        else:
            #filter design
            cutoff_low_norm = cutoff_bp[0]/(0.5*Fs)
            cutoff_high_norm = cutoff_bp[1]/(0.5*Fs)
            b,a = butter(order,[cutoff_low_norm,cutoff_high_norm],btype='bandpass',analog=False)

        #filter data
        xfilt = filtfilt(b,a,x,axis=0)
        rawdatafilt = pd.DataFrame(data=xfilt,index=rawdata.index,columns=rawdata.columns)
        act_dict[task][trial][loc][sensor] = rawdatafilt



## CURRENTLY deprecated
#returns power spectra of the signal over each channel between min and max freq at given resolution (nbins)
#returns the labels for each bin
#if binavg is True it averages the PSD within bins to reduce PSD noise
def powerspectra(x,fm,fM,nbins=10,relative=False,binavg=True):

    #feature labels
    labels=[]
    s = np.linspace(fm,fM,nbins)
    lax = ['X','Y','Z']
    for l in lax:
        for i in s:
            labels.append('fft'+l+str(int(i)))

    #signal features
    n = len(x) #number of samples in clip
    Fs = np.mean(1/(np.diff(x.index)/1000)) #sampling rate in clip
    timestep = 1/Fs
    freq = np.fft.fftfreq(n,d=timestep) #frequency bins

    #run FFT on each channel
    Xf = x.apply(np.fft.fft)
    Xf.index = np.round(freq,decimals=1) #reindex w frequency bin
    Pxx = Xf.apply(np.abs)
    Pxx = Pxx**2 #power spectra
    if relative:
        Pxx = Pxx/np.sum(Pxx,axis=0) #power relative to total

    #power spectra between fm-fM Hz
    bin1 = int(timestep*n*fm)
    bin2 = int(timestep*n*fM)
    bins = np.linspace(bin1,bin2,nbins,dtype=int)
#     print(bins/(round(timestep*n)))

    #average power spectra within bins
    if binavg:
        deltab = int(0.5*np.diff(bins)[0]) #half the size of a bin (in samples)
        Pxxm = []
        for i in bins:
            start = int(max(i-deltab,bins[0]))
            end = int(min(i+deltab,bins[-1]))
            Pxxm.append(np.mean(Pxx.iloc[start:end,:].values,axis=0))
        Pxxm = np.asarray(Pxxm)
        Pxx = pd.DataFrame(data=Pxxm,index=Pxx.index[bins],columns=Pxx.columns)
        return Pxx, labels

    else:
        return Pxx.iloc[bins,:], labels
    
     
def feature_extraction_EMG(clip_data):
    
    #extract features for EMG
    
    features_list = ['RMS','range','mean','var','skew','kurt','Pdom_rel','Dom_freq','Sen','PSD_mean','PSD_std','PSD_skew','PSD_kurt']
    
    trial = list(clip_data.keys())[0]
        
    features = []
    for c in range(len(clip_data[trial]['elec']['data'])):
        rawdata = clip_data[trial]['elec']['data'][c]

        rawdata_wmag = rawdata.copy()

        N = len(rawdata)
        RMS = 1/N*np.sqrt(np.sum(rawdata**2))

        r = np.max(rawdata) - np.min(rawdata)

        mean = np.mean(rawdata)
        var = np.std(rawdata)
        sk = skew(rawdata)
        kurt = kurtosis(rawdata)

        Pxx = power_spectra_welch(rawdata_wmag,fm=20,fM=70)
        domfreq = Pxx.iloc[:,-1].idxmax()
        Pdom_rel = Pxx.loc[domfreq]/Pxx.iloc[:,-1].sum()

        Pxx_moments = np.array([np.nanmean(Pxx.values),np.nanstd(Pxx.values),skew(Pxx.values)[0],kurtosis(Pxx.values)[0]])

        x = rawdata.iloc[:,0]
        x = x[::5]
        n = len(x)
        Fs = np.mean(1/(np.diff(x.index)/1000))
        sH_raw = nolds.sampen(x)

        X = np.concatenate((RMS,r,mean,var,sk,kurt,Pdom_rel,np.array([domfreq,sH_raw])))
        Y = np.concatenate((X,Pxx_moments))
        features.append(Y)

    F = np.asarray(features)
    clip_data[trial]['elec']['features'] = pd.DataFrame(data=F,columns=features_list,dtype='float32')
    
def gen_clips_EMG(act_dict,subj,task,trial,location,clipsize=5000,overlap=0,verbose=False,startTS=0,endTS=1,len_tol=0.8,resample=False):

    #generate EMG clips on a specific trial
    
    clip_data = {} #the dictionary with clips
    clip_data[trial] = {}
    sensor = 'elec'

    rawdata = act_dict[task][trial][location][sensor]
    #reindex time (relative to start)
    idx = rawdata.index
    idx = idx-idx[0]
    rawdata.index = idx
    #choose to create clips only on a fraction of the data (0<[startTS,endTS]<1)
    if (startTS > 0) | (endTS < 1):
        rawdata = rawdata.iloc[round(startTS*len(rawdata)):round(endTS*len(rawdata)),:]
        #reindex time (relative to start)
        idx = rawdata.index
        idx = idx-idx[0]
        rawdata.index = idx
    #create clips data
    deltat = np.median(np.diff(rawdata.index))
    clips = []
    #use entire recording
    if clipsize == 0:
        clips.append(rawdata)
    #take clips
    else:
        idx = np.arange(0,rawdata.index[-1],clipsize*(1-overlap))
        for i in idx:
            c = rawdata[(rawdata.index>=i) & (rawdata.index<i+clipsize)]
            if len(c) > len_tol*int(clipsize/deltat): #discard clips whose length is less than len_tol% of the window size
                clips.append(c)

    #store clip length
    clip_len = [clips[c].index[-1]-clips[c].index[0] for c in range(len(clips))] #store the length of each clip
    #assemble in dict
    clip_data[trial][sensor] = {'data':clips, 'clip_len':clip_len}
    
    #Unused_Data = Unused_Data[['Subject','Trial','Task','Location']]
    #Unused_Data.to_csv('Z:CIS-PD Study\MotorTasks Unused EMG Data.csv')
    
    return clip_data

def feature_extraction_DWT(clip_data):
    
    #Extract EMG features from haar wavelets
    
    features_list = ['RMS','mean','var','skew','kurt','binen','energy','MAV']
    feature_cols = []
    for w in range(11):
        for i in range(len(features_list)):
            feature_cols = feature_cols + [features_list[i] + str(w)]
    
    trial = list(clip_data.keys())[0]
        
    features_data = []
    
    for c in range(len(clip_data[trial]['elec']['data'])):
        clip = clip_data[trial]['elec']['data'][c]
        features = []
        
        if np.max(clip)[0] > 10*np.mean(clip)[0]:
            
            features = np.full(88,np.nan)
        
        else:
            
            haar = pywt.Wavelet('haar')
            clip = clip.values[:,-1]
            DWT = pywt.wavedec(clip,haar,level=10)

            for w in range(len(DWT)):

                rawdata = DWT[w]
                rawdatab = rawdata.copy()

                N = len(rawdata)
                RMS = 1/N*np.sqrt(np.sum(rawdata**2))

                #r = np.max(rawdata) - np.min(rawdata)

                mean = np.mean(rawdata)
                var = np.std(rawdata)
                sk = skew(rawdata)
                kurt = kurtosis(rawdata)

                for i in range(len(rawdatab)):
                    rawdatab[i] = round((rawdatab[i] - mean)/var)
                binen = 0
                for i in np.unique(rawdatab):
                    count = 0
                    for j in rawdatab:
                        if j==i:
                            count += 1
                    p = count / N
                    binen += -1*p*math.log2(p)

                energy = 0
                for i in rawdata:
                    energy += i**2

                mav = np.mean(np.absolute(rawdata))

                features = features + [RMS,mean,var,sk,kurt,binen,energy,mav]
            
        
        features_data.append(np.array(features))

    F = np.asarray(features_data)
    clip_data[trial]['elec']['features'] = pd.DataFrame(data=F,columns=feature_cols,dtype='float32')
    
    
def HPfilter(rawdata,cutoff=0.75,ftype='highpass'):
#highpass (or lowpass) filter data. HP to remove gravity (offset - limb orientation) from accelerometer data from each visit (trial)
#input: rawdata, cutoff freq [Hz].

    if rawdata.empty is False: #skip if no data for current sensor
        idx = rawdata.index
        idx = idx-idx[0]
        rawdata.index = idx
        x = rawdata.values
        Fs = np.mean(1/(np.diff(rawdata.index)/1000)) #sampling rate
        #filter design
        cutoff_norm = cutoff/(0.5*Fs)
        b,a = butter(4,cutoff_norm,btype=ftype,analog=False)
        #filter data
        xfilt = filtfilt(b,a,x,axis=0)
        rawdatafilt = pd.DataFrame(data=xfilt,index=rawdata.index,columns=rawdata.columns)
        return rawdatafilt

    
def filterdata(rawdata,ftype='highpass',cutoff=0.75,cutoff_bp=[3,8],order=4):
    
    #takes rawdata as a parameter

    if not rawdata.empty:
        idx = rawdata.index
        idx = idx-idx[0]
        rawdata.index = idx
        x = rawdata.values
        #print(np.unique(np.diff(rawdata.index)))
        Fs = np.mean(1/(np.diff(rawdata.index)/1000)) #sampling rate
        if ftype != 'bandpass':
            #filter design
            cutoff_norm = cutoff/(0.5*Fs)
            b,a = butter(4,cutoff_norm,btype=ftype,analog=False)
        else:
            #filter design
            cutoff_low_norm = cutoff_bp[0]/(0.5*Fs)
            cutoff_high_norm = cutoff_bp[1]/(0.5*Fs)
            b,a = butter(order,[cutoff_low_norm,cutoff_high_norm],btype='bandpass',analog=False)

        #filter data
        xfilt = filtfilt(b,a,x,axis=0)
        rawdatafilt = pd.DataFrame(data=xfilt,index=rawdata.index,columns=rawdata.columns)
        return rawdatafilt
    
    
def gen_clips_mc10(rawdata,clipsize=5000,overlap=0.5,verbose=False,startTS=0,endTS=1,len_tol=0.8,downsample=62.5,basefreq=62.5):

    #Used for resampling sensor and watch data
    
    clip_data = {} #the dictionary with clips

    #reindex time (relative to start)
    idx = rawdata.index
    idx = idx-idx[0]
    rawdata.index = idx
    #choose to create clips only on a fraction of the data (0<[startTS,endTS]<1)
    if (startTS > 0) | (endTS < 1):
        rawdata = rawdata.iloc[round(startTS*len(rawdata)):round(endTS*len(rawdata)),:]
        #reindex time (relative to start)
        idx = rawdata.index
        idx = idx-idx[0]
        rawdata.index = idx
    #create clips data
    deltat = 1000/basefreq
    clips = []
    
    #interpolate to fixed interval
    f = interp1d(rawdata.index.values,rawdata.values.transpose(),kind='cubic')
    
    ts_start = 0
    ts_end = rawdata.index.max(); ts_end = ts_end-np.mod(ts_end,deltat)
    ts = np.linspace(ts_start,ts_end,int((ts_end-ts_start)/deltat+1))
    interp = f(ts)
    
    #resample data
    if basefreq!=downsample:
        if type(basefreq)!=int or type(downsample)!=int:
            interp = resample_poly(interp,downsample*2,basefreq*2,axis=1)
            ts = np.linspace(ts_start, ts_end, interp.shape[1])
        else:
            interp = resample_poly(interp,downsample,basefreq,axis=1)
            ts = np.linspace(ts_start, ts_end, interp.shape[1])
    
    rawdata_interp = pd.DataFrame(data=interp.transpose(),index=ts.astype(int))
    
    #use entire recording
    if clipsize == 0:
        clips.append(rawdata_interp)
 
    #take clips
    else:
        idx = np.arange(0,rawdata_interp.index[-1],clipsize*(1-overlap))
        for i in idx:
            c = rawdata_interp[(rawdata_interp.index>=i) & (rawdata_interp.index<=i+clipsize-deltat)]
            #skip clips whose length (in original signal) is less than len_tol% of the window size
            if sum((rawdata.index>=i) & (rawdata.index<=i+clipsize-deltat)) > len_tol*int(clipsize/deltat): 
#                     c_ref = c[::-1].copy()
#                     c_ref.index = c.index
#                 print(c)

                clips.append(c)
                #clips.append(c.copy()*-1)
                #clips.append(c_ref)
                #clips.append(c_ref.copy()*-1)


    #store clip length
    clip_len = [clips[c].index[-1]-clips[c].index[0] for c in range(len(clips))] #store the length of each clip
    #assemble in dict
    clip_data = {'data':clips, 'clip_len':clip_len}

    return clip_data


def feature_extraction(clip_data):
    
    features_list = ['RMSX','RMSY','RMSZ','rangeX','rangeY','rangeZ','meanX','meanY','meanZ','varX','varY','varZ',
                    'skewX','skewY','skewZ','kurtX','kurtY','kurtZ','Sen_X','Sen_Y','Sen_Z',
                    'xcor_peakXY','xcorr_peakXZ','xcorr_peakYZ','xcorr_lagXY','xcorr_lagXZ','xcorr_lagYZ',
                    'Dom_freqX','Pdom_relX','PSD_meanX','PSD_stdX','PSD_skewX','PSD_kurX',
                    'Dom_freqY','Pdom_relY','PSD_meanY','PSD_stdY','PSD_skewY','PSD_kurY',
                    'Dom_freqZ','Pdom_relZ','PSD_meanZ','PSD_stdZ','PSD_skewZ','PSD_kurZ',
                    'jerk_meanX','jerk_stdX','jerk_skewX','jerk_kurX',
                    'jerk_meanY','jerk_stdY','jerk_skewY','jerk_kurY',
                    'jerk_meanZ','jerk_stdZ','jerk_skewZ','jerk_kurZ',
                    'RMS_mag','range_mag','mean_mag','var_mag','skew_mag','kurt_mag','Sen_mag',
                    'Dom_freq_mag','Pdom_rel_mag','PSD_mean_mag','PSD_std_mag','PSD_skew_mag','PSD_kur_mag',
                    'jerk_mean_mag','jerk_std_mag','jerk_skew_mag','jerk_kur_mag']

    if len(clip_data['data'])<1:
        clip_data['features'] = pd.DataFrame(columns=features_list)
        return
    
    #cycle through all clips for current trial and save dataframe of features for current trial and sensor
    features = []
    ts = []
    for c in range(len(clip_data['data'])):
        t = []
        t1 = time.time() 
        rawdata = clip_data['data'][c]
        rawdata_unfilt = rawdata.copy()
        
        Fs = np.mean(1/(np.diff(rawdata.index)/1000)) #sampling rate
        cutoff_norm = 0.75/(0.5*Fs)
        if cutoff_norm>1:
            print(Fs,cutoff_norm)
            continue
        
        rawdata = filterdata(rawdata)
        t2 = time.time()
        t.append(t2-t1) #append shared preprocessing time
        
        
        t1 = time.time()
        #acceleration magnitude
        rawdata_wmag = rawdata_unfilt
        rawdata_wmag['Accel_Mag']=np.sqrt((rawdata_unfilt**2).sum(axis=1))
        t2 = time.time()
        t.append(t2-t1) #append magnitude computation time

        
        #extract features on current clip

        t1 = time.time()
        #Root mean square of signal on each axis
        N = len(rawdata)
        RMS = 1/N*np.sqrt(np.asarray(np.sum(rawdata**2,axis=0)))

        #range on each axis
        min_xyz = np.min(rawdata,axis=0)
        max_xyz = np.max(rawdata,axis=0)
        r = np.asarray(max_xyz-min_xyz)
        
        #Moments on each axis
        mean = np.asarray(np.mean(rawdata,axis=0))
        var = np.asarray(np.std(rawdata,axis=0))
        sk = skew(rawdata)
        kurt = kurtosis(rawdata)

        t2 = time.time()
        t.append(t2-t1) # append time domain features
        
        
        t1 = time.time()
        #sample entropy raw data (magnitude) and FFT
        sH_raw = []; #sH_fft = []
        
        for a in range(3):
            x = rawdata.iloc[:,a]
            n = len(x) #number of samples in clip
            Fs = np.mean(1/(np.diff(x.index)/1000)) #sampling rate in clip
            sH_raw.append(nolds.sampen(x)) #samp entr raw data
            #for now disable SH on fft
            # f,Pxx_den = welch(x,Fs,nperseg=min(256,n/4))
            # sH_fft.append(nolds.sampen(Pxx_den)) #samp entr fft
            
        t2 = time.time()
        t.append(t2-t1) # append Sen features time
        
        
        t1 = time.time()
        #Cross-correlation between axes pairs
        xcorr_xy = np.correlate(rawdata.iloc[:,0],rawdata.iloc[:,1],mode='same')
        # xcorr_xy = xcorr_xy/np.abs(np.sum(xcorr_xy)) #normalize values
        xcorr_peak_xy = np.max(xcorr_xy)
        xcorr_lag_xy = (np.argmax(xcorr_xy))/len(xcorr_xy) #normalized lag

        xcorr_xz = np.correlate(rawdata.iloc[:,0],rawdata.iloc[:,2],mode='same')
        # xcorr_xz = xcorr_xz/np.abs(np.sum(xcorr_xz)) #normalize values
        xcorr_peak_xz = np.max(xcorr_xz)
        xcorr_lag_xz = (np.argmax(xcorr_xz))/len(xcorr_xz)

        xcorr_yz = np.correlate(rawdata.iloc[:,1],rawdata.iloc[:,2],mode='same')
        # xcorr_yz = xcorr_yz/np.abs(np.sum(xcorr_yz)) #normalize values
        xcorr_peak_yz = np.max(xcorr_yz)
        xcorr_lag_yz = (np.argmax(xcorr_yz))/len(xcorr_yz)

        #pack xcorr features
        xcorr_peak = np.array([xcorr_peak_xy,xcorr_peak_xz,xcorr_peak_yz])
        xcorr_lag = np.array([xcorr_lag_xy,xcorr_lag_xz,xcorr_lag_yz])

        t2=time.time()
        t.append(t2-t1) # append xcorr computation time
        
        
        t1 = time.time()
        axes_F = np.array([])
        for a in range(3):
            x = rawdata.iloc[:,a]
            n = len(x) #number of samples in clip
            Fs = np.mean(1/(np.diff(x.index)/1000)) #sampling rate in clip
            f,Pxx_den = welch(x,Fs,nperseg=min(256,n))
            Pxx = pd.DataFrame(data=Pxx_den,index=f,columns=['PSD'])
            F_rel = np.asarray([Pxx.iloc[Pxx.index<12,-1].idxmax()])
            P_rel = Pxx.loc[F_rel].iloc[:,-1].values/Pxx.iloc[Pxx.index<12,-1].sum()
            F_moments = np.array([np.nanmean(Pxx.values),np.nanstd(Pxx.values),skew(Pxx.values),kurtosis(Pxx.values)])
            axes_F = np.concatenate((axes_F,F_rel,P_rel,F_moments))
        
        t2 = time.time()
        t.append(t2-t1) # append frequency axes computation time
        
        
        t1 = time.time()
        #moments of jerk axes
        axes_D = np.array([])
        for a in range(3):
            ax = rawdata.iloc[:,a].diff().values
            ax_moments = np.array([np.nanmean(ax),np.nanstd(ax),skew(ax[~np.isnan(ax)]),kurtosis(ax[~np.isnan(ax)])])
            axes_D = np.concatenate([axes_D,ax_moments])
        t2 = time.time()
        t.append(t2-t1) # append axes derivative computation time

        
        t1 = time.time()
        RMS_mag = 1/N*np.sqrt(np.sum(rawdata_wmag['Accel_Mag']**2,axis=0))
        r_mag = np.max(rawdata_wmag['Accel_Mag']) - np.min(rawdata_wmag['Accel_Mag'])
        mean_mag = np.mean(rawdata_wmag['Accel_Mag'])
        var_mag = np.std(rawdata_wmag['Accel_Mag'])
        sk_mag = skew(rawdata_wmag['Accel_Mag'])
        kurt_mag = kurtosis(rawdata_wmag['Accel_Mag'])
        t2 = time.time()
        t.append(t2-t1) # append magnitude time domain computation time
        
        
        t1 = time.time()
        sH_mag = nolds.sampen(rawdata_wmag['Accel_Mag'])
        t2 = time.time()
        t.append(t2-t1) # append magnitude entropy computation time

        
        t1 = time.time()
        #Dominant freq and relative magnitude (on acc magnitude)
        Pxx = power_spectra_welch(rawdata_wmag,fm=0,fM=Fs)
        domfreq = np.asarray([Pxx.iloc[Pxx.index<12,-1].idxmax()])
        Pdom_rel = Pxx.loc[domfreq].iloc[:,-1].values/Pxx.iloc[Pxx.index<12,-1].sum() #power at dominant freq rel to total

        #moments of PSD
        Pxx_moments = np.array([np.nanmean(Pxx.values),np.nanstd(Pxx.values),skew(Pxx.values),kurtosis(Pxx.values)])
        t2 = time.time()
        t.append(t2-t1) # append magnitude frequency computation time
        
        
        t1 = time.time()
        #moments of jerk magnitude
        jerk = rawdata_wmag['Accel_Mag'].diff().values
        jerk_moments = np.array([np.nanmean(jerk),np.nanstd(jerk),skew(jerk[~np.isnan(jerk)]),kurtosis(jerk[~np.isnan(jerk)])])
        t2 = time.time()
        t.append(t2-t1) # append magnitude derivative computation time
        
        #Assemble features in array
        Y = np.array([RMS_mag,r_mag,mean_mag,var_mag,sk_mag,kurt_mag,sH_mag])
        X = np.concatenate((RMS,r,mean,var,sk,kurt,sH_raw,xcorr_peak,xcorr_lag,axes_F,axes_D,Y,domfreq,Pdom_rel,Pxx_moments,jerk_moments))
        features.append(X)
        ts.append(t)

    F = np.asarray(features) #feature matrix for all clips from current trial
    clip_data['features'] = pd.DataFrame(data=F,columns=features_list,dtype='float32')
    
    return ts, F.shape[0]