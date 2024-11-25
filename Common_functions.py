# -*- coding: utf-8 -*-

import glob,os,sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from numpy import *
import re
from tqdm import tqdm
import scipy.stats as st
from os.path import expanduser
import cmath
import scipy.optimize as op
import time
import gzip
from scipy.interpolate import interp1d
import pandas as pd
from scipy.interpolate import interp1d
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import (mark_inset,inset_axes,InsetPosition) 
import traceback
import scipy.special as sp


home = os.path.expanduser("~")
sys.path.insert(1, home+'/Library/Mobile Documents/com~apple~CloudDocs/Research/WFIRST-event-finder/Tool_Package/')

from savgol import savitzky_golay





def localize_event(lightcurve,t0,tE):
    """Function to estimate roughly the area around the peak of an event, 
    and identify which timestamps in the array belong to the event versus
    the baseline
    """
    
    idx1 = np.where(lightcurve >= t0-tE)[0]
    idx2 = np.where(lightcurve <= t0+tE)[0]
    event = list(set(idx1).intersection(set(idx2)))
    
    baseline = np.arange(0,len(lightcurve),1)
    baseline = np.delete(baseline,event)
    
    it0 = np.where(lightcurve == t0)[0][0]
    
    #print min(lightcurve)
    #print it0
    return baseline, event, it0

def prepare(t,m,err):
    
    df = pd.DataFrame({'t': t, 'magnitude': m, 'm_err': err})
    peaks = np.array([t[np.argmin(m)]])
    baseline, event, it0 = localize_event(df['t'], peaks[0],50)
    
    base_mag = np.median(df['magnitude'][baseline])
    df['A'] = 10 ** (0.4*(base_mag - df['magnitude']))
    
    interpol = interp1d(df['t'],df['A'], kind='cubic')
    dt = np.abs(df['t'][np.argmin(np.abs(interpol(df['t'])-1.06))]-peaks[0])
    #print dt
    
    if dt==0.0:
        dt = 50


        
    
    #dt = 50
    # baseline, event, it0 = localize_event(df['t'], peaks[0],dt)


    A_max = 10 ** (0.4*(base_mag - (df['magnitude']-df['m_err'])))
    A_min = 10 ** (0.4*(base_mag - (df['magnitude']+df['m_err'])))
    df['A_err'] = (A_max - A_min)/2
    

    while (np.abs((df['t'][event]).diff())).max() > 0.1:
        
        if dt>20:
            dt = dt - 10
            baseline, event, it0 = localize_event(df['t'], peaks[0],dt)
        else:
            break
    #print dt    
    return df,baseline, event, it0, dt




def empty(df):
    return len(df.index) == 0

def fun (t,t0,tE, u0, fs):
    u = np.sqrt(u0**2+((t-t0)/tE)**2)
    A = ((u**2)+2)/(u*np.sqrt(u**2+4))
    F = fs*A +(1-fs)
    return F
        
def fun2 (t, mean, sigma,amp, t0,tE, u0, fs):
    u = np.sqrt(u0**2+((t-t0)/tE)**2)
    A = (((amp/np.sqrt(2*pi*(sigma**2)))*np.exp(-((t-mean)**2)/(2*(sigma**2)))))+((u**2)+2)/(u*np.sqrt((u**2)+4))
    F = fs*A +(1-fs)
    return F

def lnlike(theta, t, f, f_err):
    t0, tE, u0, fs = theta
    model = fun(t, t0, tE, u0, fs)
    inv_sigma2 = 1.0/(f_err**2)
    return -0.5*(np.sum((f-model)**2*inv_sigma2))

def lnlike_cauchy(theta, t, f, f_err):
    c, a, b, d = theta
    model = bell_curve(t, c, a, b, d)
    inv_sigma2 = 1.0/(f_err**2)
    return -0.5*(np.sum((f-model)**2*inv_sigma2))

def lnlike_trap(theta, t, f, f_err):
    amp, t0, del_tau1, del_tau2 = theta
    model = trapezoid2(t, amp, t0, del_tau1, del_tau2)
    inv_sigma2 = 1.0/(f_err**2)
    return -0.5*(np.sum((f-model)**2*inv_sigma2))

    

def lnlike2(theta, t, f, f_err):
    mean, sigma,amp, t0, tE, u0, fs = theta
    model = fun2(t,mean, sigma,amp, t0, tE, u0, fs)
    inv_sigma2 = 1.0/(f_err**2)
    return -0.5*(np.sum((f-model)**2*inv_sigma2))

def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth


def max_finder (t, A, f_err, it0, dt, event):
    
    t_event = t
    A_event = A
    f_err_event = f_err
    
    A_max = max(A_event[A_event <np.percentile(A_event,[0,100] )[1]]) #1.0/(float(f_s_true)/(max(df['f'])-1+float(f_s_true)))
    u0_true = np.sqrt( ( (1+np.sqrt(1+16*(A_max**2)))/(2* A_max) ) - 2 )
    t0_true =  t_event[np.argmax(A_event[A_event <np.percentile(A_event,[0,100] )[1]])]
    
    return t0_true



def cal_chisqr(model, f, ferr):
    
    
    return np.sum(((f-model)**2)/((ferr)**2))

def F_t (t, t0, t_eff, f_1, f_0):

	Q = 1 + ((t-t0)/t_eff)**2

	F = f_1 *(Q**(-1.0/2) + (1 - (1 + Q/2)**-2)**(-1.0/2)) +f_0

	return F

def Gould_2_par_PSPL (t, m, t0, t_eff, f1, f0):
    
    t0_ini = t0
    t_eff_ini = t_eff
    
    paramt = [t0_ini, t_eff_ini, f1, f0]
    
    popt, pcov = scipy.optimize.curve_fit(F_t, t, m, p0=paramt)
    
    
    
    return popt




def cal_chisqr_modes(lightcurve,fx,ftype = 'm'):
    """Function to calculate the chi squared of the fit of the lightcurve
    data to the function provided"""
    if ftype=='m':
        chisq = ((lightcurve[:,1] - fx)**2 / fx).sum()
        
    if ftype=='A':
        chisq = ((lightcurve[:,3] - fx)**2 / fx).sum()
    
    return chisq
    
def bell_curve(x,c,a,b,d):
    """Function describing a bell curve of the form:
    f(x; a,b,c,d) = d / [1 + |(x-c)/a|^(2b)]
    
    Inputs:
    :param  np.array x: Series of intervals at which the function should
                        be evaluated
    :param float a,b,c: Coefficients of the bell curve function
    
    Returns:
    :param np.array f(x): Series of function values at the intervals in x
    """
    
    fx = d / ( 1.0 + (abs( (x-c)/a ))**(2.0*b) )
    
    return fx+1

def bell_curve_data(params, t, A_data):
    """Function describing a bell curve of the form:
    f(x; a,b,c,d) = d / [1 + |(x-c)/a|^(2b)]
    
    Inputs:
    :param  np.array x: Series of intervals at which the function should
                        be evaluated
    :param float a,b,c: Coefficients of the bell curve function
    
    Returns:
    :param np.array f(x): Series of function values at the intervals in x
    """
    c = params['t0'].value
    a = params['tE'].value
    b = params['b'].value
    d = params['amp'].value
    
    fx = (d / ( 1.0 + (abs( (t-c)/a ))**(2.0*b) ))
    
    return fx+1 - A_data

# def gaussian(x,a,b,c):
#     """Function describing a Gaussian of the form:
#     f(x; a,b,c) = a * exp(-(x-b)**2/2c*2)
    
#     Inputs:
#     :param  np.array x: Series of intervals at which the function should
#                         be evaluated
#     :param float a,b,c: Coefficients of the bell curve function
    
#     Returns:
#     :param np.array f(x): Series of function values at the intervals in x
#     """
    
#     fx = a * np.exp(-( (x-b)**2 / (2*c*c) ))
    
#     return fx

def PSPL_data (params, t, A_data):
    
    t0 = params['t0'].value
    tE = params['tE'].value
    u0 = params['u0'].value
    fs = params['fs'].value
#     fb = params['fb'].value

    u = np.sqrt(u0**2+((t-t0)/tE)**2)
    A = ((u**2)+2)/(u*np.sqrt(u**2+4))
    F = (fs * (A-1)) +1
    return F - A_data

def PSPL (t0, tE, u0,fs, t):
    
    u = np.sqrt(u0**2+((t-t0)/tE)**2)
    A = ((u**2)+2)/(u*np.sqrt(u**2+4))
    F = (fs * (A-1)) +1
    return F


# Smoothing the data

def low_pass_filter(y, box_pts, mode='same', base=1):
    box = base*(np.ones(box_pts)/box_pts)
    y_filtered = np.convolve(y, box, mode=mode)
    if mode=='same':
        y_filtered[0:int(box_pts/2)]=y_filtered[int(box_pts/2)]
        y_filtered[len(y_filtered)-int(box_pts/2):len(y_filtered)]=y_filtered[len(y_filtered)-int(box_pts/2)]
    return y_filtered

def count_peaks (t, m,  smooth='yes', bin_size = 30, threshold = 3):
    
    if smooth == 'yes':
        m = low_pass_filter(m,8)
    else:
        pass
    df_ = pd.DataFrame({'t':t, 'm':m})

    bins = np.linspace(df_['t'].min(),df_['t'].max(),int((df_['t'].max()-df_['t'].min())/bin_size))
    # print bins
    groups = df_.groupby(np.digitize(df_['t'], bins))

    # print(groups.)
    
    std_ = np.std(m)

    delta_m = []
    t__ = []
    c = 0
    for i in groups.indices:
        #print c
        try:
            c = c+1
            #print i
            m_ = df_['m'][groups.indices[i]]
            t_ = df_['t'][groups.indices[i]]
            # std_ = np.std(df_['m'][groups.indices[i]])
            #print t,m
            del_m = np.asarray((np.abs(m_- m_.mean())/std_))
            delta_m.append(del_m)
            t__.append(np.asarray(t_))
        except:
            pass

    peaks = []    
    n_outliers = []
    for j in range(len(delta_m)):
        n_temp = len(np.where(delta_m[j]>threshold)[0])
        n_outliers.append(n_temp) 
        if n_temp > 5:
            peaks.append(t__[j][np.argmax(delta_m[j])])
    return n_outliers, peaks

# This is a new version of count_peaks function for finding deviations in the residual of the PSPL model
def find_roots(x,y):
    s = np.abs(np.diff(np.sign(y))).astype(bool)
    return x[:-1][s] + np.diff(x)[s]/(np.abs(y[1:][s]/y[:-1][s])+1)
    
def residual_peak_finder(lc, new_t_len=400, threshold1=5, threshold2=10,\
                         bin_size=50, med_med_threshold=1, verbose=False):
    
    peaks = []
    troughs = []

    peaks_A = []
    selected_peaks = []

    troughs_A = []
    selected_troughs = []

    lc.three_sigma_peaks = np.nan
    lc.three_sigma_troughs = np.nan


    lc.df_selected_peaks = pd.DataFrame({'t':[], 'del_A':[]})
    lc.df_selected_troughs = pd.DataFrame({'t':[], 'del_A':[]})
    lc.df_selected_new_peaks = pd.DataFrame({'t':[],'del_A':[]})
    lc.df_selected_new_troughs = pd.DataFrame({'t':[],'del_A':[]})

    lc.threshold_peaks = np.nan
    lc.threshold_troughs = np.nan


    try:
        std_ = np.std(savitzky_golay(lc.df.A_residual.values[lc.baseline], 11, 3))
        mean_ = np.mean(savitzky_golay(lc.df.A_residual.values[lc.baseline], 11, 3))
    except:
        std_ = np.std(lc.df.A_residual.values[lc.baseline])
        mean_ = np.mean(lc.df.A_residual.values[lc.baseline])

    peaks_, t_selected_peaks,m_selected_peaks, del_A_selected_peaks,\
    troughs_, t_selected_troughs,m_selected_troughs, del_A_selected_troughs =count_deviations(lc.df.t,  savitzky_golay(lc.df.A_residual.values, 11, 3),\
                                                                        std_baseline = std_,\
                                                                        mean_baseline = mean_,\
                                                                        smooth='no', bin_size =bin_size,\
                                                                        threshold = threshold1)



    
    low_S_N_R = False
    std_lim = 1

    # Check lc.event to see if it includes the peak and the trough. Somtimes the event should be expanded.
    try:
        if np.max([peaks_-lc.t0_true, troughs_-lc.t0_true])> 2*lc.tE_true:
            idx1 = np.where(lc.df.t >= lc.t0_true-(np.max([peaks_-lc.t0_true, troughs_-lc.t0_true])+2))[0]
            idx2 = np.where(lc.df.t <= lc.t0_true+(np.max([peaks_-lc.t0_true, troughs_-lc.t0_true])+2))[0]
            lc.event = list(set(idx1).intersection(set(idx2)))
    except:
        # print([peaks_-lc.t0_true, troughs_-lc.t0_true])
        pass

    check_noise_level = np.abs(1-(((np.median(lc.df.A_residual[lc.event].values))+1)/\
                                  ((np.mean(lc.df.A_residual[lc.event].values)+1))))*100
    lc.check_noise_level = check_noise_level
    if verbose:
        print('Noise level is: ', check_noise_level)



#     print(check_noise_level)
    
    
    if check_noise_level<0.01:
        low_S_N_R = True
        std_lim = 0.1
        sav_lim = 101

    elif check_noise_level<0.02:
        low_S_N_R = True
        sav_lim = 101
        std_lim = 0.5
        # print(low_S_N_R)
    elif check_noise_level > 1:
        sav_lim = 11
    
    else:
        sav_lim = 51
        std_lim = 0.5
    
    
    # print(min(del_A_selected_peaks), max(del_A_selected_peaks))


#     print(len(peaks_),len(troughs_))
    if len(peaks_)==0 or len(troughs_)==0:

        if low_S_N_R:
            try:
                std_ = np.std(savitzky_golay(lc.df.A_residual.values[lc.baseline], 51, 3))
                mean_ = np.mean(savitzky_golay(lc.df.A_residual.values[lc.baseline], 51, 3))
            except:
                std_ = np.std(lc.df.A_residual.values[lc.baseline])
                mean_ = np.mean(lc.df.A_residual.values[lc.baseline])

            peaks_, t_selected_peaks, m_selected_peaks, del_A_selected_peaks,\
            troughs_, t_selected_troughs, m_selected_troughs, del_A_selected_troughs =count_deviations(lc.df.t,  savitzky_golay(lc.df.A_residual.values, 11, 3),\
                                                                                std_baseline = std_,\
                                                                                mean_baseline = mean_,\
                                                                                smooth='no', bin_size =bin_size,\
                                                                                threshold = threshold1)

    if len(peaks_)==0 and len(troughs_)==0:
        print('No deviations were found for ',lc.name)
        return peaks, troughs

    elif len(peaks_)==0 or len(troughs_)==0:
        if len(peaks_)==0:
            print (print('No peaks were found for ',lc.name))
        else:
            print (print('No troughs were found for ',lc.name))



            

    if (len(peaks_)==1) and (not len(t_selected_peaks)<3):
        
        
        lc.df_selected_peaks = pd.DataFrame({'t':t_selected_peaks,'m': m_selected_peaks, 'del_A':del_A_selected_peaks,\
                                             'del_A_savgol': savitzky_golay(del_A_selected_peaks,sav_lim, 3)})


        thre_peaks = np.percentile(lc.df_selected_peaks.del_A_savgol,98)

        lim_ = 98
        while np.std(lc.df_selected_peaks.del_A_savgol[lc.df_selected_peaks.del_A_savgol>thre_peaks]) < std_lim:
            lim_ = lim_ - 2
            thre_peaks = np.percentile(lc.df_selected_peaks.del_A_savgol,lim_)
            if lim_<70 and np.std(lc.df_selected_peaks.del_A_savgol[lc.df_selected_peaks.del_A_savgol>thre_peaks])<std_lim:
                print('No significant peaks were found.')
                # print(sav_lim,thre_peaks)
                thre_peaks = np.nan
                lc.threshold_peaks == []
                break

        # print(lim_,thre_peaks)

        if not np.isnan(thre_peaks):

            lc.threshold_peaks = thre_peaks
                
            if verbose:
                print('Peak detection threshold is ',thre_peaks)
            
            z_peaks = find_roots(lc.df_selected_peaks.t.values,\
                                 lc.df_selected_peaks.del_A_savgol.values-thre_peaks)

            lc.z_peaks = z_peaks

            if verbose:
                print('Roots of selected_A - threshold_peak: ', z_peaks)

            peaks_diff = np.diff(z_peaks)

            groups_peaks = lc.df_selected_peaks.groupby(np.digitize(lc.df_selected_peaks['t'], z_peaks))

            if low_S_N_R:
                med_ = (2.5*np.std(lc.df_selected_peaks['del_A_savgol']))
            else:
                med_ = (3*np.std(lc.df_selected_peaks['del_A_savgol']))

            lc.three_sigma_peaks = med_

            if verbose:
                print('3sigma_peaks = ', med_)

            for i in range(len(groups_peaks.indices)):

                try: 
                    xxx = len(lc.df_selected_peaks['t'][groups_peaks.indices[i]])== 0
                except:
                    continue
                
                interval_peaks = (max(lc.df_selected_peaks['t'][groups_peaks.indices[i]])-min(lc.df_selected_peaks['t'][groups_peaks.indices[i]]))
                median_peaks = np.median(lc.df_selected_peaks['del_A_savgol'][groups_peaks.indices[i]])
                med_med_peaks = np.std(lc.df_selected_peaks['del_A_savgol'][groups_peaks.indices[i]])
                percentile_ = np.percentile(lc.df_selected_peaks['del_A_savgol'][groups_peaks.indices[i]], 99)

                # if verbose:
                    # print(median_peaks, med_med_peaks, percentile_ )

                if (not len(lc.df_selected_peaks['del_A_savgol'][groups_peaks.indices[i]]) ==0) & (percentile_>med_) & (median_peaks>thre_peaks):#median_peaks>threshold2 and med_med_peaks>med_med_threshold:
                    if len(lc.df_selected_peaks['del_A_savgol'][groups_peaks.indices[i]])>3:
                        selected_peaks.append(percentile_)
                        # if verbose:
                            # print('here')
                        peaks.append(lc.df_selected_peaks['t'][groups_peaks.indices[i]][lc.df_selected_peaks['del_A_savgol'][groups_peaks.indices[i]].idxmax()])
                        peaks_A.append(lc.df_selected_peaks.m[groups_peaks.indices[i]].max())
            
            if len(peaks)>2:
                if verbose:
                    # print(selected_peaks-np.mean(selected_peaks))
                    print('More than 2 peaks were found. Larger peaks are picked.')
                peaks = np.array(peaks)[(selected_peaks-np.mean(selected_peaks))>std_lim]
                peaks_A = np.array(peaks_A)[(selected_peaks-np.mean(selected_peaks))>std_lim]

            # elif len(peaks) == 2:
                # if 



            # if not len(peaks)==0:
            #     if not len(np.where(np.diff(peaks)<3)[0])==0:
            #         ind = np.array(np.append(np.where(np.diff(peaks)<3)[0],np.where(np.diff(peaks)<3)[0][-1]+1))
            #         peaks_ = np.asarray(peaks)[ind]
            #         selected__ = np.asarray(selected_peaks)[ind]
            #         peaks_A_ = np.asarray(peaks_A)[ind]
            #         ind_ = [i for i in range(len(peaks)-1) if i not in ind]
            #         if len(peaks_)>1 and np.mean(np.diff(peaks_))<2 and np.std(selected__)<0.5:
            #             peaks = np.append(np.asarray(peaks)[ind_], peaks_[np.argmax(peaks_A_)])

    
    
    if (len(peaks_)==1) and (not len(t_selected_troughs)<3):


        
        lc.df_selected_troughs = pd.DataFrame({'t':t_selected_troughs,'m':m_selected_troughs, 'del_A':del_A_selected_troughs,\
                                                'del_A_savgol': savitzky_golay(del_A_selected_troughs, sav_lim, 3)})

        thre_troughs = np.percentile(lc.df_selected_troughs.del_A_savgol,2)

        # print(np.percentile(lc.df_selected_troughs.del_A_savgol,4))#/3

        lim_ = 2
        # print(sav_lim)
        while np.std(lc.df_selected_troughs.del_A_savgol[lc.df_selected_troughs.del_A_savgol<thre_troughs])<std_lim:
            lim_ = lim_ + 2
            thre_troughs = np.percentile(lc.df_selected_troughs.del_A_savgol,lim_)
            # print(lim_, np.std(lc.df_selected_troughs.del_A_savgol[lc.df_selected_troughs.del_A_savgol<thre_troughs]))
            if lim_>30 and np.std(lc.df_selected_troughs.del_A_savgol[lc.df_selected_troughs.del_A_savgol<thre_troughs])<std_lim:
                print('No significant troughs were found.')
                # print(lim_ , thre_troughs)
                thre_troughs = np.nan

                break
        # print(thre_troughs)

        if not np.isnan(thre_troughs):

        
            # if min(del_A_selected_troughs)>-3:
            #   thre_troughs = -1

            if verbose:
                print('Trough detection threshold is ',thre_troughs)

            lc.threshold_troughs = thre_troughs

            # if low_S_N_R:
            # lc.df_selected_troughs['del_A'] = smooth(lc.df_selected_troughs['del_A'],70)
                
            z_troughs = find_roots(lc.df_selected_troughs.t.values,\
                         lc.df_selected_troughs.del_A_savgol.values-thre_troughs)

            troughs_diff = np.diff(z_troughs)

            lc.z_troughs = z_troughs    

            if verbose:
                try:
                    print('Roots of selected_A - threshold_trough: ', z_trough)
                except:
                    pass


            groups_troughs = lc.df_selected_troughs.groupby(np.digitize(lc.df_selected_troughs['t'], z_troughs))

            if low_S_N_R:
                med_ = (-2.5*np.std(lc.df_selected_troughs['del_A_savgol']))
            else:
                med_ = (-3*np.std(lc.df_selected_troughs['del_A_savgol']))

            lc.three_sigma_troughs = med_

            if verbose:
                print('3sigma_troughs = ', med_)

            for i in range(len(groups_troughs.indices)):

                try: 
                    xxx = len(lc.df_selected_troughs['t'][groups_troughs.indices[i]])== 0
                except:
                    continue
                
                interval_troughs = (max(lc.df_selected_troughs['t'][groups_troughs.indices[i]])-min(lc.df_selected_troughs['t'][groups_troughs.indices[i]]))
                median_troughs = np.median(lc.df_selected_troughs['del_A_savgol'][groups_troughs.indices[i]])
                med_med_troughs = np.std(lc.df_selected_troughs['del_A_savgol'][groups_troughs.indices[i]])
                percentile_ = np.percentile(lc.df_selected_troughs['del_A_savgol'][groups_troughs.indices[i]], 1)

                # if verbose:
                    # print(median_troughs, med_med_troughs, percentile_ )

                if (not len(lc.df_selected_troughs['del_A_savgol'][groups_troughs.indices[i]])==0) & (percentile_<med_) & (median_troughs<thre_troughs):#median_troughs<threshold2 and med_med_troughs>med_med_threshold:
                    if len(lc.df_selected_troughs['del_A_savgol'][groups_troughs.indices[i]])>3:
                        selected_troughs.append(percentile_)
                        # if verbose:
                            # print('here')
                        troughs.append(lc.df_selected_troughs['t'][groups_troughs.indices[i]][lc.df_selected_troughs['del_A_savgol'][groups_troughs.indices[i]].idxmin()])
                        troughs_A.append(lc.df_selected_troughs.m[groups_troughs.indices[i]].min())

            if len(troughs)>2:
                if verbose:
                    print('More than 2 troughs were found. Larger troughs are picked.')
                troughs = np.array(troughs)[(np.mean(selected_troughs) - selected_troughs)>std_lim]
                

                troughs_A = np.array(troughs_A)[(np.mean(selected_troughs) - selected_troughs)>std_lim]


            # if not len(troughs)==0:
            #     if not len(np.where(np.diff(troughs)<3)[0])==0:

            #         ind = np.array(np.append(np.where(np.diff(troughs)<3)[0],np.where(np.diff(troughs)<3)[0][-1]+1))
            #         troughs_ = np.asarray(troughs)[ind]
            #         selected__ = np.asarray(selected_troughs)[ind]
            #         troughs_A_ = np.asarray(troughs_A)[ind]
            #         ind_ = [i for i in range(len(troughs)-1) if i not in ind]
            #         if len(troughs_)>1 and np.mean(np.diff(troughs_))<2 and np.std(selected__)<0.5:
            #             troughs = np.append(np.asarray(troughs)[ind_], troughs_[np.argmin(troughs_A_)])
        
    
    if verbose:
        print('The peaks are: ', peaks)
        print('The troughs are:', troughs)
    return peaks, peaks_A, troughs, troughs_A

def count_deviations (t, m, std_baseline = None, mean_baseline = None,  smooth='yes', bin_size = 60, threshold = 3, shift=0):
    
    
    df_ = pd.DataFrame({'t':t, 'm':m})

    bins = np.linspace(df_['t'].min(),df_['t'].max(),int((df_['t'].max()-df_['t'].min())/bin_size))
    # print bins
    groups = df_.groupby(np.digitize(df_['t'], bins+shift))
    
    if std_baseline == None:
        std_ = np.std(m)
    else:
        std_ = std_baseline

    if mean_baseline == None:
        mean_ = np.mean(m)
    else:
        mean_ = mean_baseline

    delta_m = []
    t__ = []
    m__ = []
    c=0
    t__selected, delta_m_selected = [], []
    for i in groups.indices:
        #print c
        c = c+1
        #print i
        m_ = df_['m'][groups.indices[i]]
        t_ = df_['t'][groups.indices[i]]
        # std_ = np.std(df_['m'][groups.indices[i]])
        #print t,m
        del_m = np.asarray(((m_- mean_)/std_))
        delta_m.append(del_m)
        t__.append(np.asarray(t_))
        m__.append(np.asarray(m_))
    peaks = []
    troughs = []    
    t__selected_peaks, m__selected_peaks, delta_m_selected_peaks,\
    t__selected_troughs, m__selected_troughs, delta_m_selected_troughs = [], [], [],[], [],[]
    n_outliers_peaks = []
    n_outliers_troughs = []

    percentile_peaks = []
    percentile_troughs = []

    for j in range(len(delta_m)):
        # n_temp_peak = len(np.where(delta_m[j]>threshold)[0])
        # n_temp_troughs = len(np.where(delta_m[j]<-1*threshold)[0])
        # n_outliers_peaks.append(n_temp_peak)
        # n_outliers_troughs.append(n_temp_troughs) 

        # if not low_S_N_R:

        #     if n_temp_peak > 3:
        #         t__selected_peaks = t__[j]
        #         delta_m_selected_peaks = delta_m[j]
        #         peaks.append(t__[j][np.argmax(delta_m[j])])

        #     if n_temp_troughs > 3:
        #         t__selected_troughs = t__[j]
        #         delta_m_selected_troughs = delta_m[j]
        #         troughs.append(t__[j][np.argmin(delta_m[j])])
        # else:
            
        percentile_peaks.append(np.percentile(delta_m[j],99.99))
        percentile_troughs.append(np.percentile(delta_m[j],0.01))
            
    # if low_S_N_R:
        
    if np.sort(percentile_peaks)[-1] - np.mean(np.sort(percentile_peaks)[:-1]) >1:
        ind = np.argmax(percentile_peaks)
        peaks.append(t__[ind][np.argmax(delta_m[ind])])
        t__selected_peaks = t__[ind]
        m__selected_peaks = m__[ind]
        delta_m_selected_peaks = delta_m[ind]
    
    if np.mean(np.sort(percentile_troughs)[1:]) - np.sort(percentile_troughs)[0]>1:
        ind = np.argmin(percentile_troughs)
        troughs.append(t__[ind][np.argmin(delta_m[ind])])
        t__selected_troughs = t__[ind]
        m__selected_troughs = m__[ind]
        delta_m_selected_troughs = delta_m[ind]



    return peaks, t__selected_peaks,m__selected_peaks, delta_m_selected_peaks,\
           troughs, t__selected_troughs, m__selected_troughs, delta_m_selected_troughs

def busy_data (params, t, A_data):
    
    xe = params['xe'].value
    xp = params['xp'].value
    b1 = params['b1'].value
    b2 = params['b2'].value
    a = params['a'].value
    n = params['n'].value
    w = params['w'].value
    c = params['c'].value
    s = params['s'].value
    
    
    A = (a/4.)* (sp.erf(b1*(w+(s*t)-xe))+1) * (sp.erf(b2*(w-(s*t)+xe))+1) * (c*(np.abs((s*t)-xp)**n)+1)
                 
    return (A - A_data)

def busy (xe,xp, b1,b2, a, n, w, c, s, t):
    
    A = (a/4.)* (sp.erf(b1*(w+(s*t)-xe))+1) * (sp.erf(b2*(w-(s*t)+xe))+1) * (c*(np.abs((s*t)-xp)**n)+1)

    return A

def busy_data_2 (params, t, A_data):
    
    # xe = params['xe'].value
    x0 = params['xp'].value
    b = params['b1'].value
    # b2 = params['b2'].value
    a = params['a'].value
    # n = params['n'].value
    # w = params['w'].value
    c = params['c'].value
    s = params['s'].value
    
    
    A = (a/4.)* (sp.erf(b*((s*t)-x0))+1) * (sp.erf(b*(-1*(s*t)+x0))+1) * (c*(np.abs((s*t)-x0)**6)+1)
                 
    return (A - A_data)

def busy_2 (x0, b, a, c, s, t):
    
    A = (a/4.)* (sp.erf(b*((s*t)-x0))+1) * (sp.erf(b*(-1*(s*t)+x0))+1) * (c*(np.abs((s*t)-x0)**6)+1)

    return A


def busy_PSPL_data (params, t, A_data):
    
    t0 = params['t0'].value
    tE = params['tE'].value
    u0 = params['u0'].value
    fs = params['fs'].value
    xe = params['xe'].value
    xp = params['xp'].value
    b1 = params['b1'].value
    b2 = params['b2'].value
    a = params['a'].value
    n = params['n'].value
    w = params['w'].value
    c = params['c'].value
    s = params['s'].value
    
    
    F = busy (xe,xp, b1,b2, a, n, w, c, s, t)+PSPL (t0, tE, u0,fs, t)
                 
    return (F - A_data)

def busy_PSPL (t0, tE, u0,fs, xe,xp, b1,b2, a, n, w, c, s, t):
    
    F = busy (xe,xp, b1,b2, a, n, w, c, s, t)+PSPL (t0, tE, u0,fs, t)

    return F

def busy_PSPL_data_2 (params, t, A_data):
    
    t0 = params['t0'].value
    tE = params['tE'].value
    u0 = params['u0'].value
    fs = params['fs'].value
    x0 = params['x0'].value
    # xp = params['xp'].value
    # b1 = params['b1'].value
    b = params['b'].value
    a = params['a'].value
    # n = params['n'].value
    # w = params['w'].value
    c = params['c'].value
    s = params['s'].value
    
    
    F = busy_2 (x0, b, a, c, s, t)+PSPL (t0, tE, u0,fs, t)
                 
    return (F - A_data)

def busy_PSPL_2 (t0, tE, u0,fs, x0, b, a, c, s, t):
    
    F = busy_2 (x0, b, a,c, s, t)+PSPL (t0, tE, u0,fs, t)

    return F

def erfs (xe, b1,b2, a, w, s, t):
    A = (a/4.)* (sp.erf(b1*(w+(s*t)-xe))+1) * (sp.erf(b2*(w-(s*t)+xe))+1)
    
    return A

def Erfs_PSPL_data (params, t, A_data):
    
    t0 = params['t0'].value
    tE = params['tE'].value
    u0 = params['u0'].value
    fs = params['fs'].value
    xe = params['xe'].value
    xp = params['xp'].value
    b1 = params['b1'].value
    b2 = params['b2'].value
    a = params['a'].value
    n = params['n'].value
    w = params['w'].value
    c = params['c'].value
    s = params['s'].value
    
    
    F = erfs (xe, b1,b2, a, w, s, t)+PSPL (t0, tE, u0,fs, t)
                 
    return (F - A_data)

def Erfs_PSPL (t0, tE, u0,fs, xe, b1,b2, a, w, s, t):
    
    F = erfs (xe, b1,b2, a, w, s, t)+PSPL (t0, tE, u0,fs, t)

    return F
    

def lnlike_busy(theta, t, f, f_err):
    xe,xp, b1,b2, a, n, w, c, s= theta
    model = busy (xe,xp, b1,b2, a, n, w, c, s, t)
    inv_sigma2 = 1.0/(f_err**2)
    return -0.5*(np.sum((f-model)**2*inv_sigma2))

def lnlike_busy_2(theta, t, f, f_err):
    x0, b, a, c, s= theta
    model = busy_2 (x0, b, a,c, s, t)
    inv_sigma2 = 1.0/(f_err**2)
    check = a*c
    check = 0 if check>0 else 1e25
#     print(-1*(-0.5*(np.sum((f-model)**2*inv_sigma2))-check))
    return -0.5*(np.sum((f-model)**2*inv_sigma2))#-check

def lnlike_busy_PSPL_2(theta, t, f, f_err):
    t0, tE, u0,fs, x0, b, a, c, s = theta
    model = busy_PSPL_2 (t0, tE, u0,fs, x0, b, a, c, s, t)
    inv_sigma2 = 1.0/(f_err**2)
    check = a*c
    check = 0 if check>0 else 1e25
    return -0.5*(np.sum((f-model)**2*inv_sigma2))#-check

def lnlike_busy_PSPL(theta, t, f, f_err):
    t0, tE, u0,fs, xe,xp, b1,b2, a, n, w, c, s, t = theta
    model = busy_PSPL (t0, tE, u0,fs, xe,xp, b1,b2, a, n, w, c, s, t)
    inv_sigma2 = 1.0/(f_err**2)
    return -0.5*(np.sum((f-model)**2*inv_sigma2))

def lnlike_erfs(theta, t, f, f_err):
    xe, b1,b2, a, w, s = theta
    model = erfs (xe, b1,b2, a, w, s, t)
    inv_sigma2 = 1.0/(f_err**2)
    return -0.5*(np.sum((f-model)**2*inv_sigma2))
    



def PSPL_Gaussian (t0,tE, u0,fs,tp, tEp,amp, t):
#     u = np.sqrt(u0**2+((t-t0)/tE)**2)
    F = PSPL (t0, tE, u0,fs, t) + Gaussian (tp, tEp,amp, t)
    #= (((amp/np.sqrt(2*pi*(sigma**2)))*np.exp(-((t-mean)**2)/(2*(sigma**2)))))+((u**2)+2)/(u*np.sqrt((u**2)+4))
    #F = (fs * (A-1)) +1
    return F

def PSPL_Gaussian_data (params, t, A_data):
    
    t0 = params['t0'].value
    tE = params['tE'].value
    u0 = params['u0'].value
    tp = params['tp'].value
    tEp = params['tEp'].value
    amp = params['amp'].value
    fs = params['fs'].value
#     fb = params['fb'].value
    
    u = np.sqrt(u0**2+((t-t0)/tE)**2)
    #A = (((amp/np.sqrt(2*pi*(tEp**2)))*np.exp(-((t-tp)**2)/(2*(tEp**2)))))+((u**2)+2)/(u*np.sqrt((u**2)+4))
    #F = (fs * (A-1)) +1
    F = PSPL (t0, tE, u0,fs, t) + Gaussian (tp, tEp,amp, t)

    return F - A_data

def Gaussian (tp, tEp,amp, t):
    A = amp*np.exp(-1*((t-tp)**2)/(2*(tEp**2)))
    #(((amp/np.sqrt(2*pi*(sigma**2)))
#     F = (fs * (A-1)) +1
    return A

def Gaussian_data (params, t, A_data):
    

    tp = params['tp'].value
    tEp = params['tEp'].value
    amp = params['amp'].value
#     fs = params['fs'].value
#     fb = params['fb'].value
    
    A = amp*np.exp(-1*((t-tp)**2)/(2*(tEp**2)))
    #(((amp/np.sqrt(2*pi*(sigma**2)))

    return A- A_data
    
def trapezoid(x, a, b, tau1, tau2, tau3, tau4):
    # a and c are slopes
    #tau1 and tau2 mark the beginning and end of the flat top
#     y = np.zeros(len(x))
#     c = -np.abs(c)
#     a = np.abs(a)
#     #(tau1,tau2) = (min(tau1,tau2),max(tau1,tau2))
#     y[:int(tau1)] = base
#     y[int(tau1):int(tau2)] =  a*x[:int(tau1)] + b
#     y[int(tau2):int(tau3)] =  a*tau1 + b 
#     y[int(tau2):int(tau4)] = c*(x[int(tau2):]-tau2) + (a*tau1 + b)
#     y[int(tau4):] = base

    y = np.zeros(len(x))
    df_trap = pd.DataFrame({'x': x, 'y': y})
    
    c1 = np.abs((b-a)/(tau2-tau1))
    c2 = -1 * np.abs((a-b)/(tau4-tau3))
    
    df_trap['y'][df_trap['x']<tau1] = a
    df_trap['y'][(df_trap['x']>tau1) & (df_trap['x']<tau2)] =  c1*df_trap['x'][(df_trap['x']>tau1) & (df_trap['x']<tau2)] + (a- c1 * tau1)
    df_trap['y'][(df_trap['x']>tau2) & (df_trap['x']<tau3)] =  b
    df_trap['y'][(df_trap['x']>tau3) & (df_trap['x']<tau4)] = c2*df_trap['x'][(df_trap['x']>tau3) & (df_trap['x']<tau4)] + (a- c2 * tau4)
    df_trap['y'][df_trap['x']>tau4] = a

    return df_trap['y']

def trapezoid2(x, amp, t0, del_tau1, del_tau2):
    # a and c are slopes
    #tau1 and tau2 mark the beginning and end of the flat top
#     y = np.zeros(len(x))
#     c = -np.abs(c)
#     a = np.abs(a)
#     #(tau1,tau2) = (min(tau1,tau2),max(tau1,tau2))
#     y[:int(tau1)] = base
#     y[int(tau1):int(tau2)] =  a*x[:int(tau1)] + b
#     y[int(tau2):int(tau3)] =  a*tau1 + b 
#     y[int(tau2):int(tau4)] = c*(x[int(tau2):]-tau2) + (a*tau1 + b)
#     y[int(tau4):] = base

    y = np.zeros(len(x))
    df_trap = pd.DataFrame({'x': x, 'y': y})
    
    c1 = np.abs((amp)/(del_tau2))
    c2 = -1 * c1

    tau1 = t0 - del_tau1 - del_tau2/2.
    tau2 = t0 - del_tau1 + del_tau2/2.
    tau3 = t0 + del_tau1 - del_tau2/2.
    tau4 = t0 + del_tau1 + del_tau2/2.
    
    df_trap['y'][(df_trap['x'] < tau1)] = 0
    df_trap['y'][(df_trap['x'] >= tau1) & (df_trap['x'] <= tau2)] =  c1*df_trap['x'][(df_trap['x']>tau1) &\
                                                                 (df_trap['x']<tau2)] + (-1*c1 * tau1)
    df_trap['y'][(df_trap['x'] > tau2) & (df_trap['x'] < tau3)] =  amp
    df_trap['y'][(df_trap['x'] >= tau3) & (df_trap['x'] <= tau4)] = c2*df_trap['x'][(df_trap['x']>tau3) & (df_trap['x']<tau4)] + (-1*c2 * tau4)
    df_trap['y'][df_trap['x'] > tau4] = 0

    return df_trap['y']

def trapezoid2_data(params,x, A_data):

    amp = params['amp'].value
    t0 = params['t0'].value
    del_tau1 = params['del_tau1'].value
    del_tau2 = params['del_tau2'].value

    return trapezoid2(x, amp, t0, del_tau1, del_tau2)-A_data
    
def med_med (true,fitted):
    temp = fitted - true
    return (np.median(np.abs(temp-np.median(temp))))

def deviation_finder (t, A_residual, PSPL_params,  binsize_initial = 700, threshold_default = 3):
        
    std_base = np.std(A_residual[(t > PSPL_params[0]+10*PSPL_params[1]) | (t < PSPL_params[0]-10*PSPL_params[1]) ])
    std_all = np.std(A_residual)
    percent_diff = (np.abs(std_base-std_all)/float(std_all))*100
    
    if percent_diff < 50:
        smoothie ='yes'
    else:
        smoothie ='no'
        
    b_s = binsize_initial
    
    n_out, temp_peaks = count_peaks (t, A_residual, smooth=smoothie, bin_size =b_s, threshold = threshold_default)

    n_peaks = len(temp_peaks)
    
    c = len(temp_peaks)
    
    if (c != 2):
        while c != 2 :
            if b_s < 0.5:
                break

            b_s = b_s/2.
            n_out, peaks = count_peaks (t, A_residual, smooth=smoothie, bin_size =b_s, threshold = 3)
            c = len(peaks)

    if c == 2 :
        if (np.abs(peaks[0]-peaks[1])>10) or (np.abs(peaks[1]-peaks[0]) <0.5):
            peaks = [temp_peaks[0]]
            n_peaks = 1 
    


    return n_peaks, peaks

