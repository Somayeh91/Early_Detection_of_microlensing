import numpy as np
import pickle as pkl
import pandas as pd
from tqdm import tqdm
import scipy.optimize as op
from Common_functions import lnlike



def read_data(path = 'all_data_muLens_larger_than_3sigma.pkl', 
              n_points = 5):
    all_data = pkl.load(open(path, 'rb'))
    filters = ['r ', 'i ']
    n_filters = len(filters)
    labels = []
    data = []
    c = 0
    
    for k, key in tqdm(enumerate(all_data.keys())):
        for key2 in all_data[key].keys():
            tmp = np.ones((n_points, n_filters*3))*np.nan
            for b, band in enumerate(filters):
               
                try: 
                    df = pd.DataFrame(data = all_data[key][key2], 
                                     columns = ['t', 'f', 'ferr', 'b'])
                    if len(df['f'][df['b'] == band].values)<n_points:
                        tmp_n_point = len(df['f'][df['b'] == band].values)
                        tmp[:tmp_n_point, 3*b] = df['t'][df['b'] == band].values
                        tmp[:tmp_n_point, 1+3*b] = df['f'][df['b'] == band].values
                        tmp[:tmp_n_point, 2+3*b] = df['ferr'][df['b'] == band].values
                    else:
                        tmp[:, 3*b] = df['t'][df['b'] == band].values[:n_points]
                        tmp[:, (3*b)+1] = df['f'][df['b'] == band].values[:n_points]
                        tmp[:, (3*b)+2] = df['ferr'][df['b'] == band].values[:n_points]
                    
                except:
                    print(key, key2, all_data[key][key2])
                    
            
                    
            data.append(tmp)
            labels.append(k)
                
            
            c += 1
    del_inds = []
    for d in range(len(data)):
        if np.sum(np.isnan(data[d]))>0:
            del_inds.append(d)
        
        
    data = np.delete(data, del_inds, axis=0)
    labels = np.delete(labels, del_inds, axis=0)
            
    return data, labels



def select_points_roman(path = '/Users/somayeh/Library/Mobile Documents/com~apple~CloudDocs/Research/Microlensing_Harvard/',
                        n_days = 5, 
                        thresh_mag = 1.1,
                        cadence = 15/(60*24)):
    
    data = np.load(path+'alllc_as_input.npy')
    labels = data[:,-1]
    IDs = data[:,0]
    data = data[:,1:-2]
    
    mjd_t = np.arange(len(data[0])) * cadence
    n_points = int(n_days/cadence)
    
    
    new_data = []
    new_labels = []
    
    for d, dat in tqdm(enumerate(data)):
        if len(dat[dat>thresh_mag])<n_points:
            continue
        
        new_data.append(dat[dat>thresh_mag][:n_points])
        new_labels.append(labels[d])
        
    return np.asarray(new_data), np.asarray(new_labels), IDs
    
def fit_PSPL(mjd_t, data):
    data_exp = data
    t0_guess = mjd_t[np.argmax(data_exp)]
    tE_guess = [1, 20]
    u0_true = 1./np.max(data_exp)
    blending = 0.5

    merr = np.ones(len(data_exp))*0.001

    all_fit_res = []

    nll = lambda *args: -lnlike(*args)
    fun_ = np.inf
    for tE in tE_guess:
        res_scipy = op.minimize(nll, [t0_guess, tE, u0_true, blending],
                                args=(mjd_t,
                                     data_exp,
                                     merr), method = 'Nelder-Mead')
        if res_scipy['fun']<fun_:
            fun_ = res_scipy['fun']
            all_fit_res = res_scipy
            
    return all_fit_res, fun_
    
    
        
def lnlike(theta, t, f, f_err):
    t0, tE, u0, fs = theta
    model = fun(t, t0, tE, u0, fs)
    inv_sigma2 = 1.0/(f_err**2)
    return -0.5*(np.sum((f-model)**2*inv_sigma2))

def fun (t,t0,tE, u0, fs):
    u = np.sqrt(u0**2+((t-t0)/tE)**2)
    A = ((u**2)+2)/(u*np.sqrt(u**2+4))
    F = fs*A +(1-fs)
    return F