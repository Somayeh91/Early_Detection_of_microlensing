import numpy as np
import pickle as pkl
import pandas as pd
from tqdm import tqdm
import scipy.optimize as op
from Common_functions import lnlike
import matplotlib.cm as cm
from scipy.interpolate import interp1d
import matplotlib.colors as mcolors
import math

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



def select_points_roman(path = '/Users/somayeh/Library/Mobile Documents/com~apple~CloudDocs/Research/Microlensing/Machine_Learning/Microlensing_Harvard/',
                        n_days = 5, 
                        thresh_mag = 0.1,
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
        new_dat = (dat-min(dat))/(max(dat)-min(dat))
        if len(new_dat[new_dat>thresh_mag])<n_points:
            continue
        
        new_data.append(new_dat[new_dat>thresh_mag][:n_points])
        new_labels.append(labels[d])
        
    return np.asarray(new_data), np.asarray(new_labels), IDs
    
def fit_PSPL(mjd_t, data, merr=np.nan):
    data_exp = data
    t0_guess = mjd_t[np.argmax(data_exp)]
    tE_guess = [1, 20]
    u0_true = 1./np.max(data_exp)
    blending = 0.5

    if not hasattr(merr, "__len__"):
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



def colorbar_index(ncolors, cmap, label):
    cmap = cmap_discretize(cmap, ncolors)
    mappable = cm.ScalarMappable(cmap=cmap)
    mappable.set_array([])
    mappable.set_clim(0.5, ncolors+0.5)
    colorbar = plt.colorbar(mappable)
    colorbar.set_ticks(np.linspace(1, ncolors, ncolors))
    colorbar.set_ticklabels(np.array([0, 1])) #range(ncolors)
    colorbar.set_label(label=label, size=20, weight='bold')
    colorbar.ax.tick_params(labelsize=20)
    return cmap

def cmap_discretize(cmap, N):
    """Return a discrete colormap from the continuous colormap cmap.

        cmap: colormap instance, eg. cm.jet.
        N: number of colors.

    Example
        x = resize(arange(100), (5,100))
        djet = cmap_discretize(cm.jet, 5)
        imshow(x, cmap=djet)
    """

    if type(cmap) == str:
        cmap = plt.get_cmap(cmap)
    colors_i = np.concatenate((np.linspace(0, 1., N), (0.,0.,0.,0.)))
    colors_rgba = cmap(colors_i)
    indices = np.linspace(0, 1., N+1)
    cdict = {}
    for ki,key in enumerate(('red','green','blue')):
        cdict[key] = [ (indices[i], colors_rgba[i-1,ki], colors_rgba[i,ki])
                       for i in range(N+1) ]
    # Return colormap object.
    return mcolors.LinearSegmentedColormap(cmap.name + "_%d"%N, cdict, 1024)

def fit_Cheby(t, y, degree=50):
    n =degree

    y_values = y
    mjd_t = t

    if n <11:
        print('Degree must be more than 10.')
        sys.exit()
    xmin = min(mjd_t)
    xmax = max(mjd_t)
    bma = 0.5 * (xmax - xmin)
    bpa = 0.5 * (xmax + xmin)
    interpoll = interp1d(mjd_t, y_values, kind='cubic')
    f = [interpoll(math.cos(math.pi * (k + 0.5) / n) * bma + bpa) for k in range(n)]
    fac = 2.0 / n
    cheby_coefficients = [fac * sum([f[k] * math.cos(math.pi * j * (k + 0.5) / n) for k in range(n)]) for j in range(n)]


    Cheby_all = {}
    Cheby_func = []

    for t_i in np.sort(mjd_t):

        y = (2.0 * t_i - xmin - xmax) * (1.0 / (xmax - xmin))
        y2 = 2.0 * y
        (d, dd) = (cheby_coefficients[-1], 0)             # Special case first step for efficiency

        for cj in cheby_coefficients[-2:0:-1]:            # Clenshaw's recurrence
            (d, dd) = (y2 * d - dd + cj, d)
        Cheby_func.append(y * d - dd + 0.5 * cheby_coefficients[0])

    Cheby_all['y_fitted'] = np.asarray(Cheby_func)

    Cheby_all['all_coeffs'] = cheby_coefficients
    Cheby_all['Cheby_a0'] = (cheby_coefficients[0])/(cheby_coefficients[0])
    Cheby_all['Cheby_a2'] = (cheby_coefficients[2])/(cheby_coefficients[0])
    Cheby_all['Cheby_a4'] = (cheby_coefficients[4])/(cheby_coefficients[0])
    Cheby_all['Cheby_a6'] = (cheby_coefficients[6])/(cheby_coefficients[0])
    Cheby_all['Cheby_a8'] = (cheby_coefficients[8])/(cheby_coefficients[0])
    Cheby_all['Cheby_a10'] = (cheby_coefficients[10])/(cheby_coefficients[0])


    Cheby_all['Cheby_cj_sqr'] = np.sum((np.asarray(cheby_coefficients)/(cheby_coefficients[0]))**2)
    Cheby_all['log10_Cheby_cj_sqr_minus_one'] = np.log10(Cheby_all['Cheby_cj_sqr'] - 1)
    Cheby_all['pos_log10_Cheby_cj_sqr_minus_one'] = -1*np.log10(Cheby_all['Cheby_cj_sqr'] - 1)
    Cheby_all['delta_A_chebyshev_sqr'] = np.sum((y_values - Cheby_func)**2)
    
    return Cheby_all

def get_column_from_dict(all_params, col_name):
    tmp = []
    for i, key in enumerate(list(all_params.keys())):
        tmp.append(all_params[key][col_name])
    return np.asarray(tmp)