import numpy as np
import pickle as pkl
import pandas as pd
from tqdm import tqdm



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
        
        