import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from numpy import *
import pandas as pd
from scipy.interpolate import interp1d
import scipy.optimize as opt
from lmfit import minimize, Parameters, Parameter, report_fit
from Common_functions import *
from PSPL_busy_fit import PSPL_busy_fit
import time
from scipy.signal import find_peaks, peak_widths, peak_prominences



class lc2():
	'''Astronomical source object for NOAO formatted light curve
	
	Parameters
	----------
	filename : str
		path to space delimited file including each time, 
		magnitude, and magnitude_uncertainty measurement 
		on each line. 
	
	Attributes
	----------
	filename : str
		the filename parameter
	
	df : pandas DataFrame
		light curve data read into a pandas dataframe
	
	'''
	
	def __init__(self, t, m, e, name, y_type, delta_t = 50, up_lim=100):

		

		self.y_type = y_type
		self.up_lim = up_lim
		self.name = name
		self.t = t
		self.m = m
		if np.sum(~np.isnan(e)) == 0:
			self.e = np.ones(len(self.m))
		else:
			self.e = e

		if delta_t == None:
			self.delta_t = max(self.t)/2
		else:
			self.delta_t = delta_t

		if self.t.min() > 2458234:
				self.t = self.t - 2458234
		else: 
				self.t = self.t

		if self.y_type == 'magnitude':

			self.t_max = self.t[(self.m == np.percentile(self.m, 100 - self.up_lim))][0]
			it0 = np.where(self.t == self.t_max)[0][0]
			idx1 = np.where(self.t >= self.t_max-self.delta_t)[0]
			idx2 = np.where(self.t <= self.t_max+self.delta_t)[0]
			event = list(set(idx1).intersection(set(idx2)))
			
			baseline = np.arange(0,len(self.t),1)
			baseline = np.delete(baseline,event)
			
			
			

			self.event = event
			self.baseline = baseline
			self.t_idx = it0


			df = pd.DataFrame({'t': self.t, 'A': np.zeros(len(self.t)), 'A_err': np.zeros(len(self.t))})
			
			base_mag = np.median(self.m[self.baseline])
			df['A'] = 10 ** (0.4*(base_mag - self.m))
			

			A_max = 10 ** (0.4*(base_mag - (self.m-self.e)))
			A_min = 10 ** (0.4*(base_mag - (self.m+self.e)))
			df['A_err'] = (A_max - A_min)/2
			
			self.df = df

			

		elif y_type == 'magnification':

			df = pd.DataFrame({'t': self.t, 'A': self.m, 'A_err': self.e})

			self.t_max = self.t[(self.m == np.percentile(self.m, self.up_lim))][0]
			idx1 = np.where(self.t >= self.t_max-self.delta_t)[0]
			idx2 = np.where(self.t <= self.t_max+self.delta_t)[0]
			event = list(set(idx1).intersection(set(idx2)))
			
			baseline = np.arange(0,len(self.t),1)
			baseline = np.delete(baseline,event)
			
			it0 = np.where(self.t == self.t_max)[0][0]
			

			self.event = event
			self.baseline = baseline
			self.t_idx = it0
			self.df = df

		self.PSPL_fit_status = False
		self.peaks_indices, _ = find_peaks(self.df.A.values)
		self.peaks_width = peak_widths(self.df.A.values, self.peaks_indices)[0]
		self.peaks_height = self.df.A[self.peaks_indices]
		self.peaks_time = self.df.t[self.peaks_indices]

		if len(self.peaks_indices)>1:
			self.peak_main_index = self.peaks_indices[np.argmax(self.peaks_width)]
			self.A_max = self.df.A[self.peak_main_index]
			self.t_max = self.df.t[self.peak_main_index]
			self.tE_true = max(self.peaks_width)/4.

		elif len(self.peaks_indices) == 1:
			self.peak_main_index = self.peaks_indices
			self.A_max = self.df.A[self.peak_main_index].values[0]
			self.t_max = self.df.t[self.peak_main_index].values[0]
			self.tE_true = self.peaks_width[0]/4.

		else:
			print('No peaks were found!')
			sys.exit()

		self.t0_true = self.t_max
		self.fs_true = 0.5
		self.tE_ini = [0.01, 0.1, 1.0, 10.0, 100]
		self.u0_true = np.sqrt( ( ( 1 + np.sqrt( 1 + 16 *( self.A_max ** 2 )))/( 2 * self.A_max ) ) - 2 )
		self.top_interval = 1 * self.tE_true * self.u0_true


	def PSPL_fitter_scipy (self):



		nll = lambda *args: -lnlike(*args)
		res_scipy = op.minimize(nll, [self.t0_true, self.tE_true, self.u0_true, 0.5],
								 args=(self.df.t[self.df.A <= self.A_max].values,
                              	 	   self.df.A[self.df.A <= self.A_max].values,
                             	 	   self.df.A_err[self.df.A <= self.A_max].values), method = 'Nelder-Mead')

		t0_ml, tE_ml, u0_ml,fs_ml = res_scipy['x']
		self.PSPL_params = [t0_ml, tE_ml, u0_ml,fs_ml]

		self.temp_model = PSPL(self.PSPL_params[0],self.PSPL_params[1],self.PSPL_params[2],self.PSPL_params[3] , self.df.t.values[self.df.A <= self.A_max])
		
		

		A_top = self.df.A[self.df.A <= self.A_max][(self.df.t.values[self.df.A <= self.A_max] < t0_ml + self.top_interval) & (self.df.t.values[self.df.A <= self.A_max] > t0_ml - self.top_interval)]
		
		while len(A_top) < 5:
			self.top_interval = 2* self.top_interval
			A_top = self.df.A[self.df.A <= self.A_max][(self.df.t.values[self.df.A <= self.A_max] < t0_ml + self.top_interval) & (self.df.t.values[self.df.A <= self.A_max] > t0_ml - self.top_interval)]


		model_top = self.temp_model[(self.df.t.values[self.df.A <= self.A_max] < t0_ml + self.top_interval) & (self.df.t.values[self.df.A <= self.A_max] > t0_ml - self.top_interval)]
		A_err_top = self.df.A_err[self.df.A <= self.A_max][(self.df.t.values[self.df.A <= self.A_max] < t0_ml + self.top_interval) & (self.df.t.values[self.df.A <= self.A_max] > t0_ml - self.top_interval)]


		self.PSPL_chisqr = cal_chisqr(self.temp_model, self.df.A.values[self.df.A <= self.A_max], self.df.A_err.values[self.df.A <= self.A_max])
		self.PSPL_chisqr_reduced = self.PSPL_chisqr/(len(self.df.t.values[self.df.A <= self.A_max])-4)
		self.PSPL_chisqr_top = cal_chisqr(model_top, A_top.values, A_err_top.values)
		self.PSPL_chisqr_top_reduced = self.PSPL_chisqr_top/(len(model_top)-4)
		self.top_number = len(model_top)

		self.PSPL_model = np.zeros(len(self.df.t))
		self.PSPL_model[self.df.A > self.A_max] = np.nan
		self.PSPL_model[self.df.A <= self.A_max] = self.temp_model

		self.PSPL_fit_status = True
	
	def Cauchy_fitter(self):

		nll = lambda *args: -lnlike_cauchy(*args)
		res_scipy = op.minimize(nll, [self.t0_true, self.tE_true/2., 1, self.A_max],
								 args=(self.df.t[self.df.A <= self.A_max].values,
                              	 	   self.df.A[self.df.A <= self.A_max].values,
                             	 	   self.df.A_err[self.df.A <= self.A_max].values), method = 'Nelder-Mead')

		t0_ml, tE_ml, b_ml, Amax_ml = res_scipy['x']
		self.Cauchy_params = [t0_ml, tE_ml, b_ml, Amax_ml]
		self.temp_model2 = bell_curve(self.df.t[self.df.A <= self.A_max].values,
									   t0_ml, tE_ml, b_ml, Amax_ml)



		A_top = self.df['A'][self.df.A <= self.A_max][(self.df.t.values[self.df.A <= self.A_max]
													   < self.Cauchy_params[0] + self.top_interval)
													    & (self.df.t.values[self.df.A <= self.A_max] 
													   > self.Cauchy_params[0] - self.top_interval)]

		model_top = self.temp_model2[(self.df.t.values[self.df.A <= self.A_max] < self.Cauchy_params[0] + self.top_interval) 
									& (self.df.t.values[self.df.A <= self.A_max] > self.Cauchy_params[0] - self.top_interval)]

		A_err_top = self.df['A_err'][self.df.A <= self.A_max][(self.df.t.values[self.df.A <= self.A_max] < self.Cauchy_params[0] + self.top_interval) 
															& (self.df.t.values[self.df.A <= self.A_max] > self.Cauchy_params[0] - self.top_interval)]
		
		self.chisqr_Cauchy = cal_chisqr(self.temp_model2, self.df['A'].values[self.df.A <= self.A_max],
										self.df['A_err'].values[self.df.A <= self.A_max])
		self.chisqr_Cauchy_reduced = self.chisqr_Cauchy/(len(self.df.t[self.df.A <= self.A_max])-4)
		self.chisqr_Cauchy_top = cal_chisqr(model_top, A_top.values, A_err_top.values)
		self.chisqr_Cauchy_top_reduced = self.chisqr_Cauchy_top/(len(model_top)-4)


		self.b_Cauchy = self.Cauchy_params[2]
		if self.PSPL_fit_status:
			self.psi = self.PSPL_chisqr_top_reduced-self.chisqr_Cauchy_top_reduced
			self.delta_chisqr_total = self.PSPL_chisqr - self.chisqr_Cauchy
		else:
			self.psi = np.nan
			self.delta_chisqr_total = np.nan
		
		self.Cauchy_model = np.zeros(len(self.df.t))
		self.Cauchy_model[self.df.A > self.A_max] = np.nan
		self.Cauchy_model[self.df.A <= self.A_max] = self.temp_model2


	
	def Chebyhev_fitter (self, degree):
		self.n = degree

		if self.n <11:
			print('Degree must be more than 10.')
			sys.exit()
		self.xmin = min(self.df['t'][self.event])
		self.xmax = max(self.df['t'][self.event])
		bma = 0.5 * (self.xmax - self.xmin)
		bpa = 0.5 * (self.xmax + self.xmin)
		interpoll = interp1d(self.df['t'],self.df['A'], kind='cubic')
		f = [interpoll(math.cos(math.pi * (k + 0.5) / self.n) * bma + bpa) for k in range(self.n)]
		fac = 2.0 / self.n
		self.cheby_coefficients = [fac * sum([f[k] * math.cos(math.pi * j * (k + 0.5) / self.n) for k in range(self.n)]) for j in range(self.n)]


		self.Cheby_func = []

		for t_i in np.sort(self.df['t'][self.event].values):

			y = (2.0 * t_i - self.xmin - self.xmax) * (1.0 / (self.xmax - self.xmin))
			y2 = 2.0 * y
			(d, dd) = (self.cheby_coefficients[-1], 0)             # Special case first step for efficiency
			
			for cj in self.cheby_coefficients[-2:0:-1]:            # Clenshaw's recurrence
				(d, dd) = (y2 * d - dd + cj, d)
			self.Cheby_func.append(y * d - dd + 0.5 * self.cheby_coefficients[0])

		self.Cheby_func = np.asarray(self.Cheby_func)

			
		self.Cheby_a0 = (self.cheby_coefficients[0])/(self.cheby_coefficients[0])
		self.Cheby_a2 = (self.cheby_coefficients[2])/(self.cheby_coefficients[0])
		self.Cheby_a4 = (self.cheby_coefficients[4])/(self.cheby_coefficients[0])
		self.Cheby_a6 = (self.cheby_coefficients[6])/(self.cheby_coefficients[0])
		self.Cheby_a8 = (self.cheby_coefficients[8])/(self.cheby_coefficients[0])
		self.Cheby_a10 = (self.cheby_coefficients[10])/(self.cheby_coefficients[0])


		self.Cheby_cj_sqr = np.sum((np.asarray(self.cheby_coefficients)/(self.cheby_coefficients[0]))**2)
		self.log10_Cheby_cj_sqr_minus_one = np.log10(self.Cheby_cj_sqr - 1)
		self.pos_log10_Cheby_cj_sqr_minus_one = -1*np.log10(self.Cheby_cj_sqr - 1)
		self.delta_A_chebyshev_sqr = np.sum((self.df.A[self.event] - self.Cheby_func)**2)

	def Trapezoidal_fitter(self):

		nll = lambda *args: -lnlike_trap(*args)
		res_scipy = op.minimize(nll, [self.A_max, self.t0_true, self.tE_true, self.tE_true],
								args=(self.df.t[self.df.A <= self.A_max].values,
                              	 	  self.df.A[self.df.A <= self.A_max].values - 1 ,
                             	 	  self.df.A_err[self.df.A <= self.A_max].values), method = 'Nelder-Mead')

		Amax_ml, t0_ml, tE1_ml, tE2_ml = res_scipy['x']
		self.trap_params = [Amax_ml, t0_ml, tE1_ml, tE2_ml]
		self.temp_model3 = trapezoid2(self.df.t[self.df.A <= self.A_max].values,
									   Amax_ml, t0_ml, tE1_ml, tE2_ml) + 1
		self.t0_trap = self.trap_params[1]
		self.max_trap = self.trap_params[0]
		self.tau1 = self.trap_params[1] - self.trap_params[2] - self.trap_params[3]/2.
		self.tau2 = self.trap_params[1] - self.trap_params[2] + self.trap_params[3]/2.
		self.tau3 = self.trap_params[1] + self.trap_params[2] - self.trap_params[3]/2.
		self.tau4 = self.trap_params[1] + self.trap_params[2] + self.trap_params[3]/2.
		self.tE_trap_total = self.tau4 - self.tau1
		self.tE_trap_flat_part = self.tau3 - self.tau2
		self.tE_trap_ratio = self.tE_trap_flat_part/self.tE_trap_total
		self.magnification_median = np.median(self.df['A'][self.baseline])
		self.chisqr_trap = cal_chisqr(self.df['A'][self.df.A <= self.A_max],
									  self.temp_model3,
									  self.df['A_err'][self.df.A <= self.A_max])
		self.chisqr_trap_reduced = self.chisqr_trap/ (len(self.df.t[self.df.A <= self.A_max])-4)

		self.trap_model = np.zeros(len(self.df.t))
		self.trap_model[self.df.A > self.A_max] = np.nan
		self.trap_model[self.df.A <= self.A_max] = self.temp_model3


	def PSPL_busy_fitter(self, direc, verbose=False):

			plfit = PSPL_busy_fit(direc+self.name, up_lim=99.9999)
			# plfit.PSPL_fitter_scipy()
			t_start = time.time()
			plfit.PSPL_residual(self.PSPL_params, verbose=verbose)
			t_end = time.time()
			self.PSPL_fit_time =  t_end - t_start
			self.peaks = plfit.peaks
			self.peaks_A = plfit.peaks_A
			self.troughs = plfit.troughs
			self.troughs_A = plfit.troughs_A
			self.df['A_residual'] = plfit.df['A_residual'] 
			plfit.dev_analyze()
			self.dev_counter = plfit.dev_counter
			self.dev_A = plfit.dev_A
			self.dev_t = plfit.dev_t
			self.perturbation_type = plfit.perturbation_type
			if plfit.dev_counter != 0:
				t_start = time.time()
				plfit.busy_fitter_scipy(verbose=verbose)
				t_end = time.time()
				self.busy_fit_time =  t_end - t_start
				self.x0_init = plfit.x0_init
				self.b_init = plfit.b_init
				self.a_init = plfit.a_init
				self.c_init = plfit.c_init
				self.s_init = plfit.s_init
				t_start = time.time()
				plfit.PSPL_busy_fitter_scipy(verbose=verbose)
				t_end = time.time()
				self.PSPL_busy_fit_time =  t_end - t_start
				plfit.calculate_s_q_2(verbose=verbose)
			else:
				self.busy_fit_time = np.nan
				self.PSPL_busy_fit_time = np.nan
				self.x0_init = np.nan
				self.b_init = np.nan
				self.a_init = np.nan
				self.c_init = np.nan
				self.s_init = np.nan




			self.busy_chisqr = plfit.busy_chisqr
			self.busy_params = plfit.busy_params
			self.PSPL_busy_chisqr = plfit.PSPL_busy_chisqr
			self.PSPL_busy_params = plfit.PSPL_busy_params
			self.PSPL_busy_model = plfit.PSPL_busy_model
			self.busy_model = plfit.busy_model
			self.PSPL_residual_peaks = plfit.peaks
			self.PSPL_residual_peaks_A = plfit.peaks_A
			self.PSPL_residual_troughs = plfit.troughs
			self.PSPL_residual_troughs_A = plfit.troughs_A
			
			self.q = plfit.q
			self.s = plfit.s
			self.tEp = plfit.tEp
			self.tp = plfit.tp
			



	# def plot(self):

	# 	# self.pd_maker()

	# 	'''Plot the 4 band light curve'''
	# 	fig, axs = plt.subplots(figsize=(15, 12))

	# 	if self.t.min() > 2458234:
	# 			t2 = self.t - 2458234
	# 	else: 
	# 			t2 = self.t

	# 	plt.plot(t2[self.event], self.df['A'][self.event], '.', color='gray', markersize=20)
	# 	plt.ylabel('Magnification', size=25)
	# 	plt.xlabel('Time - 2458234', size=25)
	# 	# plt.xlim(self.t_idx - 2 * self.tE, self.t_idx + 2 * self.tE)
	# 	# plt.legend(loc=2 , fontsize=20)

	# 	fig.tight_layout()