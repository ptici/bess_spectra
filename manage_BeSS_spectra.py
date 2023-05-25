'''
authors: Pedro Ticiani and Jonathan Labadie-Bartz
'''

from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as pl 
from os import listdir 
import math
import os
from astropy.convolution import convolve, Box1DKernel
import pyhdust.spectools as spt
from os.path import isfile, join
import time
from astropy.time import Time
from astropy.coordinates.sky_coordinate import SkyCoord
from astroquery.simbad import Simbad
from glob import glob
import xmltodict as _xmltodict
import requests as _requests
import wget as _wget
from scipy.optimize import curve_fit
from scipy.signal import detrend
from astropy.stats import median_absolute_deviation as MAD
import statistics
import seaborn as sns
import math

#Function that Calculate Root Mean Square
def rmsValue(arr, n):
    '''
    '''
    square = 0
    mean = 0.0
    root = 0.0
     
    #Calculate square
    for i in range(0,n):
        square += (arr[i]**2)
     
    #Calculate Mean
    mean = (square / float(n))
     
    #Calculate Root
    root = math.sqrt(mean)
     
    return root

def rmsPE(arr, n, predicted):
    '''
    '''
    square = 0
    mean = 0.0
    root = 0.0
    
    for i in range(0, n):
        square += (arr[i]/predicted - 1)**2
    
    mean = (square / float(n))
    root = math.sqrt(mean)
    
    return root*100

def rmsD(arr, n, predicted):
    '''
    '''
    square = 0
    mean = 0.0
    root = 0.0
    
    for i in range(0, n):
        square += (arr[i] - predicted)**2
    
    mean = (square / float(n))
    root = math.sqrt(mean)
    
    return root
    

def Sliding_Outlier_Removal(x, y, window_size, sigma=3.0, iterate=1):
    '''TODO docstring
    '''
    # remove NANs from the data
    x = x[~np.isnan(y)]
    y = y[~np.isnan(y)]
    #make sure that the arrays are in order according to the x-axis 
    y = y[np.argsort(x)]
    x = x[np.argsort(x)]
    # tells you the difference between the last and first x-value
    x_span = x.max() - x.min()  
    i = 0
    x_final = x
    y_final = y
    while i < iterate:
        i+=1
        x = x_final
        y = y_final
    # empty arrays that I will append not-clipped data points to
    x_good_ = np.array([])
    y_good_ = np.array([])
    # Creates an array with all_entries = True. index where you want to remove outliers are set to False
    tf_ar = np.full((len(x),), True, dtype=bool)
    ar_of_index_of_bad_pts = np.array([]) #not used anymore
    #this is how many days (or rather, whatever units x is in) to slide the window center when finding the outliers
    slide_by = window_size / 5.0 
    #calculates the total number of windows that will be evaluated
    Nbins = int((int(x.max()+1) - int(x.min()))/slide_by)
    for j in range(Nbins+1):
        #find the minimum time in this bin, and the maximum time in this bin
        x_bin_min = x.min()+j*(slide_by)-0.5*window_size
        x_bin_max = x.min()+j*(slide_by)+0.5*window_size
        # gives you just the data points in the window
        x_in_window = x[(x>x_bin_min) & (x<x_bin_max)]
        y_in_window = y[(x>x_bin_min) & (x<x_bin_max)]
        # if there are less than 5 points in the window, do not try to remove outliers.
        if len(y_in_window) > 5:            
            # Removes a linear trend from the y-data that is in the window.
            y_detrended = detrend(y_in_window, type='linear')
            y_in_window = y_detrended
            #print(np.median(m_in_window_))
            y_med = np.median(y_in_window)          
            # finds the Median Absolute Deviation of the y-pts in the window
            y_MAD = MAD(y_in_window)         
            #This mask returns the not-clipped data points. 
            # Maybe it is better to only keep track of the data points that should be clipped...
            mask_a = (y_in_window < y_med+y_MAD*sigma) & (y_in_window > y_med-y_MAD*sigma)
            #print(str(np.sum(mask_a)) + '   :   ' + str(len(m_in_window)))
            y_good = y_in_window[mask_a]
            x_good = x_in_window[mask_a]
            y_bad = y_in_window[~mask_a]
            x_bad = x_in_window[~mask_a]          
            #keep track of the index --IN THE ORIGINAL FULL DATA ARRAY-- of pts to be clipped out
            try:
                clipped_index = np.where([x == z for z in x_bad])[1]
                tf_ar[clipped_index] = False
                ar_of_index_of_bad_pts = np.concatenate([ar_of_index_of_bad_pts, clipped_index])
            except IndexError:
                #print('no data between {0} - {1}'.format(x_in_window.min(), x_in_window.max()))
                pass
    ar_of_index_of_bad_pts = np.unique(ar_of_index_of_bad_pts)
    #print('step {0}: remove {1} points'.format(i, len(ar_of_index_of_bad_pts)))
    x_final = x[tf_ar]
    y_final = y[tf_ar]
    
    return(x_final, y_final)

def LP_fit(vel_a, flx_a, lp_width=250., sample_peak_width=50., test_mode=False, txt_lbl='', outdir='./'):
    #ssize = 0.005

    #gaussfit = True
    #test_mode = True

    # the approximate line width (measured from line center)
    # useful so continuum outliers don't fuck everything up
    # 250 KM/S is good for He 6678. Also seems fine for Hb, Hg, Hd
    #lp_width = 250.

    vels_orig = vel_a
    flux_orig = flx_a

    flux = flux_orig[( (vels_orig > -lp_width) & (vels_orig < lp_width) )]
    vels = vels_orig[( (vels_orig > -lp_width) & (vels_orig < lp_width) )]
    #plt.plot(vels, flux)
    #plt.show()

    # resolution in km/s
    # 6.6496 km/s for DAO.
    pix_width = vel_a[3] - vel_a[2]

    # now lets say I want a total width of 50 km/s centered on the peak...
    #sample_peak_width= 50.0
    #50 km/s = n_pix * pix_width
    #n_pix = (50 km/s) / pix_width

    # this gives the total width in pixels of each peak that should be fit with a gaussian
    n_pix_peak = int(sample_peak_width / pix_width + 1)

    #ssize = int(ssize * len(vels))
    #ssize = 1
    #contmax = np.max(np.append(flux[:ssize], flux[-ssize:]))

    #fluxmax = np.max(flux)

    # gets the index of the velocity value closest to zero??
    # probably to determine the line center (or rather, the corresponding pixel (wavelength flux pair)
    ivc = np.abs(vels - 0).argmin()
    
    smooth_pix_size = int(7.5 / pix_width + 1)
    flux_smoothed_for_peak = convolve(flux, Box1DKernel(smooth_pix_size))
    
    i0 = np.abs(flux_smoothed_for_peak[:ivc] - np.max(flux_smoothed_for_peak[:ivc])).argmin()
    i1 = np.abs(flux_smoothed_for_peak[ivc:] - np.max(flux_smoothed_for_peak[ivc:])).argmin() + ivc
    
    # if the max value is really close the the line center, then try again 
    # (but excluding the inner-most 20 km/s)
    # CAUTION ERROR FIX HELP BUG FIND SOLUTION QUIT 
    # This doesn't exclude the inner-most 20 km/s! Instead, it limits the search to the inner-most 20 km/s!
    if abs(vels[i0]) < 20.0:
        #i0 = np.abs(flux[:ivc - int(20.0/pix_width)] - np.max(flux[:ivc - int(20.0/pix_width)])).argmin()
        i0 = np.abs(flux_smoothed_for_peak[:ivc - int(20.0/pix_width)] - np.max(flux_smoothed_for_peak[:ivc - int(20.0/pix_width)])).argmin()
    if abs(vels[i1] < 20.0):
        #i1 = np.abs(flux[ivc + int(20.0/pix_width):] - np.max(flux[ivc + int(20.0/pix_width):])).argmin() + ivc
        i1 = np.abs(flux_smoothed_for_peak[ivc + int(20.0/pix_width):] - np.max(flux_smoothed_for_peak[ivc + int(20.0/pix_width):])).argmin() + ivc
    
    def gauss(x, *p):
        A, mu, sigma = p
        return A * np.exp(-(x - mu)**2 / (2. * sigma**2)) + 1.

    # inverted gaussian to fit the central depression
    def gauss_inv(x, *p):
        A, mu, sigma, Z = p
        return A * np.exp(-(x - mu)**2 / (2. * sigma**2)) + 1. + Z


    try:
        p0 = [1., vels[i0], 20.]
        
        coeff0, tmp = curve_fit(gauss, vels[i0 - int(n_pix_peak/2):i0 + int(n_pix_peak/2)], flux[i0 - int(n_pix_peak/2):i0 + int(n_pix_peak/2)], p0=p0, bounds=(np.array([0.5, -lp_width, 0.001]),np.array([10.0,0,1.e5])), maxfev=20000)
        #coeff0, tmp = curve_fit(gauss, vels[i0 - int(n_pix_peak/2):i0 + int(n_pix_peak/2)], flux[i0 - int(n_pix_peak/2):i0 + int(n_pix_peak/2)], p0=p0, maxfev=20000)
        
        # fitting hte R peak
        p1 = [1., vels[i1], 20.]
        #coeff1, tmp = curve_fit(gauss, vels[ivc:], flux[ivc:], p0=p1)
        coeff1, tmp = curve_fit(gauss, vels[i1 - int(n_pix_peak/2):i1 + int(n_pix_peak/2)], flux[i1 - int(n_pix_peak/2):i1 + int(n_pix_peak/2)], p0=p1, bounds=(np.array([0.5, 0, 0.001]),np.array([10.0,lp_width,1.e5])), maxfev=20000)
        #coeff1, tmp = curve_fit(gauss, vels[i1 - int(n_pix_peak/2):i1 + int(n_pix_peak/2)], flux[i1 - int(n_pix_peak/2):i1 + int(n_pix_peak/2)], p0=p1, maxfev=20000)


        # fitting the central depression
        p2 = [-0.1, 0.0, 20., 1.1]
        #coeff1, tmp = curve_fit(gauss, vels[ivc:], flux[ivc:], p0=p1)
        #coeff2, tmp = curve_fit(gauss_inv, vels[i0 + int(n_pix_peak/2):i1 - int(n_pix_peak/2)], flux[i0 + int(n_pix_peak/2):i1 - int(n_pix_peak/2)], p0=p2, bounds=(np.array([-10, -lp_width/2, 0.001]),np.array([10.0,lp_width/2,1.e5])), maxfev=20000)
        coeff2, tmp = curve_fit(gauss_inv, vels[i0 + int(n_pix_peak/2):i1 - int(n_pix_peak/2)], flux[i0 + int(n_pix_peak/2):i1 - int(n_pix_peak/2)], p0=p2, maxfev=20000)

        #return coeff0[1], coeff1[1]
        #################################################
        # Relevant parameters from the line profile     #
        #################################################
        V_I = coeff0[0]
        V_RV = coeff0[1]

        R_I = coeff1[0]
        R_RV = coeff1[1]

        #C_I = coeff2[0]
        C_RV = coeff2[1]
        #################################################
        
        if test_mode:
            
            print('V/R = {0:.3f}'.format(V_I/R_I))
            print('PS = {0:.2f}'.format(R_RV - V_RV))
            print('Cent_RV = {0:.2f}'.format(C_RV))
            # dummy x values to plot the best-fit gaussian
            xleft = np.arange(-600, 100, 0.1)
            xright = np.arange(-100, 600, 0.1)
            xmid = np.arange(-200, 200, 0.1)

            # dummy y values to plot the best-fit gaussian
            yleft = coeff0[0] * np.exp(-(xleft - coeff0[1])**2 / (2. * coeff0[2]**2)) #+ 1.
            yright = coeff1[0] * np.exp(-(xright - coeff1[1])**2 / (2. * coeff1[2]**2)) #+ 1.
            ymid = coeff2[0] * np.exp(-(xmid - coeff2[1])**2 / (2. * coeff2[2]**2)) + 1. + coeff2[3]

            plt.plot(vels_orig, flux_orig, color='k')

            plt.scatter(V_RV, V_I + 1, color='k', marker='x', lw=3, s=180, zorder=45)
            plt.scatter(V_RV, V_I + 1, color=color_list[1], marker='x', lw=2, s=140, zorder=50)
            
            plt.scatter(R_RV, R_I + 1, color='k', marker='x', lw=3, s=180, zorder=45)
            plt.scatter(R_RV, R_I + 1, color=color_list[0], marker='x', lw=2, s=140, zorder=50)
            
            plt.scatter(C_RV, coeff2[3] + coeff2[0] + 1, color='k', marker='x', lw=3, s=180, zorder=45)
            plt.scatter(C_RV, coeff2[3] + coeff2[0] + 1, color=color_list[2], marker='x', lw=2, s=140, zorder=50)
            plt.plot(xleft, yleft + 1, color=color_list[1], linestyle='dotted', lw=2)
            plt.plot(xright, yright + 1, color=color_list[0], linestyle='dotted', lw=2)
            plt.plot(xmid, ymid, color=color_list[2], linestyle='dotted', lw=2)
            plt.gca().text(0.03,0.93,txt_lbl,color='k',fontsize=14,transform=plt.gca().transAxes,bbox=dict(facecolor='white',alpha=0.9,edgecolor='none'))
            plt.grid(True, which='both', color='0.5', linestyle='dotted')
            plt.xlabel(r'Vel (km s$^{-1}$)')
            plt.ylabel('Rel. Flux')
            plt.ylim([flux.min() - 0.1*(flux.max() - flux.min()), flux.max() + 0.1*(flux.max() - flux.min())])
            plt.xlim([-1600, 1600])
            plt.gcf().set_size_inches(10,5)
            plt.tight_layout(pad=0.5,w_pad=1,h_pad=0.2)
            
            plt.savefig('{0}/tmp_line_fit_He.png'.format(outdir))
            plt.savefig('{0}/tmp_line_fit_He.pdf'.format(outdir))
            plt.show()
            plt.close('all')
    except:
        V_I, V_RV, R_I, R_RV, C_RV = np.nan, np.nan, np.nan, np.nan, np.nan
    return V_I, V_RV, R_I, R_RV, C_RV

def find_bisectors(vl_ar, flx_ar, vlrange, counter_val = 1.5, emission_frac = 0.5, region_size_flx = 0.3):
    flx_ar = flx_ar[( (vl_ar > -vlrange) & (vl_ar < vlrange) )]
    vl_ar = vl_ar[( (vl_ar > -vlrange) & (vl_ar < vlrange) )]
    max_em_val = np.max(flx_ar)
    bis_vals = []
    bis_flx_vals = []
    bis_V_RVs = []
    bis_R_RVs = []
    #while counter_val < emission_frac * (1.0 + max_em_val):
    while counter_val < (emission_frac * (max_em_val - 1) + 1.0):
        cond_Ha = (( flx_ar > counter_val) & ( flx_ar < counter_val + 1.0*region_size_flx))
        vel_slice = vl_ar[cond_Ha]
        flx_slice = flx_ar[cond_Ha]
        counter_val += region_size_flx
        
        
        bin_slice_V = np.mean(vel_slice[vel_slice < 0])
        bin_slice_R = np.mean(vel_slice[vel_slice > 0])
        bis_slice = np.mean([bin_slice_V, bin_slice_R])
                                        
        #bis_slice = np.mean(vel_slice)
        bis_V_RVs.append(bin_slice_V)
        bis_R_RVs.append(bin_slice_R)
        bis_vals.append(bis_slice)
        bis_flx_vals.append(np.mean(flx_slice))
    
    mean_bis_val = np.mean(np.array(bis_vals)[np.logical_not(np.isnan(bis_vals))])
    EW_V = spt.EWcalc(vl_ar[( (vl_ar < mean_bis_val))] + mean_bis_val, flx_ar[( (vl_ar < mean_bis_val))], vw=vlrange)
    EW_R = spt.EWcalc(vl_ar[( (vl_ar > mean_bis_val))] + mean_bis_val, flx_ar[( (vl_ar > mean_bis_val))], vw=vlrange)
    #print(EW_V/EW_R)
    try:
        #print(EW_V/EW_R)
        return EW_V/EW_R, mean_bis_val
    except ZeroDivisionError:
        #EW_V/EW_R = 0
        return 0, mean_bis_val

def find_mainIDs(TIC):
    result = Simbad.query_objectids(TIC)
    secondID = result[1][0]
    
    return secondID

def starcoord(star_ID):
    star_coord = SkyCoord.from_name(star_ID)
    return star_coord

def automatic_BeSS(RA, DEC, path, size='0.2', date_lower='1000-01-01', date_upper="3000-01-01", band_lower='6.4e-7', band_upper='6.8e-7'):

    user_url = 'http://basebe.obspm.fr/cgi-bin/ssapBE.pl?POS={0},{1}&SIZE={2}&BAND={3}/{4}&TIME={5}/{6}'.format(RA, DEC, size, band_lower, band_upper, date_lower, date_upper)
    
    r = _requests.get(url = user_url)
    ## xml parsed => dict
    global_dict = _xmltodict.parse(r.text)
    try:
    ## Interesting data selection
        entries_list = global_dict['VOTABLE']['RESOURCE']['TABLE']['DATA']['TABLEDATA']['TR']
        for item in entries_list:
            vo_url_fits = item['TD'][0]
            _wget.download(vo_url_fits, out=path)
    except:
        print('Star does not contain published BeSS spectra. Try again later')


'''
start of the script
'''
customSimbad = Simbad()
customSimbad.add_votable_fields('sptype')

## Set equal to True if you want to plot every single full spectrum in a single
## plot. Can be useful if you want to inspect different parts of the spectrum 
## at the same time.

Plot_all_spectra = True
quiet = False

## Sets size of axis labels
params = {'legend.fontsize': 10,
         'axes.labelsize': 14,
         'axes.titlesize':16,
         'xtick.labelsize':12,
         'ytick.labelsize':12}
pl.rcParams.update(params)

## Sets the directory for all plots
fig_out_dir_02 = 'Figs/'
    
## set a csv table containing the star's common name, V mag, and ST (like B2Ve)
master_tbl = np.genfromtxt('catalog.csv', delimiter=',', dtype=str)
Plot_all_with_spectra = False
    
if Plot_all_with_spectra:
    all_spec_dirs = [f for f in listdir('BeSS_spectra/')] 
    ## 'star_ID_list' will be the stars common names or IDs you want
    star_ID_list = all_spec_dirs

else:
    ## If you set Plot_all_with_spectra = False, then here you say which name/ID you want
    star_ID_list = ['bet_CMi']

for star_ID in star_ID_list:
    sns.set_theme(style="ticks")
    sns.set_palette("colorblind")
    
    ## Main_dir is where all of the BeSS spectra .fits files are
    Main_dir = 'BeSS_spectra/{0}/BeSS/'.format(star_ID)
    fig_out_dir_01 = 'BeSS_spectra/{0}/'.format(star_ID)
    
    ## checking coordinates
    star_coord = starcoord(star_ID)
    RA = str(star_coord.ra.degree)
    DEC = str(star_coord.dec.degree)
    
    ## checking if there are old fits
    fits_list = glob(f'BeSS_spectra/{star_ID}/BeSS/*.fits')
    old = False
    for f in fits_list:
        if time.time() - os.path.getmtime(f) > (3 * 30 * 24 * 60 * 60):
            old = True
        else:
            old = False
    
    ## downloading
    if (len(fits_list) == 0) | (old == True):
        automatic_BeSS(RA, DEC, Main_dir, size='0.2')
        print(f'Download realizado para a estrela {star_ID}')
    else:
        print(f'Estrela {star_ID} ja possui espectros recentemente baixados')
    
    try:
        pasta = star_ID
        table_index = np.where(master_tbl[:,0] == star_ID)[0][0] # row of the star
        common_name = master_tbl[table_index][0]
        ST = master_tbl[table_index][1]
        Vmag = master_tbl[table_index][2]
    except:
        pasta = star_ID
        common_name = pasta
        ST = 'NULL'
        Vmag = 'NULL'

    all_spec_fnames = [ f for f in listdir(Main_dir) if isfile(join(Main_dir,f)) ]
    
    Usefull_all_list = []
    Usefull_all_dict = {'Ha': [], 'Hb': [], 'Hg': [], 'Hd': []}
        
    HJD_all = []
    Usefull_lines = {'Ha': 6562.81, 'Hb': 4861.34,'Hg': 4340.48, 'Hd': 4101.75}
    HJD_list = {'Ha': [], 'Hb': [], 'Hg': [], 'Hd': []} 
    ## stores the HJDs of all the Halpha observations
    HJD_foryear_list = {'Ha': [], 'Hb': [], 'Hg': [], 'Hd': []}

    ## In the next few lines we remove the spectrum from Ultraviolet instruments for technical
    ## reasons (IUE)
    for jj, fname in enumerate(all_spec_fnames):
        hdulist = fits.open(Main_dir + fname)
        obs_instrument = hdulist[0].header['BSS_INST']
        if obs_instrument not in ['IUE_SWP_HS', 'IUE_LWR_HS']:
            ref_pixel = hdulist[0].header['CRVAL1']
            coord_increment = hdulist[0].header['CDELT1']
            obs_date = hdulist[0].header['DATE-OBS']              
            obs_HJD = float(hdulist[0].header['MID-HJD']) - 2450000
            obs_HJD_2 = float(hdulist[0].header['MID-HJD'])
                      
        scidata_0 = hdulist[1].data
        scidata_wave = scidata_0['WAVE']        
        scidata_flux = scidata_0['FLUX']
        scidata_flux = list(scidata_flux)
        
        wls = np.zeros(np.shape(scidata_wave),)
        wls[0] = ref_pixel
        for i,value in enumerate(wls[:-1]):
            wls[i+1] = wls[i] + coord_increment
        
        correction_fact = hdulist[0].header['BSS_VHEL']
        helio = hdulist[0].header['BSS_RQVH']
        ## light speed in km/s
        c = 299792.458
        
        ## heliocentric correction
        if (correction_fact == 0) or (correction_fact == '0') or (correction_fact == 0.0):
            wls = wls * (1 - helio / c)
        else:
            pass
        
        for i, (k, xxxx) in enumerate(Usefull_lines.items()):
            if ( (xxxx < wls.max()) and (xxxx > wls.min()) ):
                try:
                    Usefull_all_dict[k].append([wls, scidata_flux, obs_HJD, fname])
                    HJD_all.append(obs_HJD)
                except KeyError:
                    Usefull_all_dict[k].append([wls, scidata_flux, obs_HJD, fname])
                    HJD_all.append(obs_HJD)                   
        
    ## Usefull_all_dict: dicionario 'Ha': wls, scidata, obs, 'Hb': wls, scidata, obs
    ## HJD_list: dicionario 'Ha': obs_HJD, 'Hb': obs_HJD, etc
    EW_dict = {'Ha': [], 'Hb': [], 'Hg': [], 'Hd': []}
    ## stores the HJDs of all the Halpha observations
    HJD_dict = {'Ha': [], 'Hb': [], 'Hg': [], 'Hd': []}
    VR_dict = {'Ha': [], 'Hb': [], 'Hg': [], 'Hd': []}
    ## auxiliar HJD dict for V/R, as V/R ratio can be NaN!
    HJD_VR = {'Ha': [], 'Hb': [], 'Hg': [], 'Hd': []}
    
    ## aqui a gente itera pelas LINHAS (Ha, Hb, Hg, Hd, etc...)
    for lkey, line in (Usefull_lines.items()):
        ax_LP = plt.subplot2grid((4,1), (0,0),rowspan=2)
        ax_EW = plt.subplot2grid((4,1), (2,0))
        ax_VR = plt.subplot2grid((4,1), (3,0))
        
        fnamelist = []
        HaEWlist = []
        HaVRlist = []
        ## aqui a gente itera por todos os arquivos .fits...
        for i, x in enumerate(Usefull_all_dict[lkey]):
            wl, flx = x[0], x[1]  # wl here is initially in Angstroms
            HJD = x[2]
            fname = x[3]
            
            ## try section because we don't know if there are more lines
            ## besides Ha!
            try:   
                ## normalization of the spectrum, and transformation of x-axis to velocity 
                vl, fx = spt.lineProf(wl, convolve(flx, Box1DKernel(2)), lbc=line, hwidth=1000)
                ## removal of outliers
                vl, fx = Sliding_Outlier_Removal(vl, fx, 50, 5, 7)
                
                ## extra correction for star's radial velocity (to try to force centralizing at v=0)
                EW_VR_ratio, RV_offset = find_bisectors(vl, fx, 600, counter_val = 1.33, emission_frac = 0.5, region_size_flx = 0.5)
                vl = vl - RV_offset
                
                ## calculating V/R variation
                V_I, V_RV, R_I, R_RV, C_RV = LP_fit(vl, fx, lp_width=600., sample_peak_width=50., test_mode=False)
                if not math.isnan(V_I) and not math.isnan(R_I):
                    VR = V_I / R_I

                ## if you wish EW in velocity (km/s, uncomment this and comment the next block)
                # EW = spt.EWcalc(vl, fx, vw=500) 
                ## EW calculation in ANGSTROMS with sum integral
                vw = 500.
                vels = vl
                idx = np.where(np.abs(vels) <= vw)
                outvels = wl[idx]
                normflux = fx[idx]
                EW = 0.
                if len(outvels) < 3:
                    # normflux = _np.ones(len(outvels))
                    EW = EW
                for idx in range(len(outvels) - 1):
                    dl = outvels[idx + 1] - outvels[idx]
                    EW += (1. - (normflux[idx + 1] + normflux[idx]) / 2.) * dl
            
                EW_dict[lkey].append(EW)
                ## storing HJDs
                HJD_dict[lkey].append(HJD)
                
                if lkey == 'Ha':
                    fnamelist.append(fname)
                    HaEWlist.append(EW)
                    HaVRlist.append(VR)
    
                if len(Usefull_all_dict[lkey]) == 1:
                    color = 'k'
                else:
                    color = pl.cm.jet((HJD - np.array(HJD_all).min())/(np.array(HJD_all).max() - np.array(HJD_all).min()))
                
                ax_EW.scatter(HJD, EW, color=color, edgecolor='k',zorder=15, s=70, lw=2)
                
                ## only saves V/R ratio if result is not NaN
                VR_dict[lkey].append(VR)
                HJD_VR[lkey].append(HJD)
                ax_VR.scatter(HJD, VR, color=color, edgecolor='k', zorder=15, s=70, lw=2)
                
                ## plotting in the top panel all spectra 
                ax_LP.plot(vl, fx, label='{0:.1f}'.format(EW), alpha=0.45, color=color)
                
                ## printa qual arquivo está gerando a imagem. 
                if not quiet and (lkey != 'Ha'):    
                    print(f'O arq. {fname} possui a linha {lkey}') 
            except:
                continue
        
        
        #ax_LP.text(0.01, 0.9, lkey+'\n{0} spectra'.format(len(Usefull_all_dict[lkey])),color='k',fontsize=18,transform=ax_LP.transAxes,bbox=dict(facecolor='white',alpha=0.9,edgecolor='none'))
        ## sort the EW values and HJD values so that they are in increasing order in time
        EW_list_s = np.array(EW_dict[lkey])[np.argsort(np.array(HJD_dict[lkey]))]
        VR_list_s = np.array(VR_dict[lkey])[np.argsort(np.array(HJD_VR[lkey]))]
        HJD_VR_s = np.array(HJD_VR[lkey])[np.argsort(np.array(HJD_VR[lkey]))]
        HJD_list_s = np.array(HJD_dict[lkey])[np.argsort(HJD_dict[lkey])]
        
        ## calculate some statistics based on EW values
        EW_min = EW_list_s.min()
        EW_max = EW_list_s.max()
        EW_mean = np.mean(EW_list_s)
        EW_std = np.std(EW_list_s)
        EW_range = EW_min - EW_max
        
        ## Add this text to the plot
        #EW_text = 'EW_min = {0:.0f}\nEW_max = {1:.0f}\nEW_avg = {2:.0f}\nEW_std = {3:.0f}\nEW_range = {4:.0f}'.format(EW_min, EW_max, EW_mean, EW_std, EW_range)
        #ax_LP.text(0.01,0.7,EW_text,color='k',fontsize=13,transform=ax_LP.transAxes,bbox=dict(facecolor='white',alpha=0.9,edgecolor='none'))
        
        if len(Usefull_all_dict[lkey]) > 1:
            EW_range = EW_list_s.max() - EW_list_s.min()
            VR_range = VR_list_s.max() - VR_list_s.min()
            
            ax_EW.set_ylim(EW_list_s.max() + EW_range*0.1, EW_list_s.min() - EW_range*0.1)
            ax_VR.set_ylim(VR_list_s.max() + VR_range*0.1, VR_list_s.min() - VR_range*0.1)

            ## this creates the top legend at EW panel with the dates in years
            new_time_array = Time(HJD_list_s + 2450000, format='jd')
            new_t_years = new_time_array.byear
            ax_EW2 = ax_EW.twiny()
            ax_EW2.xaxis.tick_top()
            ax_EW2.xaxis.set_label_position('top')
            
            ax_EW2.set_xticks(np.arange(int(new_t_years.min()) - 1, int(new_t_years.max() + 1), 2.0))
            ax_EW2.tick_params(labelsize=18)
            orig_xlimits = ax_EW.get_xlim()
            orig_xlim_time_jd = Time(np.array(orig_xlimits) + 2450000, format='jd')
            orig_xlim_time_yr = orig_xlim_time_jd.byear
            ax_EW2.set_xlim(orig_xlim_time_yr)
            
            ax_EW.plot(HJD_list_s, EW_list_s, linestyle='dotted', color='k')
            ax_VR.plot(HJD_VR_s, VR_list_s, linestyle='dotted', color='k')
            
        elif len(Usefull_all_dict[line]) == 1:
            ax_EW.set_ylim(EW_list_s.max() + 0.2*EW_list_s.max(), EW_list_s.max() - 0.2*EW_list_s.max())
            ax_VR.set_ylim(VR_list_s.max() + 0.2*VR_list_s.max(), VR_list_s.max() - 0.2*VR_list_s.max())
                  
        ax_LP.set_xlabel('Velocity (km/s)', fontsize=20)
        ax_LP.set_ylabel('Relative Flux', fontsize=20)
        
        ax_EW.tick_params(labelsize=18)
        #ax_EW.set_ticklabels([])
        ax_EW.xaxis.set_visible(False)
        ax_VR.tick_params(labelsize=18)
        ax_LP.set_xlim(-1000,1000)
        ax_LP.tick_params(labelsize=18)
        ax_LP.grid(color='gray', linestyle='--', linewidth=0.25)
        ax_EW.grid(color='gray', linestyle='--', linewidth=0.25)
        ax_VR.grid(color='gray', linestyle='--', linewidth=0.25)
        
        #ax_EW.legend(fancybox=True, framealpha=0.75, fontsize=11)
        
        LP_y_limits = ax_LP.get_ylim()
        if LP_y_limits[0] < 0.0:
            ax_LP.set_ylim([0.0, LP_y_limits[1]])
               
        #ax_EW.set_xlabel('HJD - 2450000')
        ax_EW.set_ylabel(r'EW ($\AA$)', fontsize=20)
        
        ax_VR.set_xlabel('HJD - 2450000', fontsize=20)
        ax_VR.set_ylabel(r'V/R', fontsize=20)
        
        #yticks = [0.9, 1, 1.1]
        #ticklabels = ['0.9', '1.0', '1.1']
        #ax_VR.set_yticks(yticks, ticklabels)
        
        #xticks = [2002, 2003, 2005, 2007, 2009, 2011, 2013, 2015, 2017, 2019, 2021, 2022]
        #ticklabels = ['2002', '2003', '2005', '2007', '2009', '2011', '2013', '2015', '2017', '2019', '2021', '2022']
        #ax_EW2.set_xticks(xticks, ticklabels)
        
        plt.gcf().set_size_inches(16,12)
        plt.tight_layout()
        
        plt.savefig('{0}/{1}_{2}.pdf'.format(fig_out_dir_01, common_name, lkey))
        plt.savefig('{0}/{1}_{2}.pdf'.format(fig_out_dir_02, common_name, lkey))
        print('\nPlot saved for star {0}\n'.format(common_name))
