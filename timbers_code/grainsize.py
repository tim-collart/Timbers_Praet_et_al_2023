import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from scipy.signal import argrelextrema

def wrapper(func, args):
    return func(*args)

def um_to_phi(um):
    return -1*np.log2(um/1000)

def phi_to_um(phi):
    return 1000/(2**phi)

def get_grainsize_stats(concentrations, size_classes, statistics_to_calculate = ['mode','indices','geometric moments', 'PSD slope', 'TVC assemblages'], bin_boundaries = [], plot = False,  phi = False):
    '''
    This functions allows to run the different grain size statistics calculations (defined by statistics_to_calculate) for concentrations of size_classes 
    '''
    
    functions = {
        'mode': {
            'function': get_modes,
            'arguments': [concentrations, size_classes['Median'], plot, phi]
                },
        'indices': {
            'function': get_indices,
            'arguments': [concentrations, size_classes['Median'], plot]
                },       
        'arithmetic moments': {
            'function': get_arithmetic_moments,
            'arguments': [concentrations, size_classes['Median'], phi]
                },
        'geometric moments': {
            'function': get_geometric_moments,
            'arguments': [concentrations, size_classes['Median']]
                },
        'folk and ward stats': {
            'function': get_folk_and_ward_stats,
            'arguments': [concentrations, size_classes['Median'], phi]
                },
        'PSD slope': {
            'function': get_PSDslope,
            'arguments': [concentrations, size_classes['Median'], size_classes['Lower'], size_classes['Upper'], plot]
                },
        'TVC assemblages': {
            'function': get_assemblage_TVC,
            'arguments': [concentrations, size_classes['Lower'], bin_boundaries]
                }        
    }
    if concentrations.isnull().all():
        print('No volume concentrations found. Returning null values for all grainsize stats!')
        return None
    
    statistics = pd.Series(dtype = float)
    for stat in statistics_to_calculate:
        statistics = statistics.append(wrapper(functions[stat]['function'], functions[stat]['arguments']))
    return statistics

def get_modes(concentrations, size_classes, plot = False, phi = False):
    '''
    Calculates the local maxima (modes) in a the volume concentrations (ul/l) of the different size_classes measured by the LISST-200X
     input:
    - concentration: concentrations in ul per liter provided by the LISST-200X
    - size_classes: particle size bins derived from LISST-200X manual in um
    - plot: if true, plots the grain size distribution with the modes
    - phi: if true, returns modes in phi units
    returns:
    - a pandas series with the different modes labelled mode_1, mode_2, etc.
    '''
    concentrations = np.array(concentrations)
    size_classes = np.array(size_classes)

    if phi:
        size_classes = um_to_phi(size_classes)
    concentrations_perc = concentrations*100/concentrations.sum()

    # get indixes of the local maxima
    local_conc_max = argrelextrema(concentrations_perc, np.greater)[0]
    # sort them according to concentration in descending order
    local_conc_max = local_conc_max[np.argsort(concentrations_perc[local_conc_max])[::-1]]
    modes = size_classes[local_conc_max]

    if plot:
        fig, axs = plt.subplots()
        axs.plot(size_classes, concentrations_perc, label = 'data')
        axs.plot(modes, concentrations_perc[local_conc_max] , "o", label = 'modes')
        axs.legend()
        if phi:
            axs.set_xlabel('Median Particle Diameter ($\phi$)')
        else:
            axs.set_xlabel('Median Particle Diameter ($\mu m$)')
        axs.set_ylabel('Volume conentration ($\dfrac{\mu l}{l}$)')
    
    return pd.Series(modes, index=['mode_'+str(i+1) for i in range(len(modes))])
    
def get_indices(concentrations, size_classes, plot = False):
    concentrations = np.array(concentrations)
    size_classes = np.array(size_classes)
    percentiles = get_percentiles(concentrations=concentrations,
                                  size_classes=size_classes,
                                  percentiles = [10.,25.,50.,75.,90.],
                                  plot = plot,
                                  phi = False)
    return percentiles[['D10','D50','D90']].append(pd.Series({
        "D90/D10":percentiles['D90']/percentiles['D10'],
        "D90-D10":percentiles['D90']-percentiles['D10'],
        "D75/D25":percentiles['D75']/percentiles['D25'],
        "D75-D25":percentiles['D75']-percentiles['D25'],
        "Trask(So)":(percentiles['D25']/percentiles['D75'])**.5,
        "Krumbein(Qd)":um_to_phi(percentiles['D25'])-um_to_phi(percentiles['D75'])
     }))
    
def get_arithmetic_moments(concentrations, size_classes, phi = False):
    '''
    Calculates the arithmetic moments based on the volume concentrations (ul/l) in the different size_classes measured by the LISST-200X
    This code was based on the R G2Sd package and on Gradistat (https://doi.org/10.1002/esp.261)
     TO TEST::: Testing reveals the code is similar to the output of G2Sd, but differences with the 
    input:
    - concentration: concentrations in ul per liter provided by the LISST-200X
    - size_classes: median of particle size bins derived from LISST-200X manual in um
    - phi: if True, uses phi values for the calculations, i.e. this is the logarithmic method of moments
    returns:
    - a pandas series of the arithmetic or logarithmic moments
    '''
    concentrations = np.array(concentrations)
    size_classes = np.array(size_classes)

    if phi:
        size_classes = um_to_phi(size_classes)
    concentrations_perc = concentrations*100/concentrations.sum()
    
    mean = (concentrations_perc*size_classes).sum()/100.
    sorting = ((concentrations_perc*(size_classes - mean)**2).sum()/100.)**.5
    skewness = (concentrations_perc*(size_classes - mean)**3).sum()/(100.*sorting**3)
    kurtosis = (concentrations_perc*(size_classes - mean)**4).sum()/(100.*sorting**4)
    if phi:
        return pd.Series({
            'mean_loga_phi':mean,
            "sorting_loga_phi":sorting,
            "skewness_loga_phi":skewness,
            "kurtosis_loga_phi":kurtosis
        })
    else:    
        return pd.Series({
            'mean_arith_um':mean,
            "sorting_arith_um":sorting,
            "skewness_arith_um":skewness,
            "kurtosis_arith_um":kurtosis
        })

    
def get_geometric_moments(concentrations, size_classes):
    '''
    Calculates the geometric moments based on the volume concentrations (ul/l) in the different size_classes measured by the LISST-200X
    This code was based on the R G2Sd package and on Gradistat (https://doi.org/10.1002/esp.261)
     TO TEST::: Testing reveals the code is similar to the output of G2Sd, but differences with the 
    Note that we used the natural logarithm (according to Gradistat formulas) instead of log base 10 as in the G2Sd package, though this does not change to output (TEST)
    input:
    - concentration: concentrations in ul per liter provided by the LISST-200X
    - size_classes: median of particle size bins derived from LISST-200X manual in um
    returns:
    - a pandas series of the geometric moments in um
    '''    
    concentrations = np.array(concentrations)
    size_classes = np.array(size_classes)
    
    concentrations_perc = concentrations*100/concentrations.sum()
    size_classes_log = np.log(size_classes)
    mean = np.exp((concentrations_perc*size_classes_log).sum()/100.)
    sorting = np.exp(((concentrations_perc*(size_classes_log - np.log(mean))**2).sum()/100.)**.5)
    skewness = (concentrations_perc*(size_classes_log - np.log(mean))**3).sum()/(100.*np.log(sorting)**3)
    kurtosis = (concentrations_perc*(size_classes_log - np.log(mean))**4).sum()/(100.*np.log(sorting)**4)
    return pd.Series({
        'mean_geom_um':mean,
        "sorting_geom_um":sorting,
        "skewness_geom_um":skewness,
        "kurtosis_geom_um":kurtosis        
    })

def get_folk_and_ward_stats(concentrations, size_classes, phi = False):
    '''
    Calculates the grain size statistics according to the Fold and Ward Graphical measures (original logarithmic or geometric) based on the volume concentrations (ul/l) in the different size_classes measured by the LISST-200X
    This code was based on the R G2Sd package and on Gradistat (https://doi.org/10.1002/esp.261)
     TO TEST::: Testing reveals the code is similar to the output of G2Sd, but differences with the 
    input:
    - concentration: concentrations in ul per liter provided by the LISST-200X
    - size_classes: median of particle size bins derived from LISST-200X manual in um
    - phi: if True, uses phi values for the calculations, i.e. this is the original logarithmic folk and ward graphical method
    returns:
    - a pandas series of the folk and ward grain size statistics
    '''
    concentrations = np.array(concentrations)
    size_classes = np.array(size_classes)
    
    percentiles = get_percentiles(concentrations=concentrations,
                                  size_classes=size_classes,
                                  percentiles = [5.,16.,25.,50.,75.,84.,95.],
                                  phi = phi)
    # for the geometric fold and ward method, use the log of the percentiles
    if not phi:
        percentiles = np.log(percentiles)
    mean = (percentiles['D16']+percentiles['D50']+percentiles['D84'])/3.
    sorting = (percentiles['D84']-percentiles['D16'])/4.+(percentiles['D95']-percentiles['D5'])/6.6
    skewness = (percentiles['D16']+percentiles['D84']-2.*percentiles['D50'])/(2.*(percentiles['D84']-percentiles['D16'])) + \
               (percentiles['D5']+percentiles['D95']-2.*percentiles['D50'])/(2.*(percentiles['D95']-percentiles['D5']))
    kurtosis = (percentiles['D95']-percentiles['D5'])/(2.44*(percentiles['D75']-percentiles['D25']))
    # for the geometric fold and ward method, take the exponent of mean and sorting
    if not phi:
        mean = np.exp(mean)
        sorting = np.exp(sorting)    
    if phi:
        return pd.Series({
            'mean_faw_phi':mean,
            "sorting_faw_phi":sorting,
            "skewness_faw_phi":skewness,
            "kurtosis_faw_phi":kurtosis
        })
    else:    
        return pd.Series({
            'mean_faw_um':mean,
            "sorting_faw_um":sorting,
            "skewness_faw_um":skewness,
            "kurtosis_faw_um":kurtosis
        })
    
def get_percentiles(concentrations, size_classes, percentiles = [10.,50.,90.], plot = False, phi = False):
    '''
    Calculates the percentiles based on the volume concentrations in the different size_classes measured by the LISST-200X
    This code was based on the R G2Sd package and on Gradistat (https://doi.org/10.1002/esp.261).
    Testing reveals the code is similar to the output of G2Sd, but some difference were noted with the gradistat macro
    Note that for um, the function calculates the percentiles based on Phi and converts them back to um.
    Note also that for Phi, we name the percentiles as 100-given percentiles as this appears to be a convention!
    input:
    - concentration: concentrations in ul per liter provided by the LISST-200X
    - size_classes: particle size bins derived from LISST-200X manual in um
    - percentiles: the percentiles to calculate
    - plot: if true, plots the cumulative grain size distribution
    - phi: if true, returns percentiles in phi units
    returns:
    - a pandas series of the percentile values
    '''
    # sort the percentiles as well as the concentrations and size classes
    percentiles = np.sort(np.array(percentiles))
    sort_order = np.argsort(np.array(size_classes))
    size_classes = np.array(size_classes)[sort_order]
    concentrations = np.array(concentrations)[sort_order]
    
    concentrations_perc = concentrations*100/concentrations.sum()
    concentrations_cumsum = concentrations_perc.cumsum()
    cumsum_min = concentrations_cumsum.min()
    # calculate phi values
    size_classes_phi = um_to_phi(size_classes)
    # avoid extrapolation to lower percentiles
    if (percentiles < cumsum_min).any():
        warnings.warn('Following percentiles cannot be calculated: '+ ', '.join([str(perc) for perc in percentiles if perc < cumsum_min]))
        percentiles = [perc for perc in percentiles if perc >= cumsum_min]
    # interpolate percentiles of size_classes in between the concentration cum_sums
    D_phi = np.interp(x = percentiles, xp = concentrations_cumsum, fp = size_classes_phi)
    # D_mu = np.interp(x = percentiles, xp = concentrations_cumsum, fp = size_classes)
    if plot:
        fig, axs = plt.subplots(2)
        axs[0].plot(size_classes_phi, concentrations_cumsum, label = 'data')
        axs[0].plot(D_phi, percentiles, "o", label = 'percentiles')
        axs[0].invert_xaxis()
        axs[0].legend()
        axs[0].set_xlabel('Median Particle Diameter ($\phi$)')
        axs[0].set_ylabel('Cumulative Grainsize Distribution')
        axs[1].plot(size_classes, concentrations_cumsum, label = 'data')
        axs[1].plot(phi_to_um(D_phi), percentiles, "o", label = 'percentiles')
        axs[1].legend()
        axs[1].set_xlabel('Median Particle Diameter ($\mu m$)')
        axs[1].set_ylabel('Cumulative Grainsize Distribution')
    
    return  pd.Series(D_phi if phi else phi_to_um(D_phi),
                      index = ['D'+str(int(100-perc)) for perc in percentiles] if phi else ['D'+str(int(perc)) for perc in percentiles])
    

def get_PSDslope(volume_concentrations, size_classes_mean, size_classes_min, size_classes_max, plot = False):
    '''
    Calculates the particle size distribution (PSD) slope according to Buonassissi, C. J., and Dierssen, H. M. (2010), http://doi.org/10.1029/2010JC006256 
    input:
    - volume_concentrations: volume concentrations in ul per liter provided by the LISST-200X
    - size_classes_mean: median particle size bins derived from LISST-200X manual in um
    - size_classes_min: lower limit of particle size bins derived from LISST-200X manual in um
    - size_classes_max: upper limit of particle size bins derived from LISST-200X manual in um
    returns:
    - a pandas series with the PSD slope parameter 
    '''
    # turn into numpy array to avoid pandas based index matching
    volume_concentrations = np.array(volume_concentrations)
    size_classes_mean = np.array(size_classes_mean)
    size_classes_min = np.array(size_classes_min)
    size_classes_max = np.array(size_classes_max)
    # transform volume concentrations (ul/l) to particle concentrations (particles/l)
    # according to part/L*m3/part*1e3L/m3*1e6uL/L = uL/L
    # where the particle is assumed to be a sphere with vol. 4/3*pi*(median R)^3
    particle_concentrations = volume_concentrations/(1e9*4./3.*np.pi*(size_classes_mean*1e-6/2)**3)
    # The data is now per size class and want to make it spectral (per um)
    # the size classes are roughly lognormal with 36 classes 
    # dv=log10([1.25 250]);
    # vec=dv(1):(dv(2)-dv(1))./32:dv(2);
    # vec2=10.^vec;
    # calculate the width of the size classes in um
    width = size_classes_max-size_classes_min
    # normalize particle concentrations (particles/lum) and volume concentrations (ul/lum) by scaling by the width of each size class
    norm_particle_concentrations = particle_concentrations/width
    norm_volume_concentrations = volume_concentrations/width
    
    # find the logarithmic slope of the normalize particle concentrations
    # remove 0's to avoid errors whent taking logs
    mask_0 = norm_particle_concentrations > 0
    norm_particle_concentrations_fit = norm_particle_concentrations[mask_0]
    size_classes_mean_fit = size_classes_mean[mask_0]
    # start the analysis after the maximum to avoid bad tail
    max_i = np.argmax(norm_particle_concentrations_fit)
    norm_particle_concentrations_fit = norm_particle_concentrations_fit[max_i:]
    size_classes_mean_fit = size_classes_mean_fit[max_i:]
    # get the slope of a polynomial fit
    slope,intercept = np.polyfit(np.log10(size_classes_mean_fit), np.log10(norm_particle_concentrations_fit), 1)
    # plot
    if plot:
        fig, axs = plt.subplots()
        axs.plot(size_classes_mean_fit, norm_particle_concentrations_fit, 'o',label = 'data')
        axs.plot(size_classes_mean_fit, (10**intercept)*size_classes_mean_fit**slope, label = 'fit')
        axs.set_xscale('log')
        axs.set_yscale('log')
        axs.set_xlabel('Median Particle Diameter ($\mu m$)')
        axs.set_ylabel('Normalized Particle Concentration ($ \dfrac{particles}{l*\mu m} $) ')
        axs.legend()
    # return the jungian slope = negative of the fitted slope
    return pd.Series({'PSD_slope':-slope})

def get_assemblage_TVC(concentrations, size_classes, bin_boundaries = []):
    '''
    Calculates the total volume concentration (TVC) from volume concentrations in custom bins (assemblages) of size_classes measured by the LISST-200X
    input:
    - concentration: concentrations in ul per liter provided by the LISST-200X
    - size_classes: particle size bins derived from LISST-200X manual in um
    - bin_boundaries: the cut-values in um for each assemblage
    returns:
    - a pandas series of assemblage TVC's
    '''
    # get bin intervals from the provided bin boundaries 
    bins = get_size_class_assemblages(size_classes, bin_boundaries)
    # sum the volume concentrations according to the bins and return
    return concentrations.groupby(bins).sum().add_prefix('TVC_')

def get_size_class_assemblages(size_classes, bin_boundaries = []):
    '''
    input:
    - size_classes: particle size bins derived from LISST-200X manual in um
    - bin_boundaries: the cut-values in um for each assemblage
    returns:
    - the size class bins as defined by bin_boundaries
    '''
    return np.array(pd.cut(size_classes, bins = np.concatenate([[0.],bin_boundaries,[size_classes.max()]])))

def get_assemblage_apparent_density(size_classes, bin_boundaries,
                                    rho_p = 2500,
                                    d_p = 2, d_p_li = 1, d_p_ui = 3,
                                    F = 2, F_li = 1.9, F_ui = 2.1,
                                    plot = True):
    '''
    input:
    - d_f: particle size bins derived from LISST-200X manual in um
    - bin_boundaries: the cut-values in um for each assemblage
    Note that the values for calculating the rho_a floc apparent density are the default ones defined in the floc_apparent_density_function
    See this function for more details
    output:
    -  returns the floc apparent density for each assemblage as well as the complete size_class range
    '''
    rho_a = floc_apparent_density(size_classes,
                                  rho_p = rho_p,
                                  d_p = d_p, d_p_li = d_p_li, d_p_ui = d_p_ui,
                                  F = F, F_li = F_li, F_ui = F_ui,
                                  plot = plot)
    assemblages = get_size_class_assemblages(size_classes =  size_classes,
                                            bin_boundaries = bin_boundaries)
    
    rho_assemblages = rho_a.groupby(assemblages).mean()
    rho_full = rho_a.mean().rename('({}, {}]'.format(assemblages.min().left, assemblages.max().right))
    results = rho_assemblages.append(rho_full)
    results.index = [str(i) for i in results.index]
    results.index.name = 'Assemblage'
    return results

def floc_apparent_density(d_f,
                          rho_p = 2500,
                          d_p = 2, d_p_li = 1, d_p_ui = 3,
                          F = 2, F_li = 1.9, F_ui = 2.1,
                          plot = True):
    '''
    Calculate the floc apparent density rho_a with upper and lower interval in kg/m3 as done in Fettweis et al., 2021.
    d_f in µm, list of the floc diameters for which the density needs to be calculated
    rho_p = 2500 kg/m3, primary particle density (Fettweis et al., 2008)
    d_p = 2 µm, primary particle size (Fettweis et al., 2021)
    d_p_li = 1 µm, lower interval of primary particle size for Kwinte study area (Fettweis et al., 2008)
    d_p_ui = 3 µm, upper interval of primary particle size for Kwinte study area (Fettweis et al., 2008) 
    F = 2 Fractal Dimension (Fettweis et al., 2021) 
    F_li = 1.9 Fractal Dimension, lower interval for Kwinte study area (Fettweis et al., 2008) 
    F_ui = 2.1 Fractal Dimension, upper interval for Kwinte study area (Fettweis et al., 2008) 
    '''
    rho_a = floc_apparent_density_formula(rho_p = rho_p,
                                          d_p = d_p,
                                          d_f = d_f,
                                          F = F)
    rho_a_li = floc_apparent_density_formula(rho_p = rho_p,
                                          d_p = d_p_li,
                                          d_f = d_f,
                                          F = F_li)
    rho_a_ui = floc_apparent_density_formula(rho_p = rho_p,
                                          d_p = d_p_ui,
                                          d_f = d_f,
                                          F = F_ui)
    # rho_a_li_li = floc_apparent_density_formula(rho_p = rho_p,
    #                                       d_p = d_p_li,
    #                                       d_f = d_f,
    #                                       F = F_li)
    # rho_a_li_ui = floc_apparent_density_formula(rho_p = rho_p,
    #                                       d_p = d_p_li, 
    #                                       d_f = d_f,
    #                                       F = F_ui)
    # rho_a_ui_li = floc_apparent_density_formula(rho_p = rho_p,
    #                                       d_p = d_p_ui,
    #                                       d_f = d_f,
    #                                       F = F_li)
    # rho_a_ui_ui = floc_apparent_density_formula(rho_p = rho_p,
    #                                       d_p = d_p_ui, 
    #                                       d_f = d_f,
    #                                       F = F_ui)    
    results = pd.concat([rho_a.rename('rho_a'),
                         # rho_a_li_li.rename('rho_a_li_li'),
                         # rho_a_li_ui.rename('rho_a_li_ui'),
                         # rho_a_ui_li.rename('rho_a_ui_li'),
                         # rho_a_ui_ui.rename('rho_a_ui_ui'),
                         rho_a_li.rename('rho_a_li'),
                         rho_a_ui.rename('rho_a_ui')
                        ],
                        axis = 1)
    results.index = d_f
    results.rho_a[results.index < d_p] = rho_p
    results.rho_a_li[results.index < d_p] = results.rho_a_li.iloc[np.argmax(results.index >= d_p)] # first value above the d_p
    results.rho_a_ui[results.index < d_p] = results.rho_a_ui.iloc[np.argmax(results.index >= d_p)] # first value above the d_p
    if plot:
        fig, ax = plt.subplots(figsize = (10,10))
        results.plot(ax = ax, logx = True) # logy=True
        ax.set_ylabel(r'$\rho_a [kg/m^3]$')
    return results
    
def floc_apparent_density_formula(rho_p, d_p, d_f, F):
    '''
    formula from Fall et al., 2021 to calculate floc apparent density
    rho_p in kg/m3, primary particle density
    d_p in µm, primary particle size
    d_f in µm, list of the floc diameters
    '''
    return rho_p*((d_p/d_f)**(3-F))