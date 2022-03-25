# -*- coding: utf-8 -*-
# Copyright 2022 the HERA Project
# Licensed under the MIT License
# Written by Dara Storer - dstorer@uw.edu

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import numpy as np
from pyuvdata import UVCal, UVData, UVFlag, utils
import os
import sys
import glob
import uvtools as uvt
from astropy.time import Time
from astropy.coordinates import EarthLocation, AltAz, Angle
from astropy.coordinates import SkyCoord as sc
import pandas
import warnings 
import copy
from hera_mc import cm_hookup, geo_sysdef
import math
from uvtools import dspec
import hera_qm 
from hera_mc import cm_active
from matplotlib.lines import Line2D
from matplotlib import colors, cm
import json
from hera_notebook_templates.data import DATA_PATH
from astropy.io import fits
import csv
from astropy import units as u
from astropy_healpix import HEALPix
from astropy.coordinates import Galactic
import healpy
from multiprocessing import Process, Queue
from bokeh.layouts import row, column
from bokeh.models import CustomJS, Select, RadioButtonGroup, Range1d
from bokeh.plotting import figure, output_file, show, ColumnDataSource
import scipy
from hera_notebook_templates import utils
warnings.filterwarnings('ignore')




def makeCorrMatrices(HHfiles,use_ants,sm=None,df=None,nfilesUse=10,freq_inds=[],
                    nanDiffs=False,pols=['EE','NN','EN','NE'],interleave='even_odd'):
    """
    Wrapper function to calculate correlation matrices and plot the real and imaginary components, as well as a version on a linlog scale that allows us to more easily compare the real metric value to the expected real value as estimated by the range of the imaginary values.
    
    Parameters:
    ----------
    HHfiles: List
        List of sum files (not include auto files) to get data from.
    use_ants: List
        List of antennas to use.
    sm: UVData Object
        UVData object containing sum visibilities. If provided, this data will be used to calculate the metric. If set to None, new data will be read in using the HHfiles list provided. Default is None.
    df: UVData Object
        UVData object containing diff visibilities. If provided, this data will be used to calculate the metric. If set to None, new data will be read in using the HHfiles list provided. Default is None.
    nfilesUse: Int
        Number of files to integrate over when calculating the metric. Default is 10.
    freq_inds: List
        Frequency indices used to clip the data. Format should be [minimum frequency index, maximum frequency index]. The data will be averaged over all frequencies between the two indices. The default is [], which means the metric will average over all frequencies.
    savefig: Boolean
        Option to write out the figure.
    nanDiffs: Boolean
        Option to set all diff values of zero to NaN. Useful when there are issues causing occasional zeros in the diffs, which will cause issues when dividing by the diffs. Setting to NaN allows a nanaverage to mitigate the issue. Default is False.
    pols: List
        List of polarizations to calculate the metric for. Default is ['EE','NN','EN','NE'].
    interleave: String
        Sets the interleave interval. Options are 'even_odd' or 'adjacent_integration'. When set to 'even_odd', the evens and odds in the metric calculation are set using the sums and diffs. When set to 'adjacent_integration', adjacent integrations are used in place of the evens and odds for the interleave.
    
    Returns:
    ----------
    sm: UVData Object
        Sum visibilities.
    df: UVData Object
        Diff visibilities.
    corr_real:
        2x2 numpy array containing the real values of the correlation matrix.
    corr_imag:
        2x2 numpy array containing the imaginary values of the correlation matrix.
    perBlSummary: Dict
        Dictionary containing metric summary data for each polarization, separated by node, snap, and snap input connectivity.
    """
    
    nHH = len(HHfiles)
    use_files_sum = HHfiles[nHH//2-nfilesUse//2 : nHH//2+nfilesUse//2]
    use_files_diff = [file.split('sum')[0] + 'diff.uvh5' for file in use_files_sum]
    if len(freq_inds) == 0:
        print('All frequency bins')
    else:
        nfreqs = freq_inds[1] - freq_inds[0]
        print(f'{nfreqs} frequency bins')
    print(f'{len(use_files_sum)} times')

    if sm is None:
        sm = UVData()
        print('Reading sum files')
        sm.read(use_files_sum,antenna_nums=use_ants)
        
    if df is None:
        df = UVData()
        print('Reading diff files')
        df.read(use_files_diff,antenna_nums=use_ants)
        
    
    # Calculate real and imaginary correlation matrices
    corr_real, perBlSummary = calc_corr_metric(sm,df,norm='real',freq_inds=freq_inds,nanDiffs=nanDiffs,
                                              pols=pols,interleave=interleave)
    corr_imag,_ = calc_corr_metric(sm,df,norm='imag',freq_inds=freq_inds,nanDiffs=nanDiffs,
                                  pols=pols,interleave=interleave)
    
    # Plot matrix of real values
    plot_single_matrix(sm,corr_real,logScale=True,vminIn=0.01,suptitle='|Real|',
                       interleave=interleave)
    # Plot matrix of imaginary values
    plot_single_matrix(sm,corr_imag,logScale=False,vminIn=-0.03,vmaxIn=0.03,
                                cmap='bwr',suptitle='Imaginary',interleave=interleave)
    # Plot matrix of real values on linlog color scale.
    plot_single_matrix(sm,corr_real,dataRef=corr_imag,linlog=True,vminIn=0.01,suptitle='Real',cmap='bwr')
    
    return sm, df, corr_real, corr_imag, perBlSummary

def calc_corr_metric(uvd_sum,uvd_diff,use_ants='all',norm='abs',freq_inds=[],time_inds=[],pols=['EE','NN','EN','NE'],
                    nanDiffs=False,interleave='even_odd', divideByAbs = True):
    """
    Function to calculate the correlation metric: even x conj(odd) / (|even|x|odd|). The resulting 2x2 array will have one entry per baseline, with each row and column representing an antenna. The ordering of antennas along the axes will be sorted by node number, within that by SNAP number, and within that by SNAP input number.
    
    
    Parameters:
    ----------
    uvd_sum: UVData Object
        Sum file data.
    uvd_diff: UVData Object
        Diff file data.
    use_ants: List or 'all'
        List of antennas to use, or set to 'all' to include all antennas.
    norm: String
        Can be 'abs', 'real', 'imag', or 'max', which indicate to take the absolute value, real component, imaginary component, or maximum value of the metric, respectively.
    freq_inds: List
        Frequency indices used to clip the data. Format should be [minimum frequency index, maximum frequency index]. The data will be averaged over all frequencies between the two indices. The default is [], which means the metric will average over all frequencies.
    time_inds: List
        Time axis indices used to clip the data. Format should be [minimum time index, maximum time index]. The data will be averaged over all times between the two indices. The default is [], which means the metric will average over all times in the provided dataset.
    pols: List
        Polarizations to include. Default is ['EE','NN','EN','NE'].
    nanDiffs: Boolean
        Option to set all diff values of zero to NaN. Useful when there are issues causing occasional zeros in the diffs, which will cause issues when dividing by the diffs. Setting to NaN allows a nanaverage to mitigate the issue. Default is False.
    interleave: String
        Sets the interleave interval. Options are 'even_odd' or 'adjacent_integration'. When set to 'even_odd', the evens and odds in the metric calculation are set using the sums and diffs. When set to 'adjacent_integration', adjacent integrations are used in place of the evens and odds for the interleave.
    divideByAbs: Boolean
        Option to divide the metric by the absolute value of the evens and odds. Default is True. Setting to False will result in an un-normalized metric.
        
        
    Returns:
    ---------
    corr: numpy array
        2x2 numpy array containing the resulting values of the correlation matrix.
    perBlSummary: Dict
        Dictionary containing metric summary data for each polarization, separated by node, snap, and snap input connectivity.
    """
    
    if use_ants == 'all':
        use_ants = uvd_sum.get_ants()
    useAnts,_,_ = utils.sort_antennas(uvd_sum,use_ants,['N','E'])
    antnums = [int(a[0:-1]) for a in useAnts]
    corr = np.zeros((len(useAnts),len(useAnts)))
    perBlSummary = {pol : {'all_vals' : [],
           'intranode_vals' : [],
           'intrasnap_vals' : [],
           'internode_vals' : [],
           'all_bls' : [],
           'intranode_bls' : [],
           'intrasnap_bls' : [],
           'internode_bls' : []} for pol in np.append(pols,'allpols')}
    h = cm_hookup.Hookup()
    x = h.get_hookup('HH')
    for i,a1 in enumerate(useAnts):
        for j,a2 in enumerate(useAnts):
            ant1 = int(a1[:-1])
            ant2 = int(a2[:-1])
            p1 = str(a1[-1])
            p2 = str(a2[-1])
            pol = f'{a1[-1]}{a2[-1]}'
            if pol not in pols:
                continue
            key = (ant1,ant2,pol)
            s = np.asarray(uvd_sum.get_data(key))
            d = np.asarray(uvd_diff.get_data(key))
            if nanDiffs is True:
                dAbs = np.asarray(np.abs(d))
                locs = np.where(dAbs == 0)
                d.setflags(write=1)
                d[locs] = np.nan
            if interleave is 'even_odd':
                even = (s + d)/2
                odd = (s - d)/2
            elif interleave is 'adjacent_integration':
                even = s[:len(s)//2*2:2]
                odd = s[1:len(s)//2*2:2]
            if divideByAbs is True:
                even = np.divide(even,np.abs(even))
                odd = np.divide(odd,np.abs(odd))
            else:
                even[even==0] = np.nan
                odd[odd==0] = np.nan
            product = np.multiply(even,np.conj(odd))
            if len(freq_inds) > 0:
                product = product[:,freq_inds[0]:freq_inds[1]]
            if len(time_inds) > 0:
                product = product[time_inds[0]:time_inds[1],:]
            if norm == 'abs':
                corr[i,j] = np.abs(np.nanmean(product))
            elif norm == 'real':
                corr[i,j] = np.real(np.nanmean(product))
            elif norm == 'imag':
                corr[i,j] = np.imag(np.nanmean(product))
            elif norm == 'max':
                corr[i,j] = np.max(product)
            key1 = 'HH%i:A' % (ant1)
            n1 = x[key1].get_part_from_type('node')[f'{p1}<ground'][1:]
            snapLoc1 = (x[key1].hookup[f'{p1}<ground'][-1].downstream_input_port[-1], ant1)[0]
            key2 = 'HH%i:A' % (ant2)
            n2 = x[key2].get_part_from_type('node')[f'{p2}<ground'][1:]
            snapLoc2 = (x[key2].hookup[f'{p2}<ground'][-1].downstream_input_port[-1], ant2)[0]
            if ant1 != ant2:
                perBlSummary[pol]['all_vals'].append(np.nanmean(product,axis=0))
                perBlSummary[pol]['all_bls'].append((a1,a2))
                perBlSummary['allpols']['all_vals'].append(np.nanmean(product,axis=0))
                perBlSummary['allpols']['all_bls'].append((a1,a2))
                if n1==n2:
                    if snapLoc1==snapLoc2:
                        perBlSummary[pol]['intrasnap_vals'].append(np.nanmean(product,axis=0))
                        perBlSummary[pol]['intrasnap_bls'].append((a1,a2))
                        perBlSummary['allpols']['intrasnap_vals'].append(np.nanmean(product,axis=0))
                        perBlSummary['allpols']['intrasnap_bls'].append((a1,a2))
                    else:
                        perBlSummary[pol]['intranode_vals'].append(np.nanmean(product,axis=0))
                        perBlSummary[pol]['intranode_bls'].append((a1,a2))
                        perBlSummary['allpols']['intranode_vals'].append(np.nanmean(product,axis=0))
                        perBlSummary['allpols']['intranode_bls'].append((a1,a2))
                else:
                    perBlSummary[pol]['internode_vals'].append(np.nanmean(product,axis=0))
                    perBlSummary[pol]['internode_bls'].append((a1,a2))
                    perBlSummary['allpols']['internode_vals'].append(np.nanmean(product,axis=0))
                    perBlSummary['allpols']['internode_bls'].append((a1,a2))
    return corr, perBlSummary

def plot_single_matrix(uv,data,antnums='auto',linlog=False,dataRef=None,vminIn=0,vmaxIn=1,logScale=False,pols=['E','N'],
                       savefig=False,outfig='',cmap='plasma',title='Corr Matrix',incAntLines=False):
    """
    Function to plot a single correlation matrix (rather than the standard 4x4 set of matrices).
    
    Parameters:
    ----------
    uv: UVData Object
        Sample observation used for extracting node and antenna information.
    data: numpy array
        2x2 numpy array containing the values to plot, where each axis contains one data point per antenna or antpol. These must be sorted by node, snap, and snap input, respectively when antnums is set to 'auto', otherwise they must be sorted in the same manner as the provided antnums list.
    antnums: List or 'auto'
        List of antennas or antpols represented in the data array. Set this value to 'auto' if antennas are ordered according to the sort_antennas function. Default is 'auto'.
    linlog: Boolean
        Option to plot the data on a linlog scale, such that the colorbar is on a linear scale over some range of reference metric values set by the 99th percentile of values in the provided dataRef, and on a log scale over the remainder of values. The intended use is for dataRef to represent the expected noise of the data. Default is False.
    dataRef: numpy array or None
        2x2 numpy array containing reference metric values to use when setting the linear scale range when linlog is set to True. This parameter is required to use the linlog function. Default is None.
    vminIn: Int
        Minimum colorbar value. Default is 0.
    vmaxIn: Int
        Maximum colorbar value. Default is 1.
    logScale: Boolean
        Option to plot the colorbar on a logarithmic scale. Default is False.
    pols: List
        List of antpols included in the dataset. Used in determining the ordering of antpols in the dataset. Required if antnums is 'auto'.
    savefig: Boolean
        Option to write out figure. Default is False.
    outfig: String
        Path to write figure to. Required if savefig is set to True.
    cmap: String
        Colormap to use. Must be a valid matplotlib colormap. Default is 'plasma'.
    title: String
        Displayed figure title.
    incAntLines: Boolean
        Option to include faint blue lines along the border between each antenna. Default is False.
    """
    nodeDict, antDict, inclNodes = utils.generate_nodeDict(uv,pols=['E','N'])
    nantsTotal = len(uv.get_ants())
    power = np.empty((nantsTotal,nantsTotal))
    fig, axs = plt.subplots(1,1,figsize=(16,16))
    loc = EarthLocation.from_geocentric(*uv.telescope_location, unit='m')
    jd = uv.time_array[0]
    t = Time(jd,format='jd',location=loc)
    lst = round(t.sidereal_time('mean').hour,2)
    t.format='fits'
    if antnums == 'auto':
        antnums, _, _ = utils.sort_antennas(uv,'all',pols=['E','N'])
    nantsTotal = len(antnums)
    nants = len(antnums)
    if linlog is True and dataRef is not None:
        vmin = np.min(dataRef)
        linthresh = np.percentile(dataRef,99)
        norm=colors.SymLogNorm(linthresh=linthresh, linscale=1,vmin=-linthresh, vmax=1.0)
        ptop = int((1-norm(linthresh))*10000)
        pbottom = 10000-ptop
        top = cm.get_cmap('plasma', ptop)
        bottom = cm.get_cmap('binary', pbottom)
        newcolors = np.vstack((bottom(np.linspace(0, 1, pbottom)),top(np.linspace(0, 1, ptop))))
        newcmp = colors.ListedColormap(newcolors, name='linlog')
        im = axs.imshow(data,cmap=newcmp,origin='upper',extent=[0.5,nantsTotal+.5,0.5,nantsTotal+0.5],norm=norm)
    elif linlog is True and dataRef is None:
        print('#################################################################')
        print('ERROR: dataRef parameter must be provided when linlog set to True')
        print('#################################################################')
    elif logScale is True:
        im = axs.imshow(data,cmap=cmap,origin='upper',
                                extent=[0.5,nantsTotal+.5,0.5,nantsTotal+0.5],norm=colors.LogNorm(vmin=vminIn, vmax=vmaxIn))
    else:
        im = axs.imshow(data,cmap=cmap,origin='upper',extent=
                                [0.5,nantsTotal+.5,0.5,nantsTotal+0.5],vmin=vminIn, vmax=vmaxIn)
    axs.set_xticks([])
    axs.set_yticks([])
    n=0
    s=0
    for node in sorted(inclNodes):
        s=n
        for snap in ['0','1','2','3']:
            for snapLoc in nodeDict[node]['snapLocs']:
                loc = snapLoc[0]
                if loc==snap:
                    s+=1
            axs.axhline(len(antnums)-s+.5,lw=2.5,alpha=0.5)
            axs.axvline(s+.5,lw=2.5,alpha=0.5)
        n += len(nodeDict[node]['ants'])
        axs.axhline(len(antnums)-n+.5,lw=5)
        axs.axvline(n+.5,lw=5)
        axs.text(n-len(nodeDict[node]['ants'])/2,-1.7,node,fontsize=14)
    if incAntLines is True:
        for a in range(len(antnums)):
            axs.axhline(len(antnums)-a+0.5,lw=1,alpha=0.5)
            axs.axvline(a+.5,lw=1,alpha=0.5)
    axs.text(.42,-.05,'Node Number',transform=axs.transAxes,fontsize=18)
    n=0
    for node in sorted(inclNodes):
        n += len(nodeDict[node]['ants'])
        axs.text(nantsTotal+1,nantsTotal-n+len(nodeDict[node]['ants'])/2,node,fontsize=14)
    axs.text(1.04,0.4,'Node Number',rotation=270,transform=axs.transAxes,fontsize=18)
    cbar_ax = fig.add_axes([1,0.05,0.015,0.89])
    cbar_ax.set_xlabel(r'$|C_{ij}|$', rotation=0,fontsize=18)
    cbar = fig.colorbar(im, cax=cbar_ax, format='%.2f')
    fig.subplots_adjust(top=1.28,wspace=0.05,hspace=1.1)
    fig.tight_layout(pad=2)
    axs.set_title(title)
    if savefig is True:
        plt.savefig(outfig,bbox_inches='tight')
    plt.show()