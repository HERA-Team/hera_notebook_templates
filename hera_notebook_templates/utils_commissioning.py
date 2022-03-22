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

# useful global variables
status_colors = {
    'dish_maintenance' : 'salmon',
    'dish_ok' : 'red',
    'RF_maintenance' : 'lightskyblue',
    'RF_ok' : 'royalblue',
    'digital_maintenance' : 'plum',
    'digital_ok' : 'mediumpurple',
    'calibration_maintenance' : 'lightgreen',
    'calibration_ok' : 'green',
    'calibration_triage' : 'lime'}
status_abbreviations = {
    'dish_maintenance' : 'dish-M',
    'dish_ok' : 'dish-OK',
    'RF_maintenance' : 'RF-M',
    'RF_ok' : 'RF-OK',
    'digital_maintenance' : 'dig-M',
    'digital_ok' : 'dig-OK',
    'calibration_maintenance' : 'cal-M',
    'calibration_ok' : 'cal-OK',
    'calibration_triage' : 'cal-Tri'}


#####################################################################################################################
################################################# UTILITY FUNCTIONS #################################################
#####################################################################################################################

def getRandPercentage(data,percentage):
    """
    Simple helper function to select a random subset of data points. Useful when the number of points causes plotting functions to become exceedingly slow.
    
    Parameters:
    ----------
    data: numpy array
        1D numpy array containing the data to filter.
    percentage: Int
        Percentage of data points to keep.
        
    Returns:
    ----------
    data: numpy array
        A new data array with a smaller number of data points.
    indices: List
        A list of indices that index into the original data array to extract the points that are kept in the output data array.
        
    """
    k = len(data) * percentage // 100
    indices = np.random.sample(int(k))*len(data)
    data = [data[int(i)] for i in indices]
    return data, indices


def getBlsByConnectionType(uvd,inc_autos=False):
    """
    Simple helper function to generate a dictionary that categorizes baselines by connection type. Resulting dictionary makes it easy to extract data for all baselines of a given baseline type. 
    
    Parameters:
    ----------
    uvd: UVData Object
        Any sample UVData object, used to get antenna information only.
    inc_autos: Boolean
        Option to include autocorrelations in the intrasnap set. Default is False.
        
    Returns:
    ----------
    bl_list: Dict
        Dictionary with keys 'internode', 'intranode', 'intrasnap', and 'autos', which reference lists of all baselines of that connection type. Baselines categorized as intranode will exclude those that are also intrasnap - to get all baselines within the same node, combine these two lists.
    """
    
    nodes, antDict, inclNodes = generate_nodeDict_allPols(uvd)
    bl_list = {'internode' : [], 'intranode' : [], 'intrasnap' : [], 'autos' : []}
    ants = uvd.get_ants()
    for a1 in ants:
        for a2 in ants:
            if a1 == a2:
                bl_list['autos'].append((a1,a2))
                if inc_autos is False:
                    continue
            n1 = antDict[f'{a1}E']['node']
            n2 = antDict[f'{a2}E']['node']
            s1 = antDict[f'{a1}E']['snapLocs'][0]
            s2 = antDict[f'{a2}E']['snapLocs'][0]
            if n1 == n2:
                if s1 == s2:
                    bl_list['intrasnap'].append((a1,a2))
                else:
                    bl_list['intranode'].append((a1,a2))
            else:
                bl_list['internode'].append((a1,a2))
    return bl_list

#####################################################################################################################
########################################### VISIBILITY PLOTTING FUNCTIONS ###########################################
#####################################################################################################################


def plotCrossWaterfalls(uvd_sum, perBlSummary, percentile_set = [1,20,40,60,80,99], savefig=False, outfig='', 
                        pol='allpols', metric='abs'):
    """
    Function to plot a set of cross visibility waterfalls. Baselines are with a correlation metric value equal to the percentile of the total distribution specified by the percentile_set parameter will be plot. This plot is useful for seeing how the visibilities change with a higher or lower correlation metric value.
    
    Parameters:
    ----------
    uvd_sum: UVData Object
        Sum visibility data.
    perBlSummary: Dict
        A dictionary containing a per baseline summary of the correlation data to plot.
    percentile_set: List
        Set of correlation metric percentiles to plot baselines for. Default is [1,20,40,60,80,99].
    savefig: Boolean
        Option to write out the figure.
    outfig: String
        Full path to write out the figure.
    pol: String
        Any polarization string included in perBlSummary. Default is 'allpols'.
    metric: String
        Can be 'abs', 'real', 'imag', or 'phase' to plot the absolute value, real component, imaginary component, or phase of the visibilites, respectively.
        
    Returns:
    ----------
    None
    
    """
    keys = ['all','internode','intranode','intrasnap']
    freqs = uvd_sum.freq_array[0]*1e-6
    lsts = uvd_sum.lst_array*3.819719
    inds = np.unique(lsts,return_index=True)[1]
    lsts = [lsts[ind] for ind in sorted(inds)]
    xticks = [int(i) for i in np.linspace(0,len(freqs)-1,5)]
    xticklabels = [int(f) for f in freqs[xticks]]
    yticks = [int(i) for i in np.linspace(0,len(lsts)-1,6)]
    yticklabels = [np.around(lsts[ytick],1) for ytick in yticks]
    fig, axes = plt.subplots(len(percentile_set), 4, figsize=(16,len(percentile_set)*3))
    fig.subplots_adjust(left=0.05, bottom=0.03, right=.9, top=0.95, wspace=0.15, hspace=0.3)
    corrmap = plt.get_cmap('plasma')
    cNorm = colors.Normalize(vmin=-2, vmax=0)
    scalarMap = cm.ScalarMappable(norm=cNorm, cmap=corrmap)
    for j,p in enumerate(percentile_set):
        for i,key in enumerate(keys):
            vals = np.abs(np.nanmean(perBlSummary[pol][f'{key}_vals'],axis=1))
            ax = axes[j,i]
            v = np.percentile(vals,p)
            vind = np.argmin(abs(vals-v))
            bl = perBlSummary[pol][f'{key}_bls'][vind]
            if pol == 'allpols':
                blpol = (int(bl[0][:-1]),int(bl[1][:-1]),f'{bl[0][-1]}{bl[1][-1]}')
            else:
                blpol = (int(bl[0][:-1]),int(bl[1][:-1]),pol)
            if metric == 'phase':
                dat = np.angle(uvd_sum.get_data(blpol))
                vmin = -np.pi
                vmax = np.pi
                cmap = 'twilight'
            elif metric == 'abs':
                dat = np.abs(uvd_sum.get_data(blpol))
                vmin = np.percentile(dat,1)
                vmax = np.percentile(dat,99)
                cmap = 'viridis'
            else:
                cmap = 'coolwarm'
                if metric == 'real':
                    dat = np.real(uvd_sum.get_data(blpol))
                elif metric == 'imag':
                    dat = np.imag(uvd_sum.get_data(blpol))
                if np.percentile(dat,99) > np.abs(np.percentile(dat,1)):
                    vmax = np.percentile(dat,99)
                    vmin = -vmax
                else:
                    vmin = np.percentile(dat,1)
                    vmax = -vmin
            im = ax.imshow(dat,interpolation='nearest',aspect='auto',vmin=vmin,vmax=vmax,cmap=cmap)
            if v < 0.2:
                ax.set_title(f'{blpol[0],blpol[1],blpol[2]}',backgroundcolor=scalarMap.cmap((np.log10(v)+2)/2),color='white')
            else:
                ax.set_title(f'{blpol[0],blpol[1],blpol[2]}',backgroundcolor=scalarMap.cmap((np.log10(v)+2)/2),color='black')
            if i == 0:
                ax.set_ylabel('Time (LST)')
            if j == len(percentile_set)-1:
                ax.set_xlabel('Frequency (MHz)')
            ax.set_xticks(xticks)
            ax.set_xticklabels(xticklabels)
            ax.set_yticks(yticks)
            ax.set_yticklabels(yticklabels)
            if i==0:
                ax.annotate(f'{p}th percentile',xy=(0,0.5), xytext=(-ax.yaxis.labelpad, 0),xycoords=ax.yaxis.label, 
                            textcoords='offset points', ha='right', va='center', rotation=90, fontsize=16)
            if j==0:
                ax.annotate(f'{key}',xy=(0.5,1.15), xytext=(0,5),xycoords='axes fraction', 
                            textcoords='offset points', ha='center', va='baseline', fontsize=18,annotation_clip=False)
        pos = ax.get_position()
        cbar_ax=fig.add_axes([0.91,pos.y0,0.01,pos.height])        
        cbar = fig.colorbar(im, cax=cbar_ax)
        if metric != 'phase':
            cbar.set_ticks([])
    fig.suptitle(f'cross visbilities - {metric}, {pol} pol',fontsize=20)
    if savefig is True:
        if outfig == '':
            print('#### Must provide value for outfig when savefig is True ####')
        else:
            print(f'Saving {outfig}_{pol}_{metric}.jpeg')
            plt.savefig(f'{outfig}_{pol}_{metric}.jpeg',bbox_inches='tight')

            
            
def plotRealImag2DHists_byMetric(uvd_sum,uvd_diff=None,contour=False,savefig=False,title='',bl='all',metric=None,
                                 conTypes=['all','internode','intranode','intrasnap','autos'], 
                                 selectOnConnection=False,inc_autos=True,scale='auto',xlim=None,color='auto',
                                 vmax=120, nbins=500):
    fig = plt.figure(figsize=(26,20))
    gs=gridspec.GridSpec(4,14, width_ratios=[4,0.2,0.5,4,0.2,0.5,4,0.2,0.5,4,0.2,0.5,4,0.2],wspace=0.2)
    freqInds = [(0,1535),(0,512),(512,1024),(1024,1535)]
    freqs = uvd_sum.freq_array[0]*1e-6
    for m,met in enumerate(conTypes):
        print(m)
        if met == 'autos':
            uvd = uvd_sum.select(ant_str='auto',inplace=False)
            if uvd_diff is not None:
                uvd2 = uvd_diff.select(ant_str='auto',inplace=False)
        elif met != 'all':
            bl_list = getBlsByConnectionType(uvd)
            uvd = uvd_sum.select(bls=bl_list[met],inplace=False)
            if uvd_diff is not None:
                uvd2 = uvd_diff.select(bls=bl_list[met],inplace=False)
        else:
            uvd = uvd_sum
            uvd2 = uvd_diff
        for i,r in enumerate(freqInds):
            fmin = r[0]
            fmax = r[1]
            if metric is None:
                if bl=='all':
                    dat = uvd.data_array[:,:,fmin:fmax,:]
                else:
                    data = uvd.get_data(bl)
                    if len(bl)==2:
                        dat = data[:,fmin:fmax,:]
                    else:
                        dat = data[:,fmin:fmax]
            else:
                if bl=='all':
                    sm = uvd.data_array[:,:,fmin:fmax,:]
                    df = uvd2.data_array[:,:,fmin:fmax,:]
                    if metric == 'even':
                        dat = (sm + df)/2
                    elif metric == 'odd':
                        dat = (sm - df)/2
                else:
                    sm = uvd.get_data(bl)
                    df = uvd2.get_data(bl)
                    if metric == 'even':
                        dat = (sm + df)/2
                    elif metric == 'odd':
                        dat = (sm - df)/2
                    if len(bl)==2:
                        dat = dat[:,fmin:fmax,:]
                    else:
                        dat = dat[:,fmin:fmax]
            x = np.real(dat.flatten())
            y = np.imag(dat.flatten())

            if scale == 'auto':
                xlim = np.percentile(x,98)
            elif scale == 'fixed':
                xlim = xlim
            elif scale == 'max':
                xlim = np.max(x)
            ylim = xlim
            x_bins = np.linspace(-xlim,xlim,nbins)
            y_bins = np.linspace(-ylim,ylim,nbins)
            ax = plt.subplot(gs[i,m*3])
            ax.axvline(0,color='r',linestyle='--')
            ax.axhline(0,color='r',linestyle='--')
            hist, xedges, yedges = np.histogram2d(x,y,bins=[x_bins,y_bins])
            if color=='auto':
                vmax=np.percentile(hist,99)
                if vmax<0.01:
                    vmax = np.max(hist)
            else:
                vmax=vmax
            im = ax.imshow(hist,aspect='auto',interpolation='nearest',norm=colors.LogNorm(vmin=0.01,vmax=vmax),
                   extent=[min(xedges),max(xedges),min(yedges),max(yedges)])
            if contour is True:
                plt.contour(counts,extent=[x_bins.min(),x_bins.max(),y_bins.min(),y_bins.max()],linewidths=3)
            ax.set_xlabel('Real')
            if i==0:
                ax.set_title(met)
            if m==0:
                ax.annotate(f'{int(freqs[fmin])} - {int(freqs[fmax])}MHz',xy=(0,0.5), xytext=(-ax.yaxis.labelpad, 0),xycoords=ax.yaxis.label, 
                            textcoords='offset points', ha='right', va='center', rotation=90, fontsize=16)
            ax = plt.subplot(gs[i,m*3+1])
            fig.colorbar(im,cax=ax)

    if savefig is True:
        plt.savefig(f'{title}.jpeg',bbox_inches='tight')
            

#####################################################################################################################
########################################## CORRELATION CALCULATION FUNCTIONS ########################################
#####################################################################################################################


def getPerBaselineSummary(uvd_sum,uvd_diff,interval_type='even_odd',interval=1,pols=['EE','NN','EN','NE'],avg='mean'):
    """
    Function to produce a dictionary containing correlation metric values for different polarizations and baseline types.
    
    Parameters:
    ----------
    uvd_sum: UVData Object
        Sum visibilities.
    uvd_diff: UVData Object
        Diff visibilities.
    interval_type: String
        Can be 'even_odd' (default), which sets the standard even odd interleave, or 'ns', which will result in an interleave every n seconds, where n is set by the 'interval' parameter.
    interval: Int
        Parameter to set the interleave interval if interval_type = 'ns'. Units are number of integrations.
    pols: List
        Polarizations to include. Default is ['EE','NN','EN','NE'].
    avg: String
        Sets the time averaging of the data. Can be 'mean', 'median', or None to not do any time averaging.
        
    Returns:
    ----------
    perBlSummary: Dict
        A dictionary containing a per baseline summary of the correlation data, with a key for each provided polarization, and an 'allpols' key corresponding to data with all provided polarizations combined. For each polarization, there are 'all_vals', 'internode_vals', 'intranode_vals', and 'intrasnap_vals' keys. 
    """
    antnums = uvd_sum.get_ants()
    antpos, ants = uvd_sum.get_ENU_antpos()
    h = cm_hookup.Hookup()
    x = h.get_hookup('HH')
    perBlSummary = {pol : {'all_vals' : [],
           'intranode_vals' : [],
           'intrasnap_vals' : [],
           'internode_vals' : [],
           'all_bls' : [],
           'intranode_bls' : [],
           'intrasnap_bls' : [],
           'internode_bls' : []} for pol in np.append(pols,'allpols')}
    for i,a1 in enumerate(antnums):
        for j,a2 in enumerate(antnums):
            if a1>=a2:
                continue
            for pol in pols:
                s = uvd_sum.get_data(a1,a2,pol)
                d = uvd_diff.get_data(a1,a2,pol)
                if interval_type=='even_odd':
                    e = (s + d)/2
                    o = (s - d)/2
                else:
                    e = s[:-interval:2,:]
                    o = s[interval::2,:]
                c = np.multiply(e,np.conj(o))
                c /= np.abs(e)
                c /= np.abs(o)
                if avg == 'mean':
                    val = np.nanmean(c,axis=0)
                elif avg == 'median':
                    val = np.nanmedian(c,axis=0)
                elif avg == None:
                    val = c.flatten()
                key1 = 'HH%i:A' % (a1)
                p1 = pol[0]
                n1 = x[key1].get_part_from_type('node')[f'{p1}<ground'][1:]
                snapLoc1 = (x[key1].hookup[f'{p1}<ground'][-1].downstream_input_port[-1], a1)[0]
                key2 = 'HH%i:A' % (a2)
                p2 = pol[1]
                n2 = x[key2].get_part_from_type('node')[f'{p2}<ground'][1:]
                snapLoc2 = (x[key2].hookup[f'{p2}<ground'][-1].downstream_input_port[-1], a2)[0]
                if a1 != a2:
                    perBlSummary[pol]['all_vals'].append(val)
                    perBlSummary[pol]['all_bls'].append((a1,a2))
                    perBlSummary['allpols']['all_vals'].append(val)
                    perBlSummary['allpols']['all_bls'].append((a1,a2))
                    if n1==n2:
                        if snapLoc1==snapLoc2:
                            perBlSummary[pol]['intrasnap_vals'].append(val)
                            perBlSummary[pol]['intrasnap_bls'].append((a1,a2))
                            perBlSummary['allpols']['intrasnap_vals'].append(val)
                            perBlSummary['allpols']['intrasnap_bls'].append((a1,a2))
                        else:
                            perBlSummary[pol]['intranode_vals'].append(val)
                            perBlSummary[pol]['intranode_bls'].append((a1,a2))
                            perBlSummary['allpols']['intranode_vals'].append(val)
                            perBlSummary['allpols']['intranode_bls'].append((a1,a2))
                    else:
                        perBlSummary[pol]['internode_vals'].append(val)
                        perBlSummary[pol]['internode_bls'].append((a1,a2))
                        perBlSummary['allpols']['internode_vals'].append(val)
                        perBlSummary['allpols']['internode_bls'].append((a1,a2))
    return perBlSummary


#####################################################################################################################
########################################### CORRELATION PLOTTING FUNCTIONS ##########################################
#####################################################################################################################


def plotCorrSpectraAndHists(uvd_sum,uvd_diff,perBlSummary='auto',interval_type='even_odd',interval=1,outfig='',savefig=False,
                           freq_range=[132,148], pol='allpols',percentage=100,avg='mean'):
    """
    Function to plot spectra of the correlation metric overplot for all baselines, along with histograms comparing the real and imaginary components of the metric and the internode, intranode, and intrasnap subsets. The histograms typically only use data from a subset of frequencies.
    
    Parameters:
    ----------
    uvd_sum: UVData Object
        Sum visibilities.
    uvd_diff: UVData Object
        Diff visibilities.
    perBlSummary: Dict or 'auto'
        A dictionary containing a per baseline summary of the correlation data to plot, or 'auto' to calculate this automatically using the getPerBaselineSummary function.
    interval_type: String
        Can be 'even_odd' (default), which sets the standard even odd interleave, or 'ns', which will result in an interleave every n seconds, where n is set by the 'interval' parameter.
    interval: Int
        Parameter to set the interleave interval if interval_type = 'ns'. Units are number of integrations.
    savefig: Boolean
        Option to write out the figure.
    outfig: String
        Full path to write out the figure.
    freq_range: List
        Range of frequencies to use in the histograms. Formatted as [minimum frequency, maximum frequency]. Frequencies are in MHz. Default is [132,148].
    pol: String
        Polarization to plot. Can be any polarization key in perBlSummary. Default is 'allpols'.
    percentage: Int
        Percentage of data points to include in scatter plot. Default is 100, although a value of 10 to 20 is recommended when plotting a full data set. With many baselines and times included, the number of points to plot can overwhelm the plotting function and cause it to either crash or take a long time (potentially hours) to display the plot. If set to a value less than 100, this parameter will randomly select a subset of data points to plot, so as to minimize the load on the plotting function.
    avg: String
        Sets the time averaging of the data. Can be 'mean', 'median', or None to not do any time averaging.
        
    Returns:
    ----------
    perBlSummary: Dict
        The same perBlSummary that was either provided or calculated if set to 'auto'.
        
    """

    if perBlSummary == 'auto':
        print('Calculating perBlSummary')
        perBlSummary = getPerBaselineSummary(uvd_sum,uvd_diff,interval_type=interval_type,interval=interval,avg=avg)
    
    print('Calculating Arrays')
    c = np.asarray(perBlSummary[pol]['all_vals'])
    c_re = np.reshape(c,(np.shape(c)[0],np.shape(c)[1]//1536,1536))
    c_avg = np.nanmean(c_re,axis=(0,1))

    freqs = uvd_sum.freq_array[0]*1e-6
    freqs_all = np.tile(freqs,np.shape(c)[0]*int(np.shape(c)[1]/1536))
    c_all = np.asarray(c).flatten()
    c_all, inds = getRandPercentage(c_all,percentage)
    freqs_all = [freqs_all[int(i)] for i in inds]
    
    c_snap_full = np.asarray(perBlSummary[pol]['intrasnap_vals'])
    freqs_snap = np.tile(freqs,np.shape(c_snap_full)[0]*int(np.shape(c)[1]/1536))
    c_snap = np.asarray(c_snap_full).flatten()
    c_snap, inds = getRandPercentage(c_snap,percentage)
    freqs_snap = [freqs_snap[int(i)] for i in inds]
    
    c_node_full = np.asarray(perBlSummary[pol]['intranode_vals'])
    freqs_node = np.tile(freqs,np.shape(c_node_full)[0]*int(np.shape(c)[1]/1536))
    c_node = np.asarray(c_node_full).flatten()
    c_node, inds = getRandPercentage(c_node,percentage)
    freqs_node = [freqs_node[int(i)] for i in inds]
    
    freq_min_ind = np.argmin(np.abs(np.subtract(freqs,freq_range[0])))
    freq_max_ind = np.argmin(np.abs(np.subtract(freqs,freq_range[1])))

    hist_vals = c[:,freq_min_ind:freq_max_ind].flatten()
    hist_vals_snap = c_snap_full[:,freq_min_ind:freq_max_ind].flatten()
    hist_vals_node = c_node_full[:,freq_min_ind:freq_max_ind].flatten()
    
    print('Plotting')
        
    fig,axes = plt.subplots(3,2,figsize=(16,16))
    
    legend_elements = [Line2D([0],[0],marker='o',label='internode',markerfacecolor='b',color='w',alpha=0.5),
                      Line2D([0],[0],marker='o',label='intranode',markerfacecolor='c',color='w',alpha=0.5),
                      Line2D([0],[0],marker='o',label='intrasnap',markerfacecolor='r',color='w',alpha=0.5)]

    axes[0][0].scatter(freqs_all,np.real(c_all),s=0.5,alpha=0.01,color='b',label='internode')
    axes[0][0].scatter(freqs_node,np.real(c_node),s=1.5,alpha=0.1,color='c',label='intranode')
    axes[0][0].scatter(freqs_snap,np.real(c_snap),s=1.5,alpha=0.1,color='r',label='intrasnap')
    axes[0][0].scatter(freqs,np.real(c_avg),s=1,color='k')
    axes[0][0].legend(handles=legend_elements,fontsize=12)
    axes[0][0].set_ylim(-1,1)
    if interval_type == 'even_odd':
        axes[0][0].set_ylabel('Re(even x odd*)')
    else:
        axes[0][0].set_ylabel(r'Re(T$_n$ x T$_{(n+}$' + str(interval) + r'$_)$')
    axes[0][0].set_xlabel('Frequency (MHz)')
    axes[0][0].axhline(0,color='r',linewidth=1)
    axes[0][0].axvline(freq_range[0],color='g')
    axes[0][0].axvline(freq_range[1],color='g')
    axes[0][0].set_title('Real Spectra')

    axes[0][1].scatter(freqs_all,np.imag(c_all),s=0.5,alpha=0.01,color='b',label='internode')
    axes[0][1].scatter(freqs_node,np.imag(c_node),s=1.5,alpha=0.1,color='c',label='intranode')
    axes[0][1].scatter(freqs_snap,np.imag(c_snap),s=1.5,alpha=0.1,color='r',label='intrasnap')
    axes[0][1].scatter(freqs,np.imag(c_avg),s=1,color='k')
    axes[0][1].set_ylim(-1,1)
    if interval_type == 'even_odd':
        axes[0][1].set_ylabel('Im(even x odd*)')
    else:
        axes[0][1].set_ylabel(r'Im(T$_n$ x T$_{(n+}$' + str(interval) + r'$_)$')
    axes[0][1].set_xlabel('Frequency (MHz)')
    axes[0][1].axhline(0,color='r',linewidth=1)
    axes[0][1].axvline(freq_range[0],color='g')
    axes[0][1].axvline(freq_range[1],color='g')
    axes[0][1].set_title('Imag Spectra')

    bins = np.linspace(-1.05,1.05,44)
    
    ####### Connectivity Histograms ######
    axes[1][0].hist(np.real(hist_vals),log=True,bins=bins,edgecolor='b',fill=False,label='internode',
                    linewidth=2,density=True)
    axes[1][0].hist(np.real(hist_vals_node),log=True,bins=bins,edgecolor='c',fill=False,label='intranode',
                    linewidth=2,density=True)
    axes[1][0].hist(np.real(hist_vals_snap),log=True,bins=bins,edgecolor='r',fill=False,label='intrasnap',
                    linewidth=2,density=True)
    axes[1][0].set_xlim(-1,1)
    axes[1][0].set_ylim(10e-4,10e0)
    axes[1][0].set_title('Real Histogram')

    axes[1][1].hist(np.imag(hist_vals),log=True,bins=bins,edgecolor='b',fill=False,label='internode',
                    linewidth=2,density=True)
    axes[1][1].hist(np.imag(hist_vals_node),log=True,bins=bins,edgecolor='c',fill=False,label='intranode',
                    linewidth=2,density=True)
    axes[1][1].hist(np.imag(hist_vals_snap),log=True,bins=bins,edgecolor='r',fill=False,label='intrasnap',
                    linewidth=2,density=True)
    axes[1][1].set_xlim(-1,1)
    axes[1][1].set_ylim(10e-4,10e0)
    axes[1][1].set_title('Imag Histogram')
    axes[1][0].legend()
    
    ###### Real vs Imag histograms ######
    axes[2][0].hist(np.real(hist_vals),bins=bins,edgecolor='b',fill=False,label='Real',density=True,
                   linewidth=2)
    axes[2][1].hist(np.real(hist_vals),log=True,bins=bins,edgecolor='b',fill=False,label='Real',density=True,
                   linewidth=2)
    axes[2][0].set_xlim(-1,1)
    axes[2][0].set_ylim(0,3)

    axes[2][0].hist(np.imag(hist_vals),bins=bins,edgecolor='g',fill=False,label='Imag',density=True)
    axes[2][1].hist(np.imag(hist_vals),log=True,bins=bins,edgecolor='g',fill=False,label='Imag',density=True,
                   linewidth=2)
    axes[2][1].set_xlim(-1,1)
    axes[2][1].set_ylim(10e-4,10e0)
    axes[2][0].legend()
    axes[2][0].set_title('All bls, linear scale')
    axes[2][1].set_title('All bls, log scale')

    print('Displaying - this may take several minutes.')
    if interval_type == 'ns':
        interval_type = f'{interval*10}s'
    fig.tight_layout(pad=3)
    fig.suptitle(f'{interval_type} interleave - {pol} pol')
    if savefig == True:
        plt.savefig(f'{outfig}_{pol}.jpeg',bbox_inches='tight')
        plt.close()
    else:
        plt.show()
    return perBlSummary


def plotPerBaselineSummary(file,corr_real,corr_imag,use_ants='all',excludeAnts=[],colorBy='inputs',badAnts=[],crossedAnts=[],
                          savefig=False,outfig='',pols=['E','N'],ylim=[-0.1,0.7]):
    """
    Function to plot the real and imaginary components of the correlation metric versus baseline length, colored by internode, intranode, and intrasnap baseline types.
    
    Parameters:
    ----------
    file: String
        Full path to a sample data file, used to extract antenna connectivity information.
    corr_real: numpy array
        2x2 numpy array containing the real values of the correlation metric.
    corr_imag: numpy array
        2x2 numpy array containing the imaginary values of the correlation metric.
    use_ants: List or 'all'
        List of antennas to include, or set to 'all' to include all ants.
    excludeAnts: List
        List of antennas to exclude from plotting. Can be used in place of or in conjunction with use_ants - if there are conflicts, this parameter overrules use_ants, and all antennas listed in this parameter will be excluded.
    colorBy: String
        Can be 'inputs' or 'flags'. If set to 'inputs', the data will be colored by internode, intranode, and intrasnap connections. If set to 'flags', the data will be colored by whether it is marked bad by the badAnts parameter, crossed by the crossedAnts parameter, or considered okay by being ommitted from both lists.
    badAnts: List
        List of antennas flagged as bad. If colorBy is set to 'flags', antennas listed here will be included in the set of bad antennas on the plot.
    crossedAnts: List
        List of antennas flagged as cross polarized. If colorBy is set to 'flags', antennas listed here will be included in the set of crossed antennas on the plot.
    savefig: Boolean
        Option to write out the figure.
    outfig: String
        Full path to write out the figure.
    pols: List
        Polarizations to include. Default is ['E','N']. Must be the same set of polarizations included in corr_real and corr_imag.
    ylim: List
        y-axis limits, formatted as [minimum y value, maximum y value]. Default is [-0.1, 0.7].
        
    Returns:
    ----------
    None
    
    """
    uvd = UVData()
    if use_ants != 'all':
        uvd.read(file,antenna_nums=use_ants)
    else:
        uvd.read(file)
    antnums, sortedSnapLocs, sortedSnapInputs = utils.sort_antennas(uvd,use_ants=use_ants,pols=pols)
    antpos, ants = uvd.get_ENU_antpos()
    h = cm_hookup.Hookup()
    x = h.get_hookup('HH')
    
    fig,ax = plt.subplots(1,2,figsize=(16,7))
    dirs = ['North-North','East-East','North-East','East-North']
    metrics = ['real','imag']
    for m, metric in enumerate(metrics):
        dat = {'all_lengths' : [],
               'all_lengths' : [],
               'all_vals' : [],
               'intranode_lengths' : [],
               'intranode_vals' : [],
               'intrasnap_lengths' : [],
               'intrasnap_vals' : [],
               'badAnts' : [],
               'crossedAnts' : [],
               'goodAnts' : [],
               'badAntsLengths' : [],
               'crossedAntsLengths' : [],
               'goodAntsLengths' : []}
        for i,a1 in enumerate(antnums):
            for j,a2 in enumerate(antnums):
                if len(pols) > 1:
                    ant1 = int(a1[:-1])
                    ant2 = int(a2[:-1])
                    p1 = a1[-1]
                    p2 = a2[-1]
                else:
                    ant1 = a1
                    ant2 = a2
                    p1 = pols[0]
                    p2 = pols[0]
                if a1==a2 or ant1==ant2:
                    continue
                if a1 in excludeAnts or a2 in excludeAnts or ant1 in excludeAnts or ant2 in excludeAnts:
                    continue
                a1pos = antpos[np.argwhere(ants==ant1)]
                a2pos = antpos[np.argwhere(ants==ant2)]
                length = np.sqrt(np.sum(np.square(a1pos-a2pos)))
                if metric == 'real':
                    val = corr_real[i,j]
                elif metric == 'imag':
                    val = corr_imag[i,j]
                else:
                    print('###### metric must be either real or imag #####')
                if colorBy == 'inputs':
                    key1 = 'HH%i:A' % (ant1)
                    n1 = x[key1].get_part_from_type('node')[f'{p1}<ground'][1:]
                    snapLoc1 = (x[key1].hookup[f'{p1}<ground'][-1].downstream_input_port[-1], ant1)[0]
                    key2 = 'HH%i:A' % (ant2)
                    n2 = x[key2].get_part_from_type('node')[f'{p2}<ground'][1:]
                    snapLoc2 = (x[key2].hookup[f'{p2}<ground'][-1].downstream_input_port[-1], ant2)[0]
                    if n1==n2:
                        if snapLoc1==snapLoc2:
                            dat['intrasnap_lengths'].append(length)
                            dat['intrasnap_vals'].append(val)
                        else:
                            dat['intranode_lengths'].append(length)
                            dat['intranode_vals'].append(val)
                    else:
                        dat['all_lengths'].append(length)
                        dat['all_vals'].append(val)
                elif colorBy == 'flags':
                    dat['all_lengths'].append(length)
                    dat['all_vals'].append(val)
                    if ant1 in badAnts or ant2 in badAnts:
                        dat['badAnts'].append(val)
                        dat['badAntsLengths'].append(length)
                    elif ant1 in crossedAnts or ant2 in crossedAnts:
                        dat['crossedAnts'].append(val)
                        dat['crossedAntsLengths'].append(length)
                    else:
                        dat['goodAnts'].append(val)
                        dat['goodAntsLengths'].append(length)
        axis = ax[m]
        if colorBy == 'inputs':
            axis.scatter(dat['all_lengths'],dat['all_vals'],label='Internode',color='blue')
            axis.scatter(dat['intranode_lengths'],dat['intranode_vals'],label='Intranode',color='cyan')
            axis.scatter(dat['intrasnap_lengths'],dat['intrasnap_vals'],label='Intrasnap',color='red')
        elif colorBy == 'flags':
            axis.scatter(dat['badAntsLengths'],dat['badAnts'],label='Bad Antennas',color='red')
            axis.scatter(dat['goodAntsLengths'],dat['goodAnts'],label='Good Antennas',color='blue')
            axis.scatter(dat['crossedAntsLengths'],dat['crossedAnts'],label='Crossed Antennas',color='cyan')
        axis.set_title(f'{metric}',fontsize=30)
        axis.set_yscale('symlog')
        axis.set_xlabel('Baseline Length (meters)',fontsize=24)
        axis.tick_params(axis='x', labelsize=18)
        if m==0:
            axis.set_ylabel('$C_{ij}$',fontsize=24)
            axis.tick_params(axis='y', labelsize=18)
        else:
            axis.set_yticks([])
        axis.set_ylim(ylim)
        if m==0:
            axis.legend(fontsize=18)
        if m==1:
            axis.set_yticks([])
            axis.set_yticklabels([])
        axis.axhline(linestyle='--',color='black')
        
    fig.tight_layout()
    if savefig is True:
        plt.savefig(outfig,bbox_inches='tight')