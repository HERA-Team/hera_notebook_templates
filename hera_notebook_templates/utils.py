# -*- coding: utf-8 -*-
# Copyright 2020 the HERA Project
# Licensed under the MIT License

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
from matplotlib import colors
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
warnings.filterwarnings('ignore')


def get_use_ants(uvd,statuses,jd):
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
    statuses = statuses.split(',')
#     known_good = ["digital_ok","calibration_maintenance","calibration_ok","calibration_triage"]
#     maybe_good = ["RF_ok","digital_maintenance","digital_ok","calibration_maintenance","calibration_ok","calibration_triage"]
    ants = uvd.antenna_numbers
    use_ants = []
    h = cm_active.ActiveData(at_date=jd)
    h.load_apriori()
    for ant in ants:
        status = h.apriori[f'HH{ant}:A'].status
        if status in statuses:
            use_ants.append(ant)
    return use_ants

def read_template(pol='XX'):
    if pol == 'XX':
        polstr = 'north'
    elif pol == 'YY':
        polstr = 'east'
    temp_path = f'{DATA_PATH}/templates/{polstr}_template.json'
    with open(temp_path) as f:
        data = json.load(f)
    return data

def flag_by_template(uvd,HHfiles,jd,use_ants='auto',pols=['XX','YY'],polDirs=['NN','EE'],temp_norm=True,plotMap=False):
    use_files, use_lsts, use_file_inds = get_hourly_files(uvd,HHfiles,jd)
    temp = {}
    ant_dfs = {}
    for pol in pols:
        temp[pol] = read_template(pol)
        ant_dfs[pol] = {}
    if use_ants == 'auto':
        use_ants = uvd.get_ants()
    flaggedAnts = {polDirs[0]: [], polDirs[1]: []}
    for i,lst in enumerate(use_lsts):
#         print(lst)
        hdat = UVData()
        hdat.read(use_files[i],antenna_nums=use_ants)
        for p,pol in enumerate(pols):
            ant_dfs[pol][lst] = {}
            ranges = np.asarray(temp[pol]['lst_ranges'][0])
            if len(np.argwhere(ranges[:,0]<lst)) > 0:
                ind = np.argwhere(ranges[:,0]<lst)[-1][0]
            else:
                if p == 0:
                    print(f'No template for lst={lst} - skipping')
                continue
            dat = np.abs(temp[pol][str(ind)])
            if temp_norm is True:
                medpower = np.nanmedian(np.log10(np.abs(hdat.data_array)))
                medtemp = np.nanmedian(dat)
                norm = np.divide(medpower,medtemp)
                dat = np.multiply(dat,norm)        
            for ant in use_ants:
                d = np.log10(np.abs(hdat.get_data((ant,ant,pol))))
                d = np.average(d,axis=0)
                df = np.abs(np.subtract(dat,d))
                ant_dfs[pol][lst][ant] = np.nanmedian(df)
            if plotMap is True:
                fig = plt.figure(figsize=(18,10))
                cmap = plt.get_cmap('inferno')
                sm = plt.cm.ScalarMappable(cmap=cmap,norm=plt.Normalize(vmin=0,vmax=1))
                sm._A = []
            ampmin=100000000000000
            ampmax=0
            for ant in use_ants:
                amp = ant_dfs[pol][lst][ant]
                if amp > ampmax: ampmax=amp
                elif amp < ampmin: ampmin=amp
            rang = ampmax-ampmin
            for ant in use_ants:
                idx = np.argwhere(hdat.antenna_numbers == ant)[0][0]
                antPos = hdat.antenna_positions[idx]
                amp = ant_dfs[pol][lst][ant]
                if math.isnan(amp):
                    marker="v"
                    color="r"
                    markersize=30
                else:
                    cind = float((amp-ampmin)/rang)
                    if plotMap is True:
                        coloramp = cmap(cind)
                        color=coloramp
                        marker="h"
                        markersize=40
                if cind > 0.15 and ant not in flaggedAnts[polDirs[p]]:
                    flaggedAnts[polDirs[p]].append(ant)
                if plotMap is True:
                    plt.plot(antPos[1],antPos[2],marker=marker,markersize=markersize,color=color)
                    if math.isnan(amp) or coloramp[0]>0.6:
                        plt.text(antPos[1]-3,antPos[2],str(ant),color='black')
                    else:
                        plt.text(antPos[1]-3,antPos[2],str(ant),color='white')
            if plotMap is True:
                plt.title(f'{polDirs[p]} pol, {lst} hours')
                cbar = fig.colorbar(sm)
                cbar.set_ticks([])
    return ant_dfs, flaggedAnts
    

def load_data(data_path,JD):
    HHfiles = sorted(glob.glob("{0}/zen.{1}.*.sum.uvh5".format(data_path,JD)))
    difffiles = sorted(glob.glob("{0}/zen.{1}.*.diff.uvh5".format(data_path,JD)))
    HHautos = sorted(glob.glob("{0}/zen.{1}.*.sum.autos.uvh5".format(data_path,JD)))
    diffautos = sorted(glob.glob("{0}/zen.{1}.*.diff.autos.uvh5".format(data_path,JD)))
    Nfiles = len(HHfiles)
    hhfile_bases = map(os.path.basename, HHfiles)
    hhdifffile_bases = map(os.path.basename, difffiles)
    sep = '.'
    x = sep.join(HHfiles[0].split('.')[-4:-2])
    y = sep.join(HHfiles[-1].split('.')[-4:-2])
    print(f'{len(HHfiles)} sum files found between JDs {x} and {y}')
    x = sep.join(difffiles[0].split('.')[-4:-2])
    y = sep.join(difffiles[-1].split('.')[-4:-2])
    print(f'{len(difffiles)} diff files found between JDs {x} and {y}')
    x = sep.join(HHautos[0].split('.')[-5:-3])
    y = sep.join(HHautos[-1].split('.')[-5:-3])
    print(f'{len(HHautos)} sum auto files found between JDs {x} and {y}')
    x = sep.join(diffautos[0].split('.')[-5:-3])
    y = sep.join(diffautos[-1].split('.')[-5:-3])
    print(f'{len(diffautos)} diff auto files found between JDs {x} and {y}')

    # choose one for single-file plots
    hhfile1 = HHfiles[len(HHfiles)//2]
    difffile1 = difffiles[len(difffiles)//2]
    if len(HHfiles) != len(difffiles):
        print('############################################################')
        print('######### DIFFERENT NUMBER OF SUM AND DIFF FILES ###########')
        print('############################################################')
    # Load data
    uvd_hh = UVData()

    unread = True
    while unread is True:
        try:
            uvd_hh.read(hhfile1, skip_bad_files=True)
        except:
            hhfile += 1
            continue
        unread = False
    uvd_xx1 = uvd_hh.select(polarizations = -5, inplace = False)
    uvd_xx1.ants = np.unique(np.concatenate([uvd_xx1.ant_1_array, uvd_xx1.ant_2_array]))
    # -5: 'xx', -6: 'yy', -7: 'xy', -8: 'yx'

    uvd_yy1 = uvd_hh.select(polarizations = -6, inplace = False)
    uvd_yy1.ants = np.unique(np.concatenate([uvd_yy1.ant_1_array, uvd_yy1.ant_2_array]))

   
    return HHfiles, difffiles, HHautos, diffautos, uvd_xx1, uvd_yy1

def plot_sky_map(uvd,ra_pad=20,dec_pad=30,clip=True,fwhm=11,nx=300,ny=200,sources=[]):
    map_path = f'{DATA_PATH}/haslam408_dsds_Remazeilles2014.fits'
    hdulist = fits.open(map_path)

    # Set up the HEALPix projection
    nside = hdulist[1].header['NSIDE']
    order = hdulist[1].header['ORDERING']
    hp = HEALPix(nside=nside, order=order, frame=Galactic())
    
    #Get RA/DEC coords of observation
    loc = EarthLocation.from_geocentric(*uvd.telescope_location, unit='m')
    time_array = uvd.time_array
    obstime_start = Time(time_array[0],format='jd',location=loc)
    obstime_end = Time(time_array[-1],format='jd',location=loc)
    zenith_start = sc(Angle(0, unit='deg'),Angle(90,unit='deg'),frame='altaz',obstime=obstime_start,location=loc)
    zenith_start = zenith_start.transform_to('icrs')
    zenith_end = sc(Angle(0, unit='deg'),Angle(90,unit='deg'),frame='altaz',obstime=obstime_end,location=loc)
    zenith_end = zenith_end.transform_to('icrs')
    lst_start = obstime_start.sidereal_time('mean').hour
    lst_end = obstime_end.sidereal_time('mean').hour
    start_coords = [zenith_start.ra.degree,zenith_start.dec.degree]
    if start_coords[0] > 180:
        start_coords[0] = start_coords[0] - 360
    end_coords = [zenith_end.ra.degree,zenith_end.dec.degree]
    if end_coords[0] > 180:
        end_coords[0] = end_coords[0] - 360
    
    # Sample a 300x200 grid in RA/Dec
    ra_range = [zenith_start.ra.degree-ra_pad, zenith_end.ra.degree+ra_pad]
    dec_range = [zenith_start.dec.degree-ra_pad, zenith_end.dec.degree+ra_pad]
    if clip == True:
        ra = np.linspace(ra_range[0],ra_range[1], nx)
        dec = np.linspace(dec_range[0],dec_range[1], ny)
    else:
        ra = np.linspace(-180,180,nx)
        dec = np.linspace(-90,zenith_start.dec.degree+90,ny)
    ra_grid, dec_grid = np.meshgrid(ra * u.deg, dec * u.deg)
    
    #Create alpha grid
    alphas = np.ones(ra_grid.shape)
    alphas = np.multiply(alphas,0.5)
    ra_min = np.argmin(np.abs(np.subtract(ra,start_coords[0]-fwhm/2)))
    ra_max = np.argmin(np.abs(np.subtract(ra,end_coords[0]+fwhm/2)))
    dec_min = np.argmin(np.abs(np.subtract(dec,start_coords[1]-fwhm/2)))
    dec_max = np.argmin(np.abs(np.subtract(dec,end_coords[1]+fwhm/2)))
    alphas[dec_min:dec_max, ra_min:ra_max] = 1

    # Set up Astropy coordinate objects
    coords = sc(ra_grid.ravel(), dec_grid.ravel(), frame='icrs')

    # Interpolate values
    temperature = healpy.read_map(map_path)
    tmap = hp.interpolate_bilinear_skycoord(coords, temperature)
    tmap = tmap.reshape((ny, nx))
    tmap = np.flip(tmap,axis=1)
    alphas = np.flip(alphas,axis=1)

    # Make a plot of the interpolated temperatures
    plt.figure(figsize=(12, 7))
    im = plt.imshow(tmap, extent=[ra[-1], ra[0], dec[0], dec[-1]], 
                    cmap=plt.cm.viridis, aspect='auto', vmin=10,vmax=40,alpha=alphas,origin='lower')
    plt.xlabel('RA (ICRS)')
    plt.ylabel('DEC (ICRS)')
    plt.hlines(y=start_coords[1]-fwhm/2,xmin=ra[-1],xmax=ra[0],linestyles='dashed')
    plt.hlines(y=start_coords[1]+fwhm/2,xmin=ra[-1],xmax=ra[0],linestyles='dashed')
    plt.vlines(x=start_coords[0],ymin=start_coords[1],ymax=dec[-1],linestyles='dashed')
    plt.vlines(x=end_coords[0],ymin=start_coords[1],ymax=dec[-1],linestyles='dashed')
    plt.annotate(np.around(lst_start,2),xy=(start_coords[0],dec[-1]),xytext=(0,8),
                 fontsize=10,xycoords='data',textcoords='offset points',horizontalalignment='center')
    plt.annotate(np.around(lst_end,2),xy=(end_coords[0],dec[-1]),xytext=(0,8),
                 fontsize=10,xycoords='data',textcoords='offset points',horizontalalignment='center')
    plt.annotate('LST (hours)',xy=(np.average([start_coords[0],end_coords[0]]),dec[-1]),
                xytext=(0,22),fontsize=10,xycoords='data',textcoords='offset points',horizontalalignment='center')
    for s in sources:
        if s[1] > dec[0] and s[1] < dec[-1]:
            if s[0] > 180:
                s = (s[0]-360,s[1],s[2])
            if s[2] == 'LMC' or s[2] == 'SMC':
                plt.annotate(s[2],xy=(s[0],s[1]),xycoords='data',fontsize=8,xytext=(20,-20),
                             textcoords='offset points',arrowprops=dict(facecolor='black', shrink=2,width=1,
                                                                        headwidth=4))
            else:
                plt.scatter(s[0],s[1],c='k',s=6)
                if len(s[2]) > 0:
                    plt.annotate(s[2],xy=(s[0]+3,s[1]-4),xycoords='data',fontsize=6)
    plt.show()
    plt.close()
    hdulist.close()

def plot_inspect_ants(uvd1,jd,badAnts=[],flaggedAnts={},tempAnts={},crossedAnts=[],use_ants='auto'):
    status_use = ['RF_ok','digital_ok','calibration_maintenance','calibration_ok','calibration_triage']
    if use_ants == 'auto':
        use_ants = uvd1.get_ants()
    h = cm_active.ActiveData(at_date=jd)
    h.load_apriori()
    inspectAnts = []
    for ant in use_ants:
        status = h.apriori[f'HH{ant}:A'].status
        if ant in badAnts or ant in flaggedAnts.keys() or ant in crossedAnts:
            if status in status_use:
                inspectAnts.append(ant)
        for k in tempAnts.keys():
            if ant in tempAnts[k] and status in status_use:
                inspectAnts.append(ant)
    inspectAnts = np.unique(inspectAnts)
    inspectTitles = {}
    for ant in inspectAnts:
        inspectTitles[ant] = 'Flagged by: '
        if ant in badAnts:
            inspectTitles[ant] = f'{inspectTitles[ant]} correlation matrix,'
        if ant in flaggedAnts.keys():
            inspectTitles[ant] = f'{inspectTitles[ant]} ant_metrics,'
        if ant in crossedAnts:
            inspectTitles[ant] = f'{inspectTitles[ant]} cross matrix,'
        try:
            for k in tempAnts.keys():
                if ant in tempAnts[k]:
                    inspectTitles[ant] = f'{inspectTitles[ant]} template - {k},'
        except:
            continue
        if inspectTitles[ant][-1] == ',':
            inspectTitles[ant] = inspectTitles[ant][:-1]
    print('Antennas that require further inspection are:')
    print(inspectAnts)
    
    for ant in inspectAnts:
        auto_waterfall_lineplot(uvd1,ant,jd,title=inspectTitles[ant])
        
    return inspectAnts
    
def auto_waterfall_lineplot(uv, ant, jd, pols=['xx','yy'], colorbar_min=1e6, colorbar_max=1e8, title=''):
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
    h = cm_active.ActiveData(at_date=jd)
    h.load_apriori()
    status = h.apriori[f'HH{ant}:A'].status
    freq = uv.freq_array[0]*1e-6
    fig = plt.figure(figsize=(12,8))
    gs = gridspec.GridSpec(3, 2, height_ratios=[2,0.7,1])
    it = 0
    pol_dirs = ['NN','EE']
    for p,pol in enumerate(pols):
        waterfall= plt.subplot(gs[it])
        jd_ax=plt.gca()
        times= np.unique(uv.time_array)
        d = np.abs(uv.get_data((ant,ant, pol)))
        if len(np.nonzero(d)[0])==0:
            print('#########################################')
            print(f'Data for antenna {ant} is entirely zeros')
            print('#########################################')
            plt.close()
            return
        im = plt.imshow(d,norm=colors.LogNorm(), 
                    aspect='auto')
        abb = status_abbreviations[status]
        waterfall.set_title(f'{pol_dirs[p]} pol')
        freqs = uv.freq_array[0, :] / 1000000
        xticks = np.arange(0, len(freqs), 120)
        plt.xticks(xticks, labels =np.around(freqs[xticks],2))
        if p == 0:
            jd_ax.set_ylabel('JD')
            jd_yticks = [int(i) for i in np.linspace(0,len(times)-1,8)]
            jd_labels = np.around(times[jd_yticks],2)
            jd_ax.set_yticks(jd_yticks)
            jd_ax.set_yticklabels(jd_labels)
            jd_ax.autoscale(False)
        if p == 1:
            lst_ax = jd_ax.twinx()
            lst_ax.set_ylabel('LST (hours)')
            lsts = uv.lst_array*3.819719
            inds = np.unique(lsts,return_index=True)[1]
            lsts = [lsts[ind] for ind in sorted(inds)]
            lst_yticks = [int(i) for i in np.linspace(0,len(lsts)-1,8)]
            lst_labels = np.around([lsts[i] for i in lst_yticks],2)
            lst_ax.set_yticks(lst_yticks)
            lst_ax.set_yticklabels(lst_labels)
            lst_ax.set_ylim(jd_ax.get_ylim())
            lst_ax.autoscale(False)
            jd_ax.set_yticks([])
        line= plt.subplot(gs[it+2])
        averaged_data= np.abs(np.average(uv.get_data((ant,ant,pol)),0))
        plt.plot(freq,averaged_data)
        line.set_yscale('log')
        if p == 0:
            line.set_ylabel('Night Average')
        else:
            line.set_yticks([])
        line.set_xlim(freq[0],freq[-1])
        line.set_xticks([])
        
        line2 = plt.subplot(gs[it+4])
        dat = uv.get_data((ant,ant,pol))
        dat = np.abs(dat[len(dat)//2,:])
        plt.plot(freq,dat)
        line2.set_yscale('log')
        line2.set_xlabel('Frequency (MHz)')
        if p == 0:
            line2.set_ylabel('Single Slice')
        else:
            line2.set_yticks([])
        line2.set_xlim(freq[0],freq[-1])
        
        plt.setp(waterfall.get_xticklabels(), visible=False)
        plt.subplots_adjust(hspace=.0)
        cbar = plt.colorbar(im, pad= 0.2, orientation = 'horizontal')
        cbar.set_label('Power')
        it=1
    fig.suptitle(f'{ant} ({abb})', fontsize=10, backgroundcolor=status_colors[status],y=0.96)
    plt.annotate(title, xy=(0.5,0.94), ha='center',xycoords='figure fraction')
    plt.show()
    plt.close()

def plot_autos(uvdx, uvdy):
    nodes, antDict, inclNodes = generate_nodeDict(uvdx)
    ants = uvdx.get_ants()
    sorted_ants = sort_antennas(uvdx)
    freqs = (uvdx.freq_array[0])*10**(-6)
    times = uvdx.time_array
    lsts = uvdx.lst_array  
    maxants = 0
    for node in nodes:
        n = len(nodes[node]['ants'])
        if n>maxants:
            maxants = n
    
    Nants = len(ants)
    Nside = maxants
    Yside = len(inclNodes)
    
    t_index = 0
    jd = times[t_index]
    utc = Time(jd, format='jd').datetime
    
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
    h = cm_active.ActiveData(at_date=jd)
    h.load_apriori()

    xlim = (np.min(freqs), np.max(freqs))
    ylim = (55, 85)

    fig, axes = plt.subplots(Yside, Nside, figsize=(16,Yside*3))

    ptitle = 1.92/(Yside*3)
    fig.suptitle("JD = {0}, time = {1} UTC".format(jd, utc), fontsize=10,y=1+ptitle)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.subplots_adjust(left=.1, bottom=.1, right=.9, top=1, wspace=0.05, hspace=0.3)
    k = 0
    for i,n in enumerate(inclNodes):
        ants = nodes[n]['ants']
        j = 0
        for _,a in enumerate(sorted_ants):
            if a not in ants:
                continue
            status = h.apriori[f'HH{a}:A'].status
            ax = axes[i,j]
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            px, = ax.plot(freqs, 10*np.log10(np.abs(uvdx.get_data((a, a))[t_index])), color='r', alpha=0.75, linewidth=1)
            py, = ax.plot(freqs, 10*np.log10(np.abs(uvdy.get_data((a, a))[t_index])), color='b', alpha=0.75, linewidth=1)
            ax.grid(False, which='both')
            abb = status_abbreviations[status]
            ax.set_title(f'{a} ({abb})', fontsize=10, backgroundcolor=status_colors[status])
            if k == 0:
                ax.legend([px, py], ['NN', 'EE'])
            if i == len(inclNodes)-1:
                [t.set_fontsize(10) for t in ax.get_xticklabels()]
                ax.set_xlabel('freq (MHz)', fontsize=10)
            else:
                ax.set_xticklabels([])
            if j!=0:
                ax.set_yticklabels([])
            else:
                [t.set_fontsize(10) for t in ax.get_yticklabels()]
                ax.set_ylabel(r'$10\cdot\log$(amp)', fontsize=10)
            j += 1
            k += 1
        for k in range(j,maxants):
            axes[i,k].axis('off')
        axes[i,maxants-1].annotate(f'Node {n}', (1.1,.3),xycoords='axes fraction',rotation=270)
    plt.show()
    plt.close()
    
def plot_wfs(uvd, pol, mean_sub=False, save=False, jd=''):
    amps = np.abs(uvd.data_array[:, :, :, pol].reshape(uvd.Ntimes, uvd.Nants_data, uvd.Nfreqs, 1))
    nodes, antDict, inclNodes = generate_nodeDict(uvd)
    ants = uvd.get_ants()
    sorted_ants = sort_antennas(uvd)
    freqs = (uvd.freq_array[0])*10**(-6)
    times = uvd.time_array
    lsts = uvd.lst_array*3.819719
    inds = np.unique(lsts,return_index=True)[1]
    lsts = [lsts[ind] for ind in sorted(inds)]
    maxants = 0
    polnames = ['xx','yy']
    for node in nodes:
        n = len(nodes[node]['ants'])
        if n>maxants:
            maxants = n
    
    Nants = len(ants)
    Nside = maxants
    Yside = len(inclNodes)
    
    t_index = 0
    jd = times[t_index]
    utc = Time(jd, format='jd').datetime
    
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
    h = cm_active.ActiveData(at_date=jd)
    h.load_apriori()
    ptitle = 1.92/(Yside*3)
    fig, axes = plt.subplots(Yside, Nside, figsize=(16,Yside*3))
    if pol == 0:
        fig.suptitle("North Polarization", fontsize=14, y=1+ptitle)
    else:
        fig.suptitle("East Polarization", fontsize=14, y=1+ptitle)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.subplots_adjust(left=0, bottom=.1, right=.9, top=1, wspace=0.1, hspace=0.3)
    vmin = 6.5
    vmax = 8

    for i,n in enumerate(inclNodes):
        ants = nodes[n]['ants']
        j = 0
        for _,a in enumerate(sorted_ants):
            if a not in ants:
                continue
            status = h.apriori[f'HH{a}:A'].status
            abb = status_abbreviations[status]
            ax = axes[i,j]
            dat = np.log10(np.abs(uvd.get_data(a,a,polnames[pol])))
            if mean_sub == True:
                ms = np.subtract(dat, np.nanmean(dat,axis=0))
                im = ax.imshow(ms, 
                           vmin = -0.07, vmax = 0.07, aspect='auto',interpolation='nearest')
            else:
                im = ax.imshow(dat, 
                               vmin = vmin, vmax = vmax, aspect='auto',interpolation='nearest')
            ax.set_title(f'{a} ({abb})', fontsize=10,backgroundcolor=status_colors[status])
            if i == len(inclNodes)-1:
                xticks = [int(i) for i in np.linspace(0,len(freqs)-1,3)]
                xticklabels = np.around(freqs[xticks],0)
                ax.set_xticks(xticks)
                ax.set_xticklabels(xticklabels)
                ax.set_xlabel('Freq (MHz)', fontsize=10)
                [t.set_rotation(70) for t in ax.get_xticklabels()]
            else:
                ax.set_xticklabels([])
            if j != 0:
                ax.set_yticklabels([])
            else:
                yticks = [int(i) for i in np.linspace(0,len(lsts)-1,6)]
                yticklabels = [np.around(lsts[ytick],1) for ytick in yticks]
                [t.set_fontsize(12) for t in ax.get_yticklabels()]
                ax.set_ylabel('Time(LST)', fontsize=10)
                ax.set_yticks(yticks)
                ax.set_yticklabels(yticklabels)
                ax.set_ylabel('Time(LST)', fontsize=10)
            j += 1
        for k in range(j,maxants):
            axes[i,k].axis('off')
        pos = ax.get_position()
        cbar_ax=fig.add_axes([0.91,pos.y0,0.01,pos.height])        
        cbar = fig.colorbar(im, cax=cbar_ax)
        cbar.set_label(f'Node {n}',rotation=270, labelpad=15)
    if save is True:
        plt.savefig(f'{jd}_mean_subtracted_per_node_{pol}.png',bbox_inches='tight',dpi=300)
    plt.show()
    plt.close()
    
    
def plot_mean_subtracted_wfs(uvd, use_ants, jd, pols=['xx','yy']):
    freqs = (uvd.freq_array[0])*1e-6
    times = uvd.time_array
    lsts = uvd.lst_array*3.819719
    inds = np.unique(lsts,return_index=True)[1]
    lsts = [lsts[ind] for ind in sorted(inds)]
    ants = sorted(use_ants)
    Nants = len(ants) 
    pol_labels = ['NN','EE']
    
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
    h = cm_active.ActiveData(at_date=jd)
    h.load_apriori()
    
    fig, axes = plt.subplots(Nants, 2, figsize=(7,Nants*2.2))
    fig.suptitle('Mean Subtracted Waterfalls')
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.subplots_adjust(left=.1, bottom=.1, right=.85, top=.975, wspace=0.05, hspace=0.2)

    for i,ant in enumerate(ants):
        status = h.apriori[f'HH{ant}:A'].status
        abb = status_abbreviations[status]
        color = status_colors[status]
        for j,pol in enumerate(pols):
            ax = axes[i,j]
            dat = np.log10(np.abs(uvd.get_data(ant,ant,pol)))
            ms = np.subtract(dat, np.nanmean(dat,axis=0))
            im = ax.imshow(ms, 
                           vmin = -0.07, vmax = 0.07, aspect='auto',interpolation='nearest')
            ax.set_title(f'{ant} - {pol_labels[j]} ({abb})', fontsize=10, backgroundcolor=color)
            if j != 0:
                ax.set_yticklabels([])
            else:
                yticks = [int(i) for i in np.linspace(0,len(lsts)-1,6)]
                yticklabels = [np.around(lsts[ytick],1) for ytick in yticks]
                [t.set_fontsize(12) for t in ax.get_yticklabels()]
                ax.set_ylabel('Time(LST)', fontsize=10)
                ax.set_yticks(yticks)
                ax.set_yticklabels(yticklabels)
            if i != Nants-1:
                ax.set_xticklabels([])
            else:
                xticks = [int(i) for i in np.linspace(0,len(freqs)-1,8)]
                xticklabels = np.around(freqs[xticks],0)
                ax.set_xticks(xticks)
                ax.set_xticklabels(xticklabels)
                ax.set_xlabel('Frequency (MHz)', fontsize=10)
        if j == 1:
            pos = ax.get_position()
            cbar_ax=fig.add_axes([0.88,pos.y0,0.02,pos.height])
            fig.colorbar(im, cax=cbar_ax)
    fig.show()

def plot_closure(uvd, triad_length, pol):
    """Plot closure phase for an example triad.
    Parameters
    ----------
    files : list of strings
        List of data filenames
    triad_length : float {14., 29.}
        Length of the triangle segment length. Must be 14 or 29.
    pol : str {xx, yy}
        Polarization to plot.
    Returns
    -------
    None
    """


    if triad_length == 14.:
        triad_list = [[0, 11, 12], [0, 1, 12], [1, 12, 13], [1, 2, 13],
                      [2, 13, 14], [11, 23, 24], [11, 12, 24], [12, 24, 25],
                      [12, 13, 25], [13, 25, 26], [13, 14, 26], [14, 26, 27],
                      [23, 36, 37], [23, 24, 37], [24, 37, 38], [24, 25, 38],
                      [25, 38, 39], [25, 26, 39], [26, 39, 40], [26, 27, 40],
                      [27, 40, 41], [36, 37, 51], [37, 51, 52], [37, 38, 52],
                      [38, 52, 53], [38, 39, 53], [39, 53, 54], [39, 40, 54],
                      [40, 54, 55], [40, 41, 55], [51, 66, 67], [51, 52, 67],
                      [53, 54, 69], [54, 69, 70], [54, 55, 70], [55, 70, 71],
                      [65, 66, 82], [66, 82, 83], [66, 67, 83], [67, 83, 84],
                      [70, 71, 87], [120, 121, 140], [121, 140, 141], [121, 122, 141],
                      [122, 141, 142], [122, 123, 142], [123, 142, 143], [123, 124, 143]]
    else:
        triad_list = [[0, 23, 25], [0, 2, 25], [1, 24, 26], [2, 25, 27], [11, 36, 38],
                      [11, 13, 38], [12, 37, 39], [12, 14, 39], [13, 38, 40], [14, 39, 41],
                      [23, 25, 52], [24, 51, 53], [24, 26, 53], [25, 52, 54], [25, 27, 54],
                      [26, 53, 55], [36, 65, 67], [36, 38, 67], [38, 67, 69], [38, 40, 69],
                      [39, 41, 70], [40, 69, 71], [51, 82, 84], [51, 53, 84], [52, 83, 85],
                      [52, 54, 85], [54, 85, 87], [83, 85, 120], [85, 120, 122], [85, 87, 122],
                      [87, 122, 124]]


    # Look for a triad that exists in the data
    for triad in triad_list:
        bls = [[triad[0], triad[1]], [triad[1], triad[2]], [triad[2], triad[0]]]
        triad_in = True
        for bl in bls:
            inds = uvd.antpair2ind(bl[0], bl[1], ordered=False)
            if len(inds) == 0:
                triad_in = False
                break
        if triad_in:
            break

    if not triad_in:
        raise ValueError('Could not find triad in data.')

    closure_ph = np.angle(uvd.get_data(triad[0], triad[1], pol)
                          * uvd.get_data(triad[1], triad[2], pol)
                          * uvd.get_data(triad[2], triad[0], pol))
    plt.imshow(closure_ph, aspect='auto', rasterized=True,
                           interpolation='nearest', cmap = 'twilight')
    
def plotNodeAveragedSummary(uv,HHfiles,jd,use_ants,pols=['xx','yy'],mat_pols=['xx','yy'],
                            baseline_groups=[],removeBadAnts=False,plotRatios=False,plotSummary=True):
    """
    Plots a summary of baseline correlations throughout a night for each baseline group specified, separated into inter-node and intra-node baselines, for each polarization specified.
    
    Parameters
    ----------
    uv: UVData object
        UVData object containing any file from the desired night of observation.
    HHfiles: List
        A list of all files to be looked at for the desired night of observation.
    jd: String
        The JD of the night of observation
    pols: List
        A list containing the desired polarizations to look at. Options are any polarization strings accepted by pyuvdata. 
    baseline_groups: []
        A list containing the baseline types to look at, formatted as (length, N-S separation, label (str)).
    removeBadAnts: Bool
        Option to flag seemingly dead antennas and remove them from the per-baseline-group averaging. 
    
    Returns
    -------
    badAnts: List
        A list specifying the antennas flagged as dead or non-correlating.
    """
    if baseline_groups == []:
        baseline_groups = [(14,0,'14m E-W'),(14,-11,'14m NW-SE'),(14,11,'14m SW-NE'),(29,0,'29m E-W'),(29,22,'29m SW-NE'),
                       (44,0,'44m E-W'),(58.5,0,'58m E-W'),(73,0,'73m E-W'),(87.6,0,'88m E-W'),
                      (102.3,0,'102m E-W')]
    nodeMedians,lsts,badAnts=get_correlation_baseline_evolutions(uv,HHfiles,jd,use_ants,pols=pols,mat_pols=mat_pols,
                                                                bl_type=baseline_groups,removeBadAnts=removeBadAnts,
                                                                plotRatios=plotRatios)
    pols = mat_pols
    if plotSummary is False:
        return badAnts
    if len(lsts)>1:
        fig,axs = plt.subplots(len(pols),2,figsize=(16,16))
        maxLength = 0
        cmap = plt.get_cmap('Blues')
        for group in baseline_groups:
            if group[0] > maxLength:
                maxLength = group[0]
        for group in baseline_groups:
            length = group[0]
            data = nodeMedians[group[2]]
            colorInd = float(length/maxLength)
            if len(data['inter']['xx']) == 0:
                continue
            for i in range(len(pols)):
                pol = pols[i]
                axs[i][0].plot(data['inter'][pol], color=cmap(colorInd), label=group[2])
                axs[i][1].plot(data['intra'][pol], color=cmap(colorInd), label=group[2])
                axs[i][0].set_ylabel('Median Correlation Metric')
                axs[i][0].set_title('Internode, Polarization %s' % pol)
                axs[i][1].set_title('Intranode, Polarization %s' % pol)
                xticks = np.arange(0,len(lsts),1)
                axs[i][0].set_xticks(xticks)
                axs[i][0].set_xticklabels([str(lst) for lst in lsts])
                axs[i][1].set_xticks(xticks)
                axs[i][1].set_xticklabels([str(lst) for lst in lsts])
        axs[1][1].legend()
        axs[1][0].set_xlabel('LST (hours)')
        axs[1][1].set_xlabel('LST (hours)')
        fig.tight_layout(pad=2)
    else:
        print('#############################################################################')
        print('Not enough LST coverage to show metric evolution - that plot is being skipped')
        print('#############################################################################')
    return badAnts
    
def plotVisibilitySpectra(file,jd,use_ants='auto',badAnts=[],pols=['xx','yy']):
    """
    Plots visibility amplitude spectra for a set of redundant baselines, labeled by inter vs. intranode baselines.
    
    Parameters
    ---------
    file: String
        File to calculate the spectra from
    jd: String
        JD of the night 'file' was observed on
    badAnts: List
        A list of antennas not to include in the plot
    pols: List
        Polarizations to plot. Can include any polarization strings accepted by pyuvdata.
    """
    
    pol_labels = ['NS','EW']
    plt.subplots_adjust(wspace=0.25)
    uv = UVData()
    uv.read_uvh5(file)
    h = cm_hookup.Hookup()
    x = h.get_hookup('HH')
    baseline_groups = get_baseline_groups(uv,use_ants="auto")
    freqs = uv.freq_array[0]/1000000
    loc = EarthLocation.from_geocentric(*uv.telescope_location, unit='m')
    obstime_start = Time(uv.time_array[0],format='jd',location=loc)
    startTime = obstime_start.sidereal_time('mean').hour
    JD = int(obstime_start.jd)
    j = 0
    fig, axs = plt.subplots(len(baseline_groups),2,figsize=(12,4*len(baseline_groups)))
    for orientation in baseline_groups:
        bls = baseline_groups[orientation]
        usable = 0
        for i in range(len(bls)):
            ants = uv.baseline_to_antnums(bls[i])
            if ants[0] in badAnts or ants[1] in badAnts:
                continue
            if ants[0] in use_ants and ants[1] in use_ants:
                usable += 1
        if usable <=4:
            use_all = True
            print(f'Note: not enough baselines of orientation {orientation} - using all available baselines')
        elif usable <= 10:
            print(f'Note: only a small number of baselines of orientation {orientation} are available')
            use_all = False
        else:
            use_all = False
        for p in range(len(pols)):
            inter=False
            intra=False
            pol = pols[p]
            for i in range(len(bls)):
                ants = uv.baseline_to_antnums(bls[i])
                ant1 = ants[0]
                ant2 = ants[1]
                if (ant1 in use_ants and ant2 in use_ants) or use_all == True:
                    key1 = 'HH%i:A' % (ant1)
                    n1 = x[key1].get_part_from_type('node')['E<ground'][1:]
                    key2 = 'HH%i:A' % (ant2)
                    n2 = x[key2].get_part_from_type('node')['E<ground'][1:]
                    dat = np.mean(np.abs(uv.get_data(ant1,ant2,pol)),0)
                    auto1 = np.mean(np.abs(uv.get_data(ant1,ant1,pol)),0)
                    auto2 = np.mean(np.abs(uv.get_data(ant2,ant2,pol)),0)
                    norm = np.sqrt(np.multiply(auto1,auto2))
                    dat = np.divide(dat,norm)
                    if ant1 in badAnts or ant2 in badAnts:
                        continue
                    if n1 == n2:
                        if intra is False:
                            axs[j][p].plot(freqs,dat,color='blue',label='intranode')
                            intra=True
                        else:
                            axs[j][p].plot(freqs,dat,color='blue')
                    else:
                        if inter is False:
                            axs[j][p].plot(freqs,dat,color='red',label='internode')
                            inter=True
                        else:
                            axs[j][p].plot(freqs,dat,color='red')
                    axs[j][p].set_yscale('log')
                    axs[j][p].set_title('%s: %s pol' % (orientation,pol_labels[p]))
                    if j == 0:
                        axs[len(baseline_groups)-1][p].set_xlabel('Frequency (MHz)')
            if p == 0:
                axs[j][p].legend()
        axs[j][0].set_ylabel('log(|Vij|)')
        axs[j][1].set_yticks([])
        j += 1
    fig.suptitle('Visibility spectra (JD: %i)' % (JD))
    fig.subplots_adjust(top=.94,wspace=0.05)
    plt.show()
    plt.close()
    
def plot_antenna_positions(uv, badAnts={},flaggedAnts={},use_ants='auto'):
    """
    Plots the positions of all antennas that have data, colored by node.
    
    Parameters
    ----------
    uv: UVData object
        Observation to extract antenna numbers and positions from
    badAnts: List
        A list of flagged or bad antennas. These will be outlined in black in the plot. 
    flaggedAnts: Dict
        A dict of antennas flagged by ant_metrics with value corresponding to color in ant_metrics plot
    """
    
    plt.figure(figsize=(12,10))
    nodes, antDict, inclNodes = generate_nodeDict(uv)
    N = len(inclNodes)
    cmap = plt.get_cmap('tab20')
    i = 0
    nodePos = geo_sysdef.read_nodes()
    antPos = geo_sysdef.read_antennas()
    ants = geo_sysdef.read_antennas()
    nodes = geo_sysdef.read_nodes()
    firstNode = True
    for n, info in nodes.items():
        firstAnt = True
        if n > 9:
            n = str(n)
        else:
            n = f'0{n}'
        if n in inclNodes:
            color = cmap(round(20/N*i))
            i += 1
            for a in info['ants']:
                width = 0
                widthf = 0
                if a in badAnts:
                    width = 2
                if a in flaggedAnts.keys():
                    widthf = 6
                station = 'HH{}'.format(a)
                try:
                    this_ant = ants[station]
                except KeyError:
                    continue
                x = this_ant['E']
                y = this_ant['N']
                if a in use_ants:
                    falpha = 0.5
                else:
                    falpha = 0.1
                if firstAnt:
                    if a in badAnts or a in flaggedAnts.keys():
                        if falpha == 0.1:
                            plt.plot(x,y,marker="h",markersize=40,color=color,alpha=falpha,
                                markeredgecolor='black',markeredgewidth=0)
                            plt.annotate(a, [x-1, y])
                            continue
                        plt.plot(x,y,marker="h",markersize=40,color=color,alpha=falpha,label=str(n),
                            markeredgecolor='black',markeredgewidth=0)
                        if a in flaggedAnts.keys():
                            plt.plot(x,y,marker="h",markersize=40,color=color,
                                markeredgecolor=flaggedAnts[a],markeredgewidth=widthf, markerfacecolor="None")
                        if a in badAnts:
                            plt.plot(x,y,marker="h",markersize=40,color=color,
                                markeredgecolor='black',markeredgewidth=width, markerfacecolor="None")
                    else:
                        if falpha == 0.1:
                            plt.plot(x,y,marker="h",markersize=40,color=color,alpha=falpha,
                                markeredgecolor='black',markeredgewidth=0)
                            plt.annotate(a, [x-1, y])
                            continue
                        plt.plot(x,y,marker="h",markersize=40,color=color,alpha=falpha,label=str(n),
                            markeredgecolor='black',markeredgewidth=width)
                    firstAnt = False
                else:
                    plt.plot(x,y,marker="h",markersize=40,color=color,alpha=falpha,
                        markeredgecolor='black',markeredgewidth=0)
                    if a in flaggedAnts.keys() and a in use_ants:
                        plt.plot(x,y,marker="h",markersize=40,color=color,
                            markeredgecolor=flaggedAnts[a],markeredgewidth=widthf, markerfacecolor="None")
                    if a in badAnts and a in use_ants:
                        plt.plot(x,y,marker="h",markersize=40,color=color,
                            markeredgecolor='black',markeredgewidth=width, markerfacecolor="None")
                plt.annotate(a, [x-1, y])
            if firstNode:
                plt.plot(info['E'], info['N'], '*', color='gold',markersize=20,label='Node Box',
                        markeredgecolor='k',markeredgewidth=1)
                firstNode = False
            else:
                plt.plot(info['E'], info['N'], '*', color='gold',markersize=20,markeredgecolor='k',markeredgewidth=1)
    plt.legend(title='Node Number',bbox_to_anchor=(1.15,0.9),markerscale=0.5,labelspacing=1.5)
    plt.xlabel('East')
    plt.ylabel('North')
    plt.show()
    plt.close()
    
def plot_lst_coverage(uvd):
    """
    Plots the LST and JD coverage for a particular night.
    
    Parameters
    ----------
    uvd: UVData Object
        Object containing a whole night of data, used to extract the time array.
    """
    lsts = uvd.lst_array*3.819719
    jds = np.unique(uvd.time_array)
    alltimes = np.arange(np.floor(jds[0]),np.ceil(jds[0]),jds[2]-jds[1])
    df = jds[2]-jds[1]
    truetimes = [np.min(np.abs(jds-jd))<=df*0.6 for jd in alltimes]
    usetimes = np.tile(np.asarray(truetimes),(20,1))

    fig = plt.figure(figsize=(20,2))
    ax = fig.add_subplot()
    im = ax.imshow(usetimes, aspect='auto',cmap='RdYlGn',vmin=0,vmax=1,interpolation='nearest')
    fig.colorbar(im)
    ax.set_yticklabels([])
    ax.set_yticks([])
    if len(alltimes) <= 15:
        xticks = [int(i) for i in np.linspace(0,len(alltimes)-1,len(alltimes))]
    else:
        xticks = [int(i) for i in np.linspace(0,len(alltimes)-1,14)]
    ax.set_xticks(xticks)
    ax.set_xticklabels(np.around(alltimes[xticks],2))
    ax.set_xlabel('JD')
    ax.set_title('LST (hours)')
    ax2 = ax.twiny()
    ax2.set_xticks(xticks)
    jds = alltimes[xticks]
    lstlabels = []
    loc = EarthLocation.from_geocentric(*uvd.telescope_location, unit='m')
    for jd in jds:
        t = Time(jd,format='jd',location=loc)
        lstlabels.append(t.sidereal_time('mean').hour)
    ax2.set_xticklabels(np.around(lstlabels,2))
    ax2.set_label('LST (hours)')
    ax2.tick_params(labelsize=12)
    plt.show()
    plt.close()
    
def plotEvenOddWaterfalls(uvd_sum, uvd_diff):
    """Plot Even/Odd visibility ratio waterfall.
    Parameters
    ----------
    uvd_sum : UVData Object
        Object containing autos from sum files
    uvd_diff : UVData Object
        Object containing autos from diff files
    Returns
    -------
    None
    """
    nants = len(uvd_sum.get_ants())
    freqs = uvd_sum.freq_array[0]*1e-6
    nfreqs = len(freqs)
    lsts = np.unique(uvd_sum.lst_array*3.819719)
    sm = np.abs(uvd_sum.data_array[:,0,:,0])
    df = np.abs(uvd_diff.data_array[:,0,:,0])
    sm = np.r_[sm, np.nan + np.zeros((-len(sm) % nants,len(freqs)))]
    sm = np.nanmean(sm.reshape(-1,nants,nfreqs), axis=1)
    df = np.r_[df, np.nan + np.zeros((-len(df) % nants,len(freqs)))]
    df = np.nanmean(df.reshape(-1,nants,nfreqs), axis=1)

    evens = (sm + df)/2
    odds = (sm - df)/2
    rat = np.divide(evens,odds)
    rat = np.nan_to_num(rat)
    fig = plt.figure(figsize=(14,3))
    ax = fig.add_subplot()
    my_cmap = copy.deepcopy(matplotlib.cm.get_cmap('viridis'))
    my_cmap.set_under('r')
    my_cmap.set_over('r')
    im = plt.imshow(rat,aspect='auto',vmin=0.5,vmax=2,cmap=my_cmap,interpolation='nearest')
    fig.colorbar(im)
    ax.set_title('Even/odd Visibility Ratio')
    ax.set_xlabel('Frequency (MHz)')
    ax.set_ylabel('Time (LST)')
    yticks = [int(i) for i in np.linspace(len(lsts)-1,0, 4)]
    ax.set_yticks(yticks)
    ax.set_yticklabels(np.around(lsts[yticks], 1))
    xticks = [int(i) for i in np.linspace(0,len(freqs)-1, 10)]
    ax.set_xticks(xticks)
    ax.set_xticklabels(np.around(freqs[xticks], 0))
    i = 192
    while i < len(freqs):
        ax.axvline(i,color='w')
        i += 192
    plt.show()
    plt.close()
    return rat
    
def calcEvenOddAmpMatrix(sm,df,pols=['xx','yy'],nodes='auto', badThresh=0.25, plotRatios=False):
    """
    Calculates a matrix of phase correlations between antennas, where each pixel is calculated as (even/abs(even)) * (conj(odd)/abs(odd)), and then averaged across time and frequency.
    
    Paramters:
    ---------
    sm: UVData Object
        Sum observation.
    df: UVData Object
        Diff observation. Must be the same time of observation as sm. 
    pols: List
        Polarizations to plot. Can include any polarization strings accepted by pyuvdata.
    nodes: String or List
        Nodes to include in matrix. Default is 'auto', which generates a list of all nodes included in the provided data files. 
    badThresh: Float
        Threshold correlation metric value to use for flagging bad antennas.
    
    Returns:
    -------
    data: Dict
        Dictionary containing calculated values, formatted as data[polarization][ant1,ant2]. 
    badAnts: List
        List of antennas that were flagged as bad based on badThresh.
    """
    if sm.time_array[0] != df.time_array[0]:
        print('FATAL ERROR: Sum and diff files are not from the same observation!')
        return None
    if nodes=='auto':
        nodeDict, antDict, inclNodes = generate_nodeDict(sm)
    nants = len(sm.get_ants())
    data = {}
    antnumsAll = sort_antennas(sm)
    badAnts = []
    for p in range(len(pols)):
        pol = pols[p]
        data[pol] = np.empty((nants,nants))
        for i in range(len(antnumsAll)):
            thisAnt = []
            for j in range(len(antnumsAll)):
                ant1 = antnumsAll[i]
                ant2 = antnumsAll[j]
                s = sm.get_data(ant1,ant2,pol)
                d = df.get_data(ant1,ant2,pol)
                even = (s + d)/2
                even = np.divide(even,np.abs(even))
                odd = (s - d)/2
                odd = np.divide(odd,np.abs(odd))
                product = np.multiply(even,np.conj(odd))
                data[pol][i,j] = np.abs(np.mean(product))
                thisAnt.append(np.abs(np.mean(product)))
            pgood = np.count_nonzero(~np.isnan(thisAnt))/len(thisAnt)
            if (np.nanmedian(thisAnt) < badThresh or pgood<0.2) and antnumsAll[i] not in badAnts:
                if pol[0]==pol[1]:
                    #Don't assign bad ants based on cross pols
                    badAnts.append(antnumsAll[i])
    if plotRatios is True:
        if len(pols) == 4:
            data['xx-xy'] = np.subtract(data['xx'],data['xy'])
            data['xx-yx'] = np.subtract(data['xx'],data['yx'])
            data['yy-xy'] = np.subtract(data['yy'],data['xy'])
            data['yy-yx'] = np.subtract(data['yy'],data['yx'])
        else:
            print('Can only calculate differences if cross pols were specified')
        polAnts = {}
        badAnts = []
        subs = ['xx-xy','xx-yx','yy-xy','yy-yx']
        for k in subs:
            for i,ant in enumerate(antnumsAll):
                dat = data[k][i,:]
                if np.nanmedian(dat) < 0:
                    if ant in polAnts.keys():
                        polAnts[ant] = polAnts[ant] + 1
                    else:
                        polAnts[ant] = 1
                    if polAnts[ant] == 4:
                        badAnts.append(ant)  
    return data, badAnts


def plotCorrMatrix(uv,data,pols=['xx','yy'],vminIn=0,vmaxIn=1,nodes='auto',logScale=False,plotRatios=False):
    """
    Plots a matrix representing the phase correlation of each baseline.
    
    Parameters:
    ----------
    uv: UVData Object
        Observation used for calculating the correlation metric
    data: Dict
        Dictionary containing the correlation metric for each baseline and each polarization. Formatted as data[polarization]  [ant1,ant2] 
    pols: List
        Polarizations to plot. Can include any polarization strings accepted by pyuvdata.
    vminIn: float
        Lower limit of colorbar. Default is 0.
    vmaxIn: float
        Upper limit of colorbar. Default is 1.
    nodes: Dict
        Dictionary containing the nodes (and their constituent antennas) to include in the matrix. Formatted as nodes[Node #][Ant List, Snap # List, Snap Location List].
    logScale: Bool
        Option to put colormap on a logarithmic scale. Default is False.
    """
    if nodes=='auto':
        nodeDict, antDict, inclNodes = generate_nodeDict(uv)
    nantsTotal = len(uv.get_ants())
    power = np.empty((nantsTotal,nantsTotal))
    fig, axs = plt.subplots(2,2,figsize=(16,16))
    dirs = ['NN','EE','NE','EN']
    cmap='plasma'
    if plotRatios is True:
        pols = ['xx-xy','yy-xy','xx-yx','yy-yx']
        dirs=pols
        vminIn=-1
        cmap='seismic'
    loc = EarthLocation.from_geocentric(*uv.telescope_location, unit='m')
    jd = uv.time_array[0]
    t = Time(jd,format='jd',location=loc)
    lst = round(t.sidereal_time('mean').hour,2)
    t.format='fits'
    antnumsAll = sort_antennas(uv)
    i = 0
    for p in range(len(pols)):
        if p >= 2:
            i=1
        pol = pols[p]
        nants = len(antnumsAll)
        if logScale is True:
            im = axs[i][p%2].imshow(data[pol],cmap=cmap,origin='upper',extent=[0.5,nantsTotal+.5,0.5,nantsTotal+0.5],norm=LogNorm(vmin=vminIn, vmax=vmaxIn))
        else:
            im = axs[i][p%2].imshow(data[pol],cmap=cmap,origin='upper',extent=[0.5,nantsTotal+.5,0.5,nantsTotal+0.5],vmin=vminIn, vmax=vmaxIn)
        axs[i][p%2].set_xticks(np.arange(0,nantsTotal)+1)
        axs[i][p%2].set_xticklabels(antnumsAll,rotation=90,fontsize=6)
        axs[i][p%2].xaxis.set_ticks_position('top')
        axs[i][p%2].set_title('polarization: ' + dirs[p] + '\n')
        n=0
        for node in sorted(inclNodes):
            n += len(nodeDict[node]['ants'])
            axs[i][p%2].axhline(len(antnumsAll)-n+.5,lw=4)
            axs[i][p%2].axvline(n+.5,lw=4)
            axs[i][p%2].text(n-len(nodeDict[node]['ants'])/2,-.5,node)
        axs[i][p%2].text(.42,-.05,'Node Number',transform=axs[i][p%2].transAxes)
    n=0
    for node in sorted(inclNodes):
        n += len(nodeDict[node]['ants'])
        axs[0][1].text(nantsTotal+1,nantsTotal-n+len(nodeDict[node]['ants'])/2,node)
        axs[1][1].text(nantsTotal+1,nantsTotal-n+len(nodeDict[node]['ants'])/2,node)
    axs[0][1].text(1.05,0.4,'Node Number',rotation=270,transform=axs[0][1].transAxes)
    axs[0][1].set_yticklabels([])
    axs[0][1].set_yticks([])
    axs[0][0].set_yticks(np.arange(nantsTotal,0,-1))
    axs[0][0].set_yticklabels(antnumsAll,fontsize=6)
    axs[0][0].set_ylabel('Antenna Number')
    axs[1][1].text(1.05,0.4,'Node Number',rotation=270,transform=axs[1][1].transAxes)
    axs[1][1].set_yticklabels([])
    axs[1][1].set_yticks([])
    axs[1][0].set_yticks(np.arange(nantsTotal,0,-1))
    axs[1][0].set_yticklabels(antnumsAll,fontsize=6)
    axs[1][0].set_ylabel('Antenna Number')
    cbar_ax = fig.add_axes([0.98,0.18,0.015,0.6])
    cbar_ax.set_xlabel('|V|', rotation=0)
    cbar = fig.colorbar(im, cax=cbar_ax)
    fig.suptitle('Correlation Matrix - JD: %s, LST: %.0fh' % (str(jd),np.round(lst,0)))
    fig.subplots_adjust(top=1.28,wspace=0.05,hspace=1.1)
    fig.tight_layout(pad=2)
    plt.show()
    plt.close()
    
def plot_single_matrix(uv,data,vminIn=0,vmaxIn=1,nodes='auto',logScale=False):
    if nodes=='auto':
        nodeDict, antDict, inclNodes = generate_nodeDict(uv)
    nantsTotal = len(uv.get_ants())
    power = np.empty((nantsTotal,nantsTotal))
    fig, axs = plt.subplots(1,1,figsize=(16,16))
    loc = EarthLocation.from_geocentric(*uv.telescope_location, unit='m')
    jd = uv.time_array[0]
    t = Time(jd,format='jd',location=loc)
    lst = round(t.sidereal_time('mean').hour,2)
    t.format='fits'
    antnumsAll = sort_antennas(uv)
    nants = len(antnumsAll)
    if logScale is True:
        im = axs[0][0].imshow(data[pol],cmap='plasma',origin='upper',
                                extent=[0.5,nantsTotal+.5,0.5,nantsTotal+0.5],norm=LogNorm(vmin=vminIn, vmax=vmaxIn))
    else:
        im = axs[0][0].imshow(data[pol],cmap='plasma',origin='upper',extent=
                                [0.5,nantsTotal+.5,0.5,nantsTotal+0.5],vmin=vminIn, vmax=vmaxIn)
    axs[0][0].set_xticks(np.arange(0,nantsTotal)+1)
    axs[0][0].set_xticklabels(antnumsAll,rotation=90,fontsize=6)
    axs[0][0].xaxis.set_ticks_position('top')
    axs[0][0].set_title('polarization: ' + dirs[p] + '\n')
    n=0
    for node in sorted(inclNodes):
        n += len(nodeDict[node]['ants'])
        axs[0][0].axhline(len(antnumsAll)-n+.5,lw=4)
        axs[0][0].axvline(n+.5,lw=4)
        axs[0][0].text(n-len(nodeDict[node]['ants'])/2,-.5,node)
    axs[0][0].text(.42,-.05,'Node Number',transform=axs[0][0].transAxes)
    n=0
    for node in sorted(inclNodes):
        n += len(nodeDict[node]['ants'])
        axs[0][0].text(nantsTotal+1,nantsTotal-n+len(nodeDict[node]['ants'])/2,node)
    axs[0][0].text(1.05,0.4,'Node Number',rotation=270,transform=axs[0][0].transAxes)
    axs[0][0].set_yticks(np.arange(nantsTotal,0,-1))
    axs[0][0].set_yticklabels(antnumsAll,fontsize=6)
    axs[0][0].set_ylabel('Antenna Number')
    axs[0][0].text(1.05,0.4,'Node Number',rotation=270,transform=axs[0][0].transAxes)
    cbar_ax = fig.add_axes([0.98,0.18,0.015,0.6])
    cbar_ax.set_xlabel('|V|', rotation=0)
    cbar = fig.colorbar(im, cax=cbar_ax)
    fig.suptitle('Correlation Matrix - JD: %s, LST: %.0fh' % (str(jd),np.round(lst,0)))
    fig.subplots_adjust(top=1.28,wspace=0.05,hspace=1.1)
    fig.tight_layout(pad=2)
    plt.show()
    plt.close()
    
def get_hourly_files(uv, HHfiles, jd):
    """
    Generates a list of files spaced one hour apart throughout a night of observation, and the times those files were observed.
    
    Parameters:
    ----------
    uv: UVData Object
        Sample observation from the given night, used only for grabbing the telescope location
    HHFiles: List
        List of all files from the desired night of observation
    jd: String
        JD of the night of observation
        
    Returns:
    -------
    use_files: List
        List of files separated by one hour
    use_lsts: List
        List of LSTs of observations in use_files
    """
    use_lsts = []
    use_files = []
    use_file_inds = []
    loc = EarthLocation.from_geocentric(*uv.telescope_location, unit='m')
    for i,file in enumerate(HHfiles):
        try:
            dat = UVData()
            dat.read(file, read_data=False)
        except KeyError:
            continue
        jd = dat.time_array[0]
        t = Time(jd,format='jd',location=loc)
        lst = round(t.sidereal_time('mean').hour,2)
        if np.round(lst,0) == 24:
            continue
        if np.abs((lst-np.round(lst,0)))<0.05:
            if len(use_lsts)>0 and np.abs(use_lsts[-1]-lst)<0.5:
                if np.abs((lst-np.round(lst,0))) < abs((use_lsts[-1]-np.round(lst,0))):
                    use_lsts[-1] = lst
                    use_files[-1] = file
                    use_file_inds[-1] = i
            else:
                use_lsts.append(lst)
                use_files.append(file)
                use_file_inds.append(i)
    return use_files, use_lsts, use_file_inds

def get_baseline_groups(uv, bl_groups=[(14,0,'14m E-W'),(29,0,'29m E-W'),(14,-11,'14m NW-SE'),(14,11,'14m SW-NE')],
                       use_ants='auto'):
    """
    Generate dictionary containing baseline groups.
    
    Parameters:
    ----------
    uv: UVData Object
        Observation to extract antenna position information from
    bl_groups: List
        Desired baseline types to extract, formatted as (length (float), N-S separation (float), label (string))
        
    Returns:
    --------
    bls: Dict
        Dictionary containing list of lists of redundant baseline numbers, formatted as bls[group label]
    """
    
    bls={}
    baseline_groups,vec_bin_centers,lengths = uv.get_redundancies(use_antpos=True,include_autos=False)
    for i in range(len(baseline_groups)):
        bl = baseline_groups[i]
        for group in bl_groups:
            if np.abs(lengths[i]-group[0])<1:
                ant1 = uv.baseline_to_antnums(bl[0])[0]
                ant2 = uv.baseline_to_antnums(bl[0])[1]
                if use_ants == 'auto' or (ant1 in use_ants and ant2 in use_ants):
                    antPos1 = uv.antenna_positions[np.argwhere(uv.antenna_numbers == ant1)]
                    antPos2 = uv.antenna_positions[np.argwhere(uv.antenna_numbers == ant2)]
                    disp = (antPos2-antPos1)[0][0]
                    if np.abs(disp[2]-group[1])<0.5:
                        bls[group[2]] = bl
    return bls


    
def get_correlation_baseline_evolutions(uv,HHfiles,jd,use_ants='auto',badThresh=0.35,pols=['xx','yy'],bl_type=(14,0,'14m E-W'),
                                        removeBadAnts=False, plotMatrix=True,mat_pols=['xx','yy','xy','yx'],plotRatios=False):
    """
    Calculates the average correlation metric for a set of redundant baseline groups at one hour intervals throughout a night of observation.
    
    Parameters:
    ----------
    uv: UVData Object
        Sample observation from the desired night, used only for getting telescope location information.
    HHfiles: List
        List of all files for a night of observation
    jd: String
        JD of the given night of observation
    badThresh: Float
        Threshold correlation metric value to use for flagging bad antennas. Default is 0.35.
    pols: List
        Polarizations to plot. Can include any polarization strings accepted by pyuvdata.
    bl_type: Tuple
        Redundant baseline group to calculate correlation metric for. Default is 14m E-W baselines
    removeBadAnts: Bool
        Option to exclude antennas marked as bad from calculation. Default is False.
    plotMatrix: Bool
        Option to plot the correlation matrix for observations once each hour. Default is True.
        
    Returns:
    -------
    result: Dict
        Per hour correlation metric, formatted as result[baseline type]['inter' or 'intra'][polarization]
    lsts: List
        LSTs that metric was calculated for, spaced 1 hour apart.
    bad_antennas: List
        Antenna numbers flagged as bad based on badThresh parameter.
    """
    files, lsts, inds = get_hourly_files(uv, HHfiles, jd)
    if use_ants == 'auto':
        use_ants = uv.get_ants()
    if plotRatios is True:
        files = [files[len(files)//2]]
        nTimes=1
    else:
        nTimes = len(files)
    if nTimes > 3:
        plotTimes = [0,nTimes//2,nTimes-1]
    else:
        plotTimes = np.arange(0,nTimes,1)
    nodeDict, antDict, inclNodes = generate_nodeDict(uv)
    JD = math.floor(uv.time_array[0])
    bad_antennas = []
    pols = mat_pols
    corrSummary = generateDataTable(uv,pols=pols)
    result = {}
    for f in range(nTimes):
        file = files[f]
        ind = inds[f]
        sm = UVData()
        df = UVData()
        try:
#             print(f'Trying to read {file}')
            sm.read(file, skip_bad_files=True, antenna_nums=use_ants)
            dffile = '%sdiff%s' % (file[0:-8],file[-5:])
            df.read(dffile, skip_bad_files=True, antenna_nums=use_ants)
        except:
            i = -5
            read = False
            while i<5 and read==False:
                try:
                    file = HHfiles[ind+i]
#                     print(f'trying to read {file}')
                    sm.read(file, skip_bad_files=True, antenna_nums=use_ants)
                    dffile = '%sdiff%s' % (file[0:-8],file[-5:])
                    df.read(dffile, skip_bad_files=True, antenna_nums=use_ants)
                    read = True
                except:
                    i += 1
            if read == False:
                print(f'WARNING: unable to read {file}')
                continue
        matrix, badAnts = calcEvenOddAmpMatrix(sm,df,nodes='auto',pols=mat_pols,badThresh=badThresh,plotRatios=plotRatios)
        if plotMatrix is True and f in plotTimes:
            plotCorrMatrix(sm, matrix, pols=mat_pols, nodes='auto',plotRatios=plotRatios)
        for group in bl_type:
            medians = {
                'inter' : {},
                'intra' : {}
                }
            for pol in pols:
                medians['inter'][pol] = []
                medians['intra'][pol] = []
            if group[2] not in result.keys():
                result[group[2]] = {
                    'inter' : {},
                    'intra' : {}
                }
                for pol in pols:
                    result[group[2]]['inter'][pol] = []
                    result[group[2]]['intra'][pol] = []
            bls = get_baseline_type(uv,bl_type=group,use_ants=use_ants)
            if bls == None:
#                 print(f'No baselines of type {group}')
                continue
            baselines = [uv.baseline_to_antnums(bl) for bl in bls]
            for ant in badAnts:
                if ant not in bad_antennas:
                    bad_antennas.append(ant)
            if removeBadAnts is True:
                nodeInfo = {
                    'inter' : getInternodeMedians(sm,matrix,badAnts=bad_antennas, baselines=baselines,pols=pols),
                    'intra' : getIntranodeMedians(sm,matrix,badAnts=bad_antennas, baselines=baselines,pols=pols)
                }
            else:
                nodeInfo = {
                    'inter' : getInternodeMedians(sm,matrix, baselines=baselines,pols=pols),
                    'intra' : getIntranodeMedians(sm,matrix,baselines=baselines,pols=pols)
                }
            for node in nodeDict:
                for pol in pols:
                    corrSummary[node][pol]['inter'].append(nodeInfo['inter'][node][pol])
                    corrSummary[node][pol]['intra'].append(nodeInfo['intra'][node][pol])
                    medians['inter'][pol].append(nodeInfo['inter'][node][pol])
                    medians['intra'][pol].append(nodeInfo['intra'][node][pol])
            for pol in pols:
                result[group[2]]['inter'][pol].append(np.nanmedian(medians['inter'][pol]))
                result[group[2]]['intra'][pol].append(np.nanmedian(medians['intra'][pol]))
    return result,lsts,bad_antennas

def generateDataTable(uv,pols=['xx','yy']):
    """
    Simple helper function to generate an empty dictionary of the format desired for get_correlation_baseline_evolutions()
    
    Parameters:
    ----------
    uv: UVData Object
        Sample observation to extract node and antenna information from.
    pols: List
        Polarizations to plot. Can include any polarization strings accepted by pyuvdata. Default is ['xx','yy'].
        
    Returns:
    -------
    dataObject: Dict
        Empty dict formatted as dataObject[node #][polarization]['inter' or 'intra']
    """
    
    nodeDict, antDict, inclNodes = generate_nodeDict(uv)
    dataObject = {}
    for node in nodeDict:
        dataObject[node] = {}
        for pol in pols:
            dataObject[node][pol] = {
                'inter' : [],
                'intra' : []
            }
    return dataObject

def getInternodeMedians(uv,data,pols=['xx','yy'],badAnts=[],baselines='all'):
    """
    Identifies internode baseliens and performs averaging of correlation metric.
    
    Parameters:
    ----------
    uv: UVData Object
        Sample observation to extract node and antenna information from.
    data: Dict
        Dictionary containing correlation metric information, formatted as data[polarization][ant1,ant2].
    pols: List
        Polarizations to plot. Can include any polarization strings accepted by pyuvdata. Default is ['xx','yy'].
    badAnts: List
        List of antennas that have been flagged as bad - if provided, they will be excluded from averaging.
    baselines: List
        List of baseline types to include in calculation.
        
    Returns:
    -------
    nodeMeans: Dict
        Per-node averaged correlation metrics, formatted as nodeMeans[node #][polarization].
    """
    
    nodeDict, antDict, inclNodes = generate_nodeDict(uv)
    antnumsAll=sort_antennas(uv)
    nants = len(antnumsAll)
    nodeMeans = {}
    nodeCorrs = {}
    for node in nodeDict:
        nodeCorrs[node] = {}
        nodeMeans[node] = {}
        for pol in pols:
            nodeCorrs[node][pol] = []        
    start=0
    h = cm_hookup.Hookup()
    x = h.get_hookup('HH')
    for pol in pols:
        for i in range(nants):
            for j in range(nants):
                ant1 = antnumsAll[i]
                ant2 = antnumsAll[j]
                if ant1 not in badAnts and ant2 not in badAnts and ant1 != ant2:
                    if baselines=='all' or (ant1,ant2) in baselines:
                        key1 = 'HH%i:A' % (ant1)
                        n1 = x[key1].get_part_from_type('node')['E<ground'][1:]
                        key2 = 'HH%i:A' % (ant2)
                        n2 = x[key2].get_part_from_type('node')['E<ground'][1:]
                        dat = data[pol][i,j]
                        if n1 != n2:
                            nodeCorrs[n1][pol].append(dat)
                            nodeCorrs[n2][pol].append(dat)
    for node in nodeDict:
        for pol in pols:
            nodeMeans[node][pol] = np.nanmedian(nodeCorrs[node][pol])
    return nodeMeans

def getIntranodeMedians(uv, data, pols=['xx','yy'],badAnts=[],baselines='all'):
    """
    Identifies intranode baseliens and performs averaging of correlation metric.
    
    Parameters:
    ----------
    uv: UVData Object
        Sample observation to extract node and antenna information from.
    data: Dict
        Dictionary containing correlation metric information, formatted as data[polarization][ant1,ant2].
    pols: List
        Polarizations to plot. Can include any polarization strings accepted by pyuvdata. Default is ['xx','yy'].
    badAnts: List
        List of antennas that have been flagged as bad - if provided, they will be excluded from averaging.
    baselines: List
        List of baseline types to include in calculation.
        
    Returns:
    -------
    nodeMeans: Dict
        Per-node averaged correlation metrics, formatted as nodeMeans[node #][polarization].
    """
    
    nodeDict, antDict, inclNodes = generate_nodeDict(uv)
    antnumsAll=sort_antennas(uv)
    nodeMeans = {}
    start=0
    for node in nodeDict:
        nodeMeans[node]={}
        for pol in pols:
            nodeCorrs = []
            for i in range(start,start+len(nodeDict[node]['ants'])):
                for j in range(start,start+len(nodeDict[node]['ants'])):
                    ant1 = antnumsAll[i]
                    ant2 = antnumsAll[j]
                    if ant1 not in badAnts and ant2 not in badAnts and i != j:
                        if baselines=='all' or (ant1,ant2) in baselines:
                            nodeCorrs.append(data[pol][i,j])
            nodeMeans[node][pol] = np.nanmedian(nodeCorrs)
        start += len(nodeDict[node]['ants'])
    return nodeMeans

def get_baseline_type(uv,bl_type=(14,0,'14m E-W'),use_ants='auto'):
    """
    Parameters:
    ----------
    uv: UVData Object
        Sample observation to get baseline information from.
    bl_type: Tuple
        Redundant baseline group to extract baseline numbers for. Formatted as (length, N-S separation, label).
    
    Returns:
    -------
    bl: List
        List of lists of redundant baseline numbers. Returns None if the provided bl_type is not found.
    """
    
    baseline_groups,vec_bin_centers,lengths = uv.get_redundancies(use_antpos=True,include_autos=False)
    for i in range(len(baseline_groups)):
        bl = baseline_groups[i]
        if np.abs(lengths[i]-bl_type[0])<1:
            ant1 = uv.baseline_to_antnums(bl[0])[0]
            ant2 = uv.baseline_to_antnums(bl[0])[1]
            if (ant1 in use_ants and ant2 in use_ants) or use_ants == 'auto':
                antPos1 = uv.antenna_positions[np.argwhere(uv.antenna_numbers == ant1)]
                antPos2 = uv.antenna_positions[np.argwhere(uv.antenna_numbers == ant2)]
                disp = (antPos2-antPos1)[0][0]
                if np.abs(disp[2]-bl_type[1])<0.5:
                    return bl
    return None

def generate_nodeDict(uv):
    """
    Generates dictionaries containing node and antenna information.
    
    Parameters:
    ----------
    uv: UVData Object
        Sample observation to extract node and antenna information from.
    
    Returns:
    -------
    nodes: Dict
        Dictionary containing entry for all nodes, each of which has keys: 'ants', 'snapLocs', 'snapInput'.
    antDict: Dict
        Dictionary containing entry for all antennas, each of which has keys: 'node', 'snapLocs', 'snapInput'.
    inclNodes: List
        Nodes that have hooked up antennas.
    """
    
    antnums = uv.get_ants()
    h = cm_hookup.Hookup()
    x = h.get_hookup('HH')
    nodes = {}
    antDict = {}
    inclNodes = []
    for ant in antnums:
        key = 'HH%i:A' % (ant)
        n = x[key].get_part_from_type('node')['E<ground'][1:]
        snapLoc = (x[key].hookup['E<ground'][-1].downstream_input_port[-1], ant)
        snapInput = (x[key].hookup['E<ground'][-2].downstream_input_port[1:], ant)
        antDict[ant] = {}
        antDict[ant]['node'] = str(n)
        antDict[ant]['snapLocs'] = snapLoc
        antDict[ant]['snapInput'] = snapInput
        inclNodes.append(n)
        if n in nodes:
            nodes[n]['ants'].append(ant)
            nodes[n]['snapLocs'].append(snapLoc)
            nodes[n]['snapInput'].append(snapInput)
        else:
            nodes[n] = {}
            nodes[n]['ants'] = [ant]
            nodes[n]['snapLocs'] = [snapLoc]
            nodes[n]['snapInput'] = [snapInput]
    inclNodes = np.unique(inclNodes)
    return nodes, antDict, inclNodes

def sort_antennas(uv):
    """
    Helper function that sorts antennas by snap input number.
    
    Parameters:
    ----------
    uv: UVData Object
        Sample observation used for extracting node and antenna information.
        
    Returns:
    -------
    sortedAntennas: List
        All antennas with data, sorted into order of ascending node number, and within that by ascending snap number, and within that by ascending snap input number.
    """
    
    nodes, antDict, inclNodes = generate_nodeDict(uv)
    sortedAntennas = []
    for n in sorted(inclNodes):
        snappairs = []
        h = cm_hookup.Hookup()
        x = h.get_hookup('HH')
        for ant in nodes[n]['ants']:
            snappairs.append(antDict[ant]['snapLocs'])
        snapLocs = {}
        locs = []
        for pair in snappairs:
            ant = pair[1]
            loc = pair[0]
            locs.append(loc)
            if loc in snapLocs:
                snapLocs[loc].append(ant)
            else:
                snapLocs[loc] = [ant]
        locs = sorted(np.unique(locs))
        ants_sorted = []
        for loc in locs:
            ants = snapLocs[loc]
            inputpairs = []
            for ant in ants:
                key = 'HH%i:A' % (ant)
                pair = (int(x[key].hookup['E<ground'][-2].downstream_input_port[1:]), ant)
                inputpairs.append(pair)
            for _,a in sorted(inputpairs):
                ants_sorted.append(a)
        for ant in ants_sorted:
            sortedAntennas.append(ant)
    return sortedAntennas

def plot_crosses(uvd, ref_ant):
    ants = uvd.get_ants()
    freqs = (uvd.freq_array[0])*10**(-6)
    times = uvd.time_array
    lsts = uvd.lst_array
    
    Nants = len(ants)
#     Nside = int(np.ceil(np.sqrt(Nants)))*3
    Nside = 4
    Yside = int(np.ceil(float(Nants)/Nside))
    
    t_index = 0
    jd = times[t_index]
    utc = Time(jd, format='jd').datetime

    xlim = (np.min(freqs), np.max(freqs))
    ylim = (60, 90)

    fig, axes = plt.subplots(Yside, Nside, figsize=(Yside*2, Nside*60))

    fig.suptitle("JD = {0}, time = {1} UTC".format(jd, utc), fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.subplots_adjust(left=.1, bottom=.1, right=.9, top=.9, wspace=0.05, hspace=0.2)

    k = 0
    for i in range(Yside):
        for j in range(Nside):
            ax = axes[i,j]
            ax.set_xlim(xlim)
#             ax.set_ylim(ylim)
            if k < Nants:
                px, = ax.plot(freqs, 10*np.log10(np.abs(np.mean(uvd.get_data((ants[k], ref_ant, 'xx')),axis=0))), color='red', alpha=0.75, linewidth=1)
                py, = ax.plot(freqs, 10*np.log10(np.abs(np.mean(uvd.get_data((ants[k], ref_ant, 'yy')),axis=0))), color='darkorange', alpha=0.75, linewidth=1)
                pxy, = ax.plot(freqs, 10*np.log10(np.abs(np.mean(uvd.get_data((ants[k], ref_ant, 'xy')),axis=0))), color='royalblue', alpha=0.75, linewidth=1)
                pyx, = ax.plot(freqs, 10*np.log10(np.abs(np.mean(uvd.get_data((ants[k], ref_ant, 'yx')),axis=0))), color='darkviolet', alpha=0.75, linewidth=1)
            
                ax.grid(False, which='both')
                ax.set_title(str(ants[k]), fontsize=14)
            
                if k == 0:
                    ax.legend([px, py, pxy, pyx], ['XX', 'YY', 'XY','YX'])
            
            else:
                ax.axis('off')
            if j != 0:
                ax.set_yticklabels([])
            else:
                [t.set_fontsize(10) for t in ax.get_yticklabels()]
                ax.set_ylabel(r'$10\cdot\log_{10}$ amplitude', fontsize=10)
            if i != Yside-1:
                ax.set_xticklabels([])
            else:
                [t.set_fontsize(10) for t in ax.get_xticklabels()]
                ax.set_xlabel('freq (MHz)', fontsize=10)
            k += 1
    fig.show()
    plt.close()
    
def gather_source_list():
    sources = []
    sources.append((50.6750,-37.2083,'Fornax A'))
    sources.append((201.3667,-43.0192,'Cen A'))
    # sources.append((83.6333,22.0144,'Taurus A'))
    sources.append((252.7833,4.9925,'Hercules A'))
    sources.append((139.5250,-12.0947,'Hydra A'))
    sources.append((79.9583,-45.7789,'Pictor A'))
    sources.append((187.7042,12.3911,'Virgo A'))
    sources.append((83.8208,-59.3897,'Orion A'))
    sources.append((80.8958,-69.7561,'LMC'))
    sources.append((13.1875,-72.8286,'SMC'))
    sources.append((201.3667,-43.0192,'Cen A'))
    sources.append((83.6333,20.0144,'Crab Pulsar'))
    sources.append((128.8375,-45.1764,'Vela SNR'))
    cat_path = f'{DATA_PATH}/G4Jy_catalog.tsv'
    cat = open(cat_path)
    f = csv.reader(cat,delimiter='\n')
    for row in f:
        if len(row)>0 and row[0][0]=='J':
            s = row[0].split(';')
            tup = (float(s[1]),float(s[2]),'')
            sources.append(tup)
    return sources
    
def _clean_ds_per_bl_pol(bl, pol, uvd, uvd_diff, area, tol, skip_wgts, freq_range):
    """
    CLEAN function of delay spectra at given baseline and polarization.
    
    Parameters:
    ----------
    bl: Tuple
        Tuple of baseline (ant1, ant2)
    pol: String
        String of polarization
    uvd: UVData Object
        Sample observation from the desired night to compute delay spectra
    uvd_diff: UVData Object
        Diff of observation from the desired night to calculate even/odd visibilities and delay spectra
    area: Float
        The half-width (i.e. the width of the positive part) of the region in fourier space, symmetric about 0, that is filtered out in ns.
    tol: Float
        CLEAN algorithm convergence tolerance (see aipy.deconv.clean)
    skip_wgts: Float
        Skips filtering rows with very low total weight (unflagged fraction ~< skip_wgt). See uvtools.dspec.high_pass_fourier_filter for more details
    freq_range: Float
        Frequecy range for making delay spectra in MHz
        
    Returns:
    -------
    _data_cleaned_sq: Dict
        Square of CLEANed delay spectra, formatted as _data_cleaned_sq[(ant1, ant2, pol)]
    data_rs_sq: Dict
        Square of residual from CLEAN in frequency domain, formatted as data_rs_sq[(ant1, ant2, pol)]
    """
    key = (bl[0], bl[1], pol)
    freqs = uvd.freq_array[0]
    FM_idx = np.searchsorted(freqs*1e-6, [85,110])
    flag_FM = np.zeros(freqs.size, dtype=bool)
    flag_FM[FM_idx[0]:FM_idx[1]] = True

    freq_low, freq_high = np.sort(freq_range)
    idx_freqs = np.where(np.logical_and(freqs*1e-6 > freq_low, freqs*1e-6 < freq_high))[0]
    freqs = freqs[idx_freqs]
    win = dspec.gen_window('bh7', freqs.size)

    data = uvd.get_data(key)[:, idx_freqs]
    diff = uvd_diff.get_data(key)[:, idx_freqs]
    wgts = (~uvd.get_flags(key)*~flag_FM[np.newaxis,:])[:, idx_freqs].astype(float)

    idx_zero = np.where(np.abs(data) == 0)[0]
    if(len(idx_zero)/len(data) < 0.5):
        d_even = (data+diff)*0.5
        d_odd = (data-diff)*0.5
        d_even_cl, d_even_rs, _ = dspec.high_pass_fourier_filter(d_even, wgts, area*1e-9, freqs[1]-freqs[0], 
                                                                 tol=tol, skip_wgt=skip_wgts, window='bh7')
        d_odd_cl, d_odd_rs, _ = dspec.high_pass_fourier_filter(d_odd, wgts, area*1e-9, freqs[1]-freqs[0],
                                                               tol=tol, skip_wgt=skip_wgts, window='bh7')

        idx = np.where(np.mean(np.abs(d_even_cl), axis=1) == 0)[0]
        d_even_cl[idx] = np.nan
        d_even_rs[idx] = np.nan        
        idx = np.where(np.mean(np.abs(d_odd_cl), axis=1) == 0)[0]
        d_odd_cl[idx] = np.nan
        d_odd_rs[idx] = np.nan

        _d_even = np.fft.fftshift(np.fft.ifft((d_even_cl+d_even_rs)*win), axes=1)
        _d_odd = np.fft.fftshift(np.fft.ifft((d_odd_cl+d_odd_rs)*win), axes=1)
        _data_cleaned_sq = _d_odd.conj()*_d_even
        data_rs_sq = d_odd_rs.conj()*d_even_rs
    else:
        _data_cleaned_sq = np.zeros_like(data)
        data_rs_sq = np.zeros_like(data)

    return _data_cleaned_sq, data_rs_sq

def clean_ds(HHfiles, uvd_data_ds, uvd_diff_ds, area=500., tol=1e-7, skip_wgts=0.2, N_threads=4, freq_range=[45,240], pols=['nn', 'ee', 'ne', 'en'], autos=True):
    
    Nants = len(uvd_data_ds.get_ants())
    
    flag_folders = [HHfile.split('.sum')[0]+'.stage_1_xrfi' for HHfile in HHfiles]
    flag_file = [sorted(glob.glob(os.path.join(flag_folder,'*sum.combined_flags1.h5')))[0] for flag_folder in flag_folders]
    uvf = UVFlag()
    uvf.read(flag_file)
    if(autos == True):
        bls = [(ant, ant) for ant in uvd_data_ds.get_ants()]
    else:
        bls = uvd_data_ds.get_antpairs()        
    uvd_data_ds.flag_array[:,0,:,:] = np.repeat(uvf.flag_array, len(bls), axis=0)

    def func_ds_clean_mpi(rank, queue, N_threads, uvd_ds, uvd_diff, freq_range):
        _data_cleaned_sq, data_rs_sq = {}, {}
            
        N_jobs_each_thread = len(bls)*len(pols)/N_threads
        k = 0
        for i, bl in enumerate(bls):
            for j, pol in enumerate(pols):
                which_rank = int(k/N_jobs_each_thread)
                if(rank == which_rank):
                    key = (bl[0], bl[1], pol)
                    _data_cleaned_sq[key], data_rs_sq[key] = _clean_ds_per_bl_pol(bl, pol, uvd_ds, uvd_diff, area, tol, skip_wgts, freq_range)
                k += 1
        queue.put([_data_cleaned_sq, data_rs_sq])

    def run_clean_ds_mpi(uvd_ds, uvd_diff, N_threads, freq_range):
        _data_cleaned_sq, data_rs_sq = {}, {}

        queue = Queue()
        for rank in range(N_threads):
            p = Process(target=func_ds_clean_mpi, args=(rank, queue, N_threads, uvd_ds, uvd_diff, freq_range))
            p.start()

        for rank in range(N_threads):
            data = queue.get()
            _d_cleaned_sq = data[0]
            d_rs_sq = data[1]
            _data_cleaned_sq = {**_data_cleaned_sq, **_d_cleaned_sq}
            data_rs_sq = {**data_rs_sq, **d_rs_sq}

        return _data_cleaned_sq, data_rs_sq
    
    _data_cleaned_sq, data_rs_sq = run_clean_ds_mpi(uvd_data_ds, uvd_diff_ds, N_threads, freq_range)
    
    return _data_cleaned_sq, data_rs_sq

def plot_wfds(uvd, _data_sq, pol):
    """
    Waterfall diagram for autocorrelation delay spectrum
    
    Parameters:
    ----------
    uvd: UVData Object
        Sample observation from the desired night, used for getting antenna information.
    _data_sq: Dict
        Square of delay spectra, formatted as _data_sq[(ant1, ant2, pol)]
    pol: String
        String of polarization
    """
    nodes, antDict, inclNodes = generate_nodeDict(uvd)
    ants = uvd.get_ants()
    sorted_ants = sort_antennas(uvd)
    freqs = uvd.freq_array[0]
    taus = np.fft.fftshift(np.fft.fftfreq(freqs.size, np.diff(freqs)[0]))*1e9
    times = uvd.time_array
    lsts = uvd.lst_array*3.819719
    inds = np.unique(lsts,return_index=True)[1]
    lsts = [lsts[ind] for ind in sorted(inds)]
    
    maxants = 0
    polnames = ['nn','ee','ne','en']
    for node in nodes:
        n = len(nodes[node]['ants'])
        if n>maxants:
            maxants = n
    
    Nants = len(ants)
    Nside = maxants
    Yside = len(inclNodes)
    
    t_index = 0
    jd = times[t_index]
    utc = Time(jd, format='jd').datetime
    
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
    h = cm_active.ActiveData(at_date=jd)
    h.load_apriori()
    
    custom_lines = []
    labels = []
    for s in status_colors.keys():
        c = status_colors[s]
        custom_lines.append(Line2D([0],[0],color=c,lw=2))
        labels.append(s)
    ptitle = 1.92/(Yside*3)
    fig, axes = plt.subplots(Yside, Nside, figsize=(16,Yside*3))
    if pol == 0:
        fig.suptitle("nn polarization", fontsize=14, y=1+ptitle)
        vmin, vmax = -50, -30
    elif pol == 1:
        fig.suptitle("ee polarization", fontsize=14, y=1+ptitle)
        vmin, vmax = -50, -30
    elif pol == 2:
        fig.suptitle("ne polarization", fontsize=14, y=1+ptitle)
        vmin, vmax = -30, 0
    else:
        fig.suptitle("en polarization", fontsize=14, y=1+ptitle)
        vmin, vmax = -30, 0
    fig.legend(custom_lines,labels,bbox_to_anchor=(0.7,1),ncol=3)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.subplots_adjust(left=0, bottom=.1, right=.9, top=1, wspace=0.1, hspace=0.3)

    xticks = np.int32(np.ceil(np.linspace(0,len(taus)-1,5)))
    xticklabels = np.around(taus[xticks],0)
    yticks = [int(i) for i in np.linspace(0,len(lsts)-1,6)]
    yticklabels = [np.around(lsts[ytick],1) for ytick in yticks]
    for i,n in enumerate(inclNodes):
        ants = nodes[n]['ants']
        j = 0
        for _,a in enumerate(sorted_ants):
            if a not in ants:
                continue
            status = h.apriori[f'HH{a}:A'].status
            abb = status_abbreviations[status]
            ax = axes[i,j]
            key = (a, a, polnames[pol])
            ds = 10.*np.log10(np.sqrt(np.abs(_data_sq[key])/np.abs(_data_sq[key]).max(axis=1)[:,np.newaxis]))
            im = ax.imshow(ds, aspect='auto', interpolation='nearest', vmin=vmin, vmax=vmax)
            ax.set_title(f'{a} ({abb})', fontsize=10, backgroundcolor=status_colors[status])
            if i == len(inclNodes)-1:
                ax.set_xticks(xticks)
                ax.set_xticklabels(xticklabels)
                ax.set_xlabel('Delay (ns)', fontsize=10)
                [t.set_rotation(70) for t in ax.get_xticklabels()]
            else:
                ax.set_xticks(xticks)
                ax.set_xticklabels([])
            if j != 0:
                ax.set_yticks(yticks)
                ax.set_yticklabels([])
            else:
                [t.set_fontsize(12) for t in ax.get_yticklabels()]
                ax.set_ylabel('Time (LST)', fontsize=10)
                ax.set_yticks(yticks)
                ax.set_yticklabels(yticklabels)
                ax.set_ylabel('Time (LST)', fontsize=10)
            j += 1
        for k in range(j,maxants):
            axes[i,k].axis('off')
        pos = ax.get_position()
        cbar_ax=fig.add_axes([0.91,pos.y0,0.01,pos.height])        
        cbar = fig.colorbar(im, cax=cbar_ax)
        cbar.set_label(f'Node {n}',rotation=270, labelpad=15)
#         cbarticks = [np.around(x,1) for x in np.linspace(vmin,vmax,7)[i] for i in cbar.get_ticks()]
#         cbar.set_ticklabels(cbarticks)
#         axes[i,maxants-1].annotate(f'Node {n}', (.97,pos.y0+.03),xycoords='figure fraction',rotation=270)
    fig.show()

def plot_antFeatureMap(uvd, _data_sq, JD, pol='ee'):
    """
    Plots the positions of all antennas that have data, colored by feature strength.
    Parameters
    ----------
    uvd: UVData object
        Observation to extract antenna numbers and positions from
    _data_sq: Dict
        Dictionary structured as _data_sq[(antenna number, antenna number, pol)], where the values are the 
        feature strength that will determined the color on the map.
    JD: Int
        Julian date of the data
    pol: String
        Polarization to plot
    """
    
    nd = {0: {'pos': [21.427320986820824, -30.722353385032143],
      'ants': [0, 1, 2, 11, 12, 13, 14, 23, 24, 25, 26, 39]},
     1: {'pos': [21.427906055943357, -30.722367970752067],
      'ants': [3, 4, 5, 6, 15, 16, 17, 18, 27, 28, 29, 30]},
     2: {'pos': [21.428502498826337, -30.722356438400826],
      'ants': [7, 8, 9, 10, 19, 20, 21, 31, 32, 33, 321, 323]},
     3: {'pos': [21.427102788863543, -30.72199587048034],
      'ants': [36, 37, 38, 50, 51, 52, 53, 65, 66, 67, 68, 320]},
     4: {'pos': [21.427671849802184, -30.7220282862175],
      'ants': [40, 41, 42, 54, 55, 56, 57, 69, 70, 71, 72, 324]},
     5: {'pos': [21.42829977472493, -30.722027118338183],
      'ants': [43, 44, 45, 46, 58, 59, 60, 73, 74, 75, 76, 322]},
     6: {'pos': [21.428836727299945, -30.72219119740069],
      'ants': [22, 34, 35, 47, 48, 49, 61, 62, 63, 64, 77, 78]},
     7: {'pos': [21.426862825121685, -30.72169978685838],
      'ants': [81, 82, 83, 98, 99, 100, 116, 117, 118, 119, 137, 138]},
     8: {'pos': [21.427419087275524, -30.72169615183073],
      'ants': [84, 85, 86, 87, 101, 102, 103, 104, 120, 121, 122, 123]},
     9: {'pos': [21.42802904166864, -30.721694142092485],
      'ants': [88, 89, 90, 91, 105, 106, 107, 108, 124, 125, 126, 325]},
     10: {'pos': [21.42863899600041, -30.721692129488424],
      'ants': [92, 93, 94, 109, 110, 111, 112, 127, 128, 129, 130, 328]},
     11: {'pos': [21.42914035998215, -30.721744794462655],
      'ants': [79, 80, 95, 96, 97, 113, 114, 115, 131, 132, 133, 134]},
     12: {'pos': [21.426763768223857, -30.72133448059758],
      'ants': [135, 136, 155, 156, 157, 158, 176, 177, 178, 179, 329, 333]},
     13: {'pos': [21.42734159294201, -30.72141297904905],
      'ants': [139, 140, 141, 142, 159, 160, 161, 162, 180, 181, 182, 183]},
     14: {'pos': [21.428012089958028, -30.721403280585722],
      'ants': [143, 144, 145, 146, 163, 164, 165, 166, 184, 185, 186, 187]},
     15: {'pos': [21.428561498114107, -30.721408957468245],
      'ants': [147, 148, 149, 150, 167, 168, 169, 170, 188, 189, 190, 191]},
     16: {'pos': [21.42914681969319, -30.721434635693182],
      'ants': [151, 152, 153, 154, 171, 172, 173, 174, 192, 193, 194, 213]},
     17: {'pos': [21.426857989080208, -30.72109992091893],
      'ants': [196, 197, 198, 199, 215, 216, 217, 218, 233, 234, 235, 337]},
     18: {'pos': [21.427443064426363, -30.7210702936363],
      'ants': [200, 201, 202, 203, 219, 220, 221, 222, 236, 237, 238, 239]},
     19: {'pos': [21.428053014877808, -30.72106828382215],
      'ants': [204, 205, 206, 207, 223, 224, 225, 226, 240, 241, 242, 243]},
     20: {'pos': [21.428662965267904, -30.721066271142263],
      'ants': [208, 209, 210, 211, 227, 228, 229, 244, 245, 246, 261, 262]},
     21: {'pos': [21.429383860959977, -30.721211242305866],
      'ants': [175, 195, 212, 214, 231, 232, 326, 327, 331, 332, 336, 340]},
     22: {'pos': [21.427060077987438, -30.720670550054763],
      'ants': [250, 251, 252, 253, 266, 267, 268, 269, 281, 282, 283, 295]},
     23: {'pos': [21.42767002595312, -30.720668542063535],
      'ants': [254, 255, 256, 257, 270, 271, 272, 273, 284, 285, 286, 287]},
     24: {'pos': [21.42838974031629, -30.720641805595115],
      'ants': [258, 259, 260, 274, 275, 276, 288, 289, 290, 291, 302, 303]},
     25: {'pos': [21.429052089734615, -30.720798251186455],
      'ants': [230, 247, 248, 249, 263, 264, 265, 279, 280, 335, 339]},
     26: {'pos': [21.427312432981267, -30.720413813332755],
      'ants': [296, 297, 298, 308, 309, 310, 330, 334, 338, 341, 346, 347]},
     27: {'pos': [21.42789750442093, -30.72038427427254],
      'ants': [299, 300, 301, 311, 312, 313, 314, 342, 343]},
     28: {'pos': [21.428507450517774, -30.72038226236355],
      'ants': [304, 305, 315, 316, 317, 318, 348]},
     29: {'pos': [21.42885912979846, -30.72052728164184],
      'ants': [277, 278, 292, 293, 294, 306, 307, 319, 344, 345, 349]}}
    
    freqs = uvd.freq_array[0]
    taus = np.fft.fftshift(np.fft.fftfreq(freqs.size, np.diff(freqs)[0]))*1e9
    idx_region1 = np.where(np.logical_and(taus > 2500, taus < 3000))[0]
    idx_region2 = np.where(np.logical_and(taus > 2000, taus < 2500))[0]
    
    fig = plt.figure(figsize=(14,10))
    nodes, antDict, inclNodes = generate_nodeDict(uvd)
    antnums = uvd.get_ants()
    cmap = plt.get_cmap('inferno')
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=10))
    sm._A = []
    ampmax = 10
    ampmin = 0
    rang = ampmax-ampmin
    for node in sorted(inclNodes):
        ants = sorted(nodes[node]['ants'])
        nodeamps = []
        points = np.zeros((len(ants),2))
        for i,antNum in enumerate(ants):
            key = (antNum, antNum, pol)
            idx = np.argwhere(uvd.antenna_numbers == antNum)[0][0]
            antPos = uvd.antenna_positions[idx]
            amp = 10*np.log10(np.sqrt(np.nanmean(np.abs(_data_sq[key][:,idx_region1]))/np.nanmean(np.abs(_data_sq[key][:,idx_region2]))))
            nodeamps.append(amp)
            points[i,:] = [antPos[1],antPos[2]]
        hull = scipy.spatial.ConvexHull(points)
        center = np.average(points,axis=0)
        hullpoints = np.zeros((len(hull.simplices),2))
        namp = np.nanmean(nodeamps)
        ncolor = cmap(float((namp-ampmin)/rang))
        plt.fill(points[hull.vertices,0], points[hull.vertices,1],alpha=0.5,color=ncolor)
    for node in sorted(inclNodes):
        ants = sorted(nodes[node]['ants'])
        npos = nd[int(node)]['pos']
        plt.plot(npos[0],npos[1],marker="s",markersize=15,color="black")
        for antNum in ants:
            idx = np.argwhere(uvd.antenna_numbers == antNum)[0][0]
            antPos = uvd.antenna_positions[idx]
            key = (antNum, antNum, pol)
            amp = 10*np.log10(np.sqrt(np.nanmean(np.abs(_data_sq[key][:,idx_region1]))/np.nanmean(np.abs(_data_sq[key][:,idx_region2]))))
            if math.isnan(amp):
                marker="v"
                color="r"
                markersize=30
            else:
                coloramp = cmap(float((amp-ampmin)/rang))
                color = coloramp
                marker="h"
                markersize=40
            plt.plot(antPos[1],antPos[2],marker=marker,markersize=markersize,color=color)
            if coloramp[0]>0.6 or math.isnan(amp):
                plt.text(antPos[1]-3,antPos[2],str(antNum),color='black')
            else:
                plt.text(antPos[1]-3,antPos[2],str(antNum),color='white')
    plt.title('Antenna map - {} polarization (JD{})'.format(pol, JD))
    cbar = fig.colorbar(sm)
    cbar.set_label('2700ns Feature Amplitude (dB)')

def make_html_dspec(HHfiles, uvd, uvd_diff, _data_cleaned_sq, data_rs_sq, JD):

    freqs = uvd.freq_array[0]

    FM_idx = np.searchsorted(freqs*1e-6, [85,110])
    flag_FM = np.zeros(freqs.size, dtype=bool)
    flag_FM[FM_idx[0]:FM_idx[1]] = True

    pols = ['nn', 'ee']
    freqs1 = [40, 50, 120, 155, 190]
    freqs2 = [250, 85, 155, 190, 225]
    freq_range = freqs1+freqs2

    _data_sq_sub = {}
    data_rs_sq_sub = {}
    for freq1, freq2 in zip(freqs1, freqs2):
        if(freq1 == 40):
            _data_sq_sub[(freq1, freq2)] = _data_cleaned_sq
            data_rs_sq_sub[(freq1, freq2)] = data_rs_sq
        else:
            _d_sq_sub, d_rs_sq_sub = clean_ds(HHfiles, uvd, uvd_diff, freq_range=[freq1, freq2], pols=['nn','ee'], autos=True)
            _data_sq_sub[(freq1, freq2)] = _d_sq_sub
            data_rs_sq_sub[(freq1, freq2)] = d_rs_sq_sub

    bls = [(ant, ant) for ant in uvd.get_ants()]
    pols = ['nn', 'ee']
    nodes, antDict, inclNodes = generate_nodeDict(uvd)

    data_full = []
    wgts_full = []
    taus_full = []
    _data_full = []
    _diff_full = []
    _diff_full2 = []
    N_xaxis = []
    N_aggr = [0]
    keys = []
    for i, bl in enumerate(bls):
        for j, pol in enumerate(pols):
            key = (bl[0],bl[1],pol)
            keys.append(str(key)+' -- node {} (snap {})'.format(int(antDict[bl[0]]['node']),antDict[bl[0]]['snapLocs'][0]))
            auto = np.abs(uvd.get_data(key))
            auto /= np.median(auto, axis=1)[:,np.newaxis]
            auto[np.isinf(auto)] = np.nan
            auto_ave = np.nanmean(auto, axis=0, dtype=np.float32)

            wgts = (~uvd.get_flags(key)*~flag_FM[np.newaxis,:]).astype(np.float32)
            wgts_ave = np.mean(wgts, axis=0)
            wgts_ave = np.where(wgts_ave > 0.8, 1, 0)

            if(np.isnan(np.mean(auto_ave)) != True):
                data_full = data_full + list(auto_ave)
            else:
                data_full = data_full + list(np.isnan(auto_ave).astype(float))
            wgts_full = wgts_full + list(wgts_ave)

            for freq1, freq2 in zip(freqs1, freqs2):
                _data_ave = np.sqrt(np.abs(np.nanmean(_data_sq_sub[(freq1,freq2)][key], axis=0))).astype(np.float32)
                idx_freq = np.where(np.logical_and(freqs*1e-6 > freq1, freqs*1e-6 < freq2))[0]
            
                win = dspec.gen_window('bh7', idx_freq.size)
                _diff = np.fft.fftshift(np.fft.ifft(uvd_diff.get_data(key)[:,idx_freq]*win), axes=1)
#                 _diff_ave = np.abs(np.mean(_diff, axis=0)).astype(np.float32)
                _diff_ave = np.sqrt(np.abs(np.mean(_diff.conj()*_diff, axis=0))/(_diff.shape[0])).astype(np.float32)
#                 _diff_ave = np.sqrt(np.abs(_diff.conj()*_diff)[0]/(_diff.shape[0])).astype(np.float32)

                if(np.isnan(np.mean(_data_ave)) != True and np.mean(_data_ave) != 0):
                    _data_full = _data_full + list(10*np.log10(_data_ave/_data_ave.max()))
                    _diff_full = _diff_full + list(10*np.log10(_diff_ave/_data_ave.max()))
                else:
                    _data_full = _data_full + list(np.isnan(_data_ave).astype(float))
                    _diff_full = _diff_full + list(np.isnan(_diff_ave).astype(float))

                if(i == 0 and j == 0):
                    freqs_sub = freqs[idx_freq]
                    taus_sub = np.float32(np.fft.fftshift(np.fft.fftfreq(freqs_sub.size, np.diff(freqs_sub)[0])))
                    taus_full = taus_full + list(taus_sub*1e9)
                    N_xaxis.append(len(freqs_sub))
                    N_aggr.append(np.sum(N_xaxis))

    x_le = taus_full[:freqs.size]
    ds_update = _data_full[:freqs.size]
    dff_update = _diff_full[:freqs.size]
    x_ri = freqs/1e6
    auto_update = np.log10(data_full[:freqs.size])
    auto_flagged_update = np.array(auto_update, dtype=np.float32)/np.array(wgts_full[:freqs.size], dtype=np.float32)-0.1

    source = ColumnDataSource(data=dict(x_le=x_le, ds_update=ds_update, dff_update=dff_update, N_xaxis=N_xaxis, N_aggr=N_aggr, 
                                        taus_full=taus_full, _data_full=_data_full, _diff_full=_diff_full,
                                        x_ri=x_ri, auto_update=auto_update, auto_flagged_update=auto_flagged_update,
                                        data_full=data_full, wgts_full=wgts_full))

    plot1 = figure(title="Delay spectrum", x_range=(0, 4500), y_range=(-60, 0), plot_width=450, plot_height=400, output_backend="canvas")
    plot1.line('x_le', 'ds_update', source=source, color='#1f77b4', line_width=2, alpha=0.8, legend_label='delay spectrum')
    plot1.line('x_le', 'dff_update', source=source, color='red', line_width=1.5, alpha=0.8, legend_label='noise from diff')
    plot1.xaxis.axis_label = '𝜏 (ns)'
    plot1.yaxis.axis_label = '|Ṽ (𝜏)| in dB'

    plot2 = figure(title="Autocorrelation", y_range=(-0.6, 0.4), x_range=Range1d(start=freqs.min()/1e6, end=freqs.max()/1e6), plot_width=450, plot_height=400, output_backend="canvas")
    plot2.line('x_ri', 'auto_update', source=source, color='#ff7f0e', line_width=2, alpha=0.8, legend_label='unflagged auto')
    plot2.line('x_ri', 'auto_flagged_update', source=source, color='#1f77b4', line_width=2, alpha=0.8, legend_label='flagged auto')
    plot2.xaxis.axis_label = '𝜈 (MHz)'
    plot2.yaxis.axis_label = 'log10(|V(𝜈)|)'

    radio_button = RadioButtonGroup(labels=["Full band", "50-85 MHz", "120-155 MHz", "155-190 MHz", "190-225 MHz"], active=0)
    select = Select(title="key:", value=keys[0], options=keys, width=300)

    callback = CustomJS(args=dict(source=source, select=select, radio_button=radio_button, xr=plot2.x_range),
                        code="""
        var data = source.data;
        var active = radio_button.active;
        var key = select.value;
        var keys = select.options;
        var x_le = [];
        var y1_le = [];
        var y2_le = [];
        var y1_ri = [];
        var y2_ri = [];
        var N_xaxis = data['N_xaxis'];
        var N_aggr = data['N_aggr'];
        var taus_full = data['taus_full'];
        var _data_full = data['_data_full'];
        var _diff_full = data['_diff_full'];
        var x_ri = data['x_ri']
        var data_full = data['data_full']
        var wgts_full = data['wgts_full']
        for (var i = 0; i < keys.length; i++) {
            if (key == keys[i]) {
                for (j = 0; j < N_xaxis[active]; j++) {
                    x_le.push(taus_full[N_aggr[active]+j]);
                    y1_le.push(_data_full[N_aggr[5]*i+N_aggr[active]+j]);
                    y2_le.push(_diff_full[N_aggr[5]*i+N_aggr[active]+j]);
                }
                for (var j = 0; j < x_ri.length; j++) {
                    y1_ri.push(Math.log10(data_full[x_ri.length*i+j]));
                    y2_ri.push(Math.log10(data_full[x_ri.length*i+j])/wgts_full[x_ri.length*i+j]-0.1);
                }
            }
        }
        data['x_le'] = x_le;
        data['ds_update'] = y1_le;
        data['dff_update'] = y2_le;
        data['auto_update'] = y1_ri;
        data['auto_flagged_update'] = y2_ri;
        if (active == 0) {
            var start = 46.92
            var end = 234.30
        }
        else if (active == 1) {
            var start = 50
            var end = 85
        }
        else {
            var start = 120+(active-2)*35
            var end = 120+(active-1)*35
        }
        xr.setv({"start": start, "end": end})
        source.change.emit();
    """)

    radio_button.js_on_change('active', callback)
    select.js_on_change('value', callback)
    plot2.x_range.js_on_change('start', callback)
    plot2.x_range.js_on_change('end', callback)


    layout = column(
        row(plot1, plot2),
        select,
        column(radio_button)
    )

    output_file("delay_spectrum_JD{}.html".format(JD), title="delay_spectrum_JD{}".format(JD))

    show(layout)
    
def CorrMatrix_2700ns(uvd, HHfiles, difffiles, JD):
    """
    Plots a matrix representing the 2700ns feature correlation of each baseline.
    
    Parameters:
    ----------
    uvd: UVData Object
        Sample observation from the desired night, used for getting antenna information.
    HHfiles: List
        List of all files for a night of observation
    HHfiles: List
        List of diff files for a night of observation
    JD: String
        JD of the given night of observation
    """
    pols = ['nn','ee','ne','en']
    
    Nants = len(uvd.get_ants())
    files, lsts, inds = get_hourly_files(uvd, HHfiles, JD)
    
    antpos, ants = uvd.get_ENU_antpos()
    bl_len = []
    for antpair in uvd.get_antpairs():
        idx_ant1 = np.where(antpair[0] == ants)[0]
        idx_ant2 = np.where(antpair[1] == ants)[0]
        bl_len.append(np.sqrt(np.sum((antpos[idx_ant2]-antpos[idx_ant1])**2)))
    bl_len = np.array(bl_len)
    area = 250+bl_len/scipy.constants.c*1e9
    
    nTimes = len(files)
    if nTimes > 3:
        plotTimes = [0,nTimes//2,nTimes-1]
    else:
        plotTimes = np.arange(0,nTimes,1)

    for t in plotTimes:
        ind = inds[t]
        HHfile = HHfiles[ind]
        difffile = difffiles[ind]
        uvd_data_ds = UVData()
        uvd_data_ds.read(HHfile)
        uvd_diff_ds = UVData()
        uvd_diff_ds.read(difffile)

        _d_cleaned_sq, _ = clean_ds([HHfile], uvd_data_ds, uvd_diff_ds, pols=pols, autos=False, area=area)

        freqs = uvd_data_ds.freq_array[0]
        taus = np.fft.fftshift(np.fft.fftfreq(freqs.size, np.diff(freqs)[0]))*1e9
        idx_region1 = np.where(np.logical_and(taus > 2500, taus < 3000))[0]
        idx_region2 = np.where(np.logical_and(taus > 2000, taus < 2500))[0]

        amp = {}
        for pol in pols:
            _data_cleaned_sq = np.zeros((Nants, Nants, len(taus)), dtype=np.complex128)
            for i, ant1 in enumerate(uvd_data_ds.get_ants()):
                for j, ant2 in enumerate(uvd_data_ds.get_ants()):
                    if(i <= j):
                        bl = (ant1, ant2)
                        try:
                            _data_cleaned_sq[i,j] = np.nanmean(_d_cleaned_sq[(bl[0],bl[1],pol)], axis=0)
                        except:
                            _data_cleaned_sq[i,j] = np.nanmean(_d_cleaned_sq[(bl[1],bl[0],pol)], axis=0)
                        _data_cleaned_sq[j,i] = _data_cleaned_sq[i,j]
            amp[pol] = 10*np.log10(np.sqrt(np.nanmean(np.abs(_data_cleaned_sq[:,:,idx_region1]), axis=-1)/np.nanmean(np.abs(_data_cleaned_sq[:,:,idx_region2]), axis=-1)))

        plotCorrMatrix(uvd_data_ds, amp, pols=pols, nodes='auto', vminIn=0, vmaxIn=3)
        
def plot_metric(metrics, ants=None, antpols=None, title='', ylabel='Modified z-Score', xlabel=''):
    '''Helper function for quickly plotting an individual antenna metric.'''

    if ants is None:
        ants = list(set([key[0] for key in metrics.keys()]))
    if antpols is None:
        antpols = list(set([key[1] for key in metrics.keys()]))
    for antpol in antpols:
        for i,ant in enumerate(ants):
            metric = 0
            if (ant,antpol) in metrics:
                metric = metrics[(ant,antpol)]
            plt.plot(i,metric,'.')
            plt.annotate(str(ant)+antpol,xy=(i,metrics[(ant,antpol)]))
        plt.gca().set_prop_cycle(None)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
def show_metric(ant_metrics, antmetfiles, ants=None, antpols=None, title='', ylabel='Modified z-Score', xlabel=''):
    print("Ant Metrics for {}".format(antmetfiles[1]))
    plt.figure()
    plot_metric(ant_metrics['final_mod_z_scores']['meanVij'],
            title = 'Mean Vij Modified z-Score')

    plt.figure()
    plot_metric(ant_metrics['final_mod_z_scores']['redCorr'],
            title = 'Redundant Visibility Correlation Modified z-Score')

    plt.figure()
    plot_metric(ant_metrics['final_mod_z_scores']['meanVijXPol'], antpols=['n'],
            title = 'Modified z-score of (Vxy+Vyx)/(Vxx+Vyy)')

    plt.figure()
    plot_metric(ant_metrics['final_mod_z_scores']['redCorrXPol'], antpols=['n'],
            title = 'Modified z-Score of Power Correlation Ratio Cross/Same')
    plt.figure()
    plot_metric(ant_metrics['final_mod_z_scores']['redCorrXPol'], antpols=['e'],
            title = 'Modified z-Score of Power Correlation Ratio Cross/Same')

def all_ant_mets(antmetfiles,HHfiles):
    file = HHfiles[0]
    uvd_hh = UVData()
    uvd_hh.read_uvh5(file)
    uvdx = uvd_hh.select(polarizations = -5, inplace = False)
    uvdx.ants = np.unique(np.concatenate([uvdx.ant_1_array, uvdx.ant_2_array]))
    ants = uvdx.get_ants()
    times = uvd_hh.time_array
    Nants = len(ants)
    jd_start = np.floor(times.min())
    antfinfiles = []
    for i,file in enumerate(antmetfiles):
        if i%50==0:
            antfinfiles.append(antmetfiles[i])
    Nfiles = len(antfinfiles)
    Nfiles2 = len(antmetfiles)
    xants = np.zeros((Nants*2, Nfiles2))
    dead_ants = np.zeros((Nants*2, Nfiles2))
    cross_ants = np.zeros((Nants*2, Nfiles2))
    badants = []
    pol2ind = {'n':0, 'e':1}
    times = []

    for i,file in enumerate(antfinfiles):
        time = file[54:60]
        times.append(time)

    for i,file in enumerate(antmetfiles):
        antmets = hera_qm.ant_metrics.load_antenna_metrics(file)
        for j in antmets['xants']:
            xants[2*np.where(ants==j[0])[0]+pol2ind[j[1]], i] = 1
        badants.extend(map(lambda x: x[0], antmets['xants']))
        for j in antmets['crossed_ants']:
            cross_ants[2*np.where(ants==j[0])[0]+pol2ind[j[1]], i] = 1
        for j in antmets['dead_ants']:
            dead_ants[2*np.where(ants==j[0])[0]+pol2ind[j[1]], i] = 1

    badants = np.unique(badants)
    xants[np.where(xants==1)] *= np.nan
    dead_ants[np.where(dead_ants==0)] *= np.nan
    cross_ants[np.where(cross_ants==0)] *= np.nan

    antslabels = []
    for i in ants:
        labeln = str(i) + 'n'
        labele = str(i) + 'e'
        antslabels.append(labeln)
        antslabels.append(labele)

    fig, ax = plt.subplots(1, figsize=(16,20))

    # plotting
    ax.matshow(xants, aspect='auto', cmap='RdYlGn_r', vmin=-.3, vmax=1.3,
           extent=[0, len(times), Nants*2, 0])
    ax.matshow(dead_ants, aspect='auto', cmap='RdYlGn_r', vmin=-.3, vmax=1.3,
           extent=[0, len(times), Nants*2, 0])
    ax.matshow(cross_ants, aspect='auto', cmap='RdBu', vmin=-.3, vmax=1.3,
           extent=[0, len(times), Nants*2, 0])

    # axes
    ax.grid(color='k')
    ax.xaxis.set_ticks_position('bottom')
    ax.set_xticks(np.arange(len(times))+0.5)
    ax.set_yticks(np.arange(Nants*2)+0.5)
    ax.tick_params(size=8)

    if Nfiles > 20:
        ticklabels = times
        ax.set_xticklabels(ticklabels)
    else:
        ax.set_xticklabels(times)

    ax.set_yticklabels(antslabels)

    [t.set_rotation(30) for t in ax.get_xticklabels()]
    [t.set_size(12) for t in ax.get_xticklabels()]
    [t.set_rotation(0) for t in ax.get_yticklabels()]
    [t.set_size(12) for t in ax.get_yticklabels()]

    ax.set_title("Ant Metrics bad ants over observation", fontsize=14)
    ax.set_xlabel('decimal of JD = {}'.format(int(jd_start)), fontsize=16)
    ax.set_ylabel('antenna number and pol', fontsize=16)
    red_ptch = mpatches.Patch(color='red')
    grn_ptch = mpatches.Patch(color='green')
    blu_ptch = mpatches.Patch(color='blue')
    ax.legend([red_ptch, blu_ptch, grn_ptch], ['dead ant', 'cross ant', 'good ant'], fontsize=14)
