"""
Functions to compute source-space time series and functional connectivity
from HCP resting-state MEG data with MNE-HCP



Usage
====================================================================================================

from mne_src_fc_rest import compute_src_label_ts,fc_mnespectral,fc_aec

subject = 105928


src_label_ts = compute_src_label_ts(subject)

source_scs = fc_mnespectral(source_label_ts,sfreq,freq_bands)

source_aecs = [fc_aec(t,sfreq,freq_bands) for t in source_scs]
source_aec_avgs = {freq: np.mean(np.array([a[freq] for a in source_aecs]),axis=0)
                        for freq in freq_bands.keys()}

"""



"""
Setup
====================================================================================================
"""


# Generic stuff

import os,sys,glob,numpy as np,pandas as pd


# Neuroimaging & analysis stuff

import hcp
from mne import create_info,read_labels_from_annot,pick_types,extract_label_time_course,compute_raw_covariance
from mne.io import RawArray
from mne.preprocessing import create_ecg_epochs, create_eog_epochs,compute_proj_eog
from mne.minimum_norm import make_inverse_operator,apply_inverse_raw
from mne.filter import next_fast_len
from mne.connectivity import spectral_connectivity



"""
Define some variables
====================================================================================================
"""

# Stuff from mne-hcp

storage_dir = '/mnt/rotmangrid-scratch/mcintosh_lab/jgriffiths/Data/connectomeDB/downloaded_HCP900';
hcp_path = '/mnt/rotmangrid-scratch/mcintosh_lab/jgriffiths/Data/connectomeDB/downloaded_HCP900';

subjects_dir = 'hcp-subjects'
recordings_path = 'hcp-meg'

subject = '105923'
data_type = 'rest'
run_index = 0


# Freq band definitions for FC

freq_bands = dict(delta=(2, 4), theta=(5, 7), alpha=(8, 12), 
                  beta=(15, 29), gamma=(30, 45))


# Node reordering for connectivity matrix visualizations

nidx = np.array([46, 32, 50, 58, 16,  6, 22, 42, 26, 20, 12, 34, 16,  8, 64, 18, 60,
                 66, 30,  0, 62, 44, 48,  4, 56,  2, 54, 36, 40, 38, 24, 28, 52, 10,
                 47, 33, 51, 59, 15,  7, 23, 43, 27, 21, 13, 35, 17,  9, 65, 19, 61,
                 67, 31,  1, 63, 45, 49,  5, 57,  3, 55, 37, 41, 39, 25, 29, 53, 11])


# Freesurfer subjects dir

fs_subjects_dir = '/software/freesurfer/subjects'


# If not already present, create:
# - 'hcp-meg' path
# - 'hcp-subjects' path
# - symlink to freesurfer fsaverage with hcp-subjects

if not os.path.isdir(subjects_dir): os.makedirs(subjects_dir)
if not os.path.isdir(recordings_path): os.makedirs(recordings_path)
cmd = 'ln -s %s/fsaverage %s/fsaverage' %(fs_subjects_dir, subjects_dir)   
if not os.path.isdir(subjects_dir + '/fsaverage'):    
    os.system(cmd)
    # ! $cmd
    
  


"""
Define some functions
====================================================================================================
"""


# Wrappper for MNE spectral connectivity function

def fc_mnespectral(data,sfreq,freq_bands,metrics=None,verbose=None):

    """
    
    data is 
    freq bands is a dict of (lower,upper) freq tuples

    """
    
    if metrics is None: metrics = ['coh','cohy','imcoh','plv',
                                   'ppc','pli','pli2_unbiased',
                                   'wpli', 'wpli2_debiased']

    scs = {}
    for freq_band,(lfreq,hfreq) in freq_bands.items():
        
        res = spectral_connectivity(data, method=metrics, mode='multitaper',
                                    sfreq=sfreq, fmin=lfreq, fmax=hfreq, 
                                    faverage=True, n_jobs=1,
                                    verbose=verbose)

        con, freqs, times, n_epochs, n_tapers = res
        
        scs[freq_band] = con
        
    return scs


# Amplitude envelope correlations function

def fc_aec(data,sfreq,freq_bands):
    
    """
    
    data is  ROI x TIME
    freq bands is a dict of (lower,upper) freq tuples

    using bits from 
    https://martinos.org/mne/dev/auto_tutorials/plot_modifying_data_inplace.html
    
    """
    
    # N rois
    nr = data.shape[0]

    # Convert time series data to MNE raw datatype so we can use filter and hilbert functions
    ch_names = [str(s) for s in range(0,nr)]
    ch_types = ['eeg' for _ in ch_names]
    info = create_info(ch_names, sfreq, ch_types=ch_types)
    raw = RawArray(data.copy(),info)
    
    aecs = {}

    for freq_band,(lfreq,hfreq) in freq_bands.items():
        
        # Filter
        raw_band = raw.copy()
        raw_band.filter(lfreq, hfreq, l_trans_bandwidth=2., h_trans_bandwidth=2.,
                        fir_design='firwin')
        raw_hilb = raw_band.copy()
        
        # Compute hilbert transform
        hilb_picks = pick_types(raw_band.info, meg=False, eeg=True)
        raw_hilb.apply_hilbert(hilb_picks)

        # Take the amplitude and phase
        raw_amp = raw_hilb.copy()
        raw_amp.apply_function(np.abs, hilb_picks)
        raw_phase = raw_hilb.copy()
        raw_phase.apply_function(np.angle, hilb_picks)

        aecs[freq_band] = raw_amp.to_data_frame().corr().values
        
    return aecs





# This is the main analysis function. Computes and returns windowed source time series


def compute_src_label_ts(subject, crop_to = [0,250], resample_to = 100.,
                         bads = None, mag_reject = 5e-12,
                         win_len = 2000,n_wins = 11,verbose=None,lambda2=1./9.,
                         inv_method = 'dSPM', extract_ts_mode = 'mean_flip'):


    
    
    """
    Compute source label time series
    """
    
    
    
    """
    Compute anatomy
    """
    
    hcp.make_mne_anatomy(subject, subjects_dir=subjects_dir,
                         hcp_path=hcp_path,recordings_path=hcp_path)
    
    
    
    """
    Read surface labels
    """
    labels = read_labels_from_annot(subject, parc='aparc', subjects_dir=subjects_dir)
    labels_fsav = read_labels_from_annot('fsaverage',parc='aparc',subjects_dir=subjects_dir)
    
    
    
    """
    Read raw data
    """

    raw = hcp.read_raw(subject=subject,
                     data_type=data_type, hcp_path=hcp_path,
                     run_index=run_index)

    raw.load_data()

    raw.crop(crop_to[0],crop_to[1])
    
    raw.resample(resample_to)
    
    raw.info['bads'] = bads
    
    hcp.preprocessing.set_eog_ecg_channels(raw)

    hcp.preprocessing.apply_ref_correction(raw)

    info = raw.info.copy()

    raw.info['projs'] = []

    ecg_ave = create_ecg_epochs(raw).average()

    eog_ave = create_eog_epochs(raw).average()

    ssp_eog, _ = compute_proj_eog(raw, n_grad=1, n_mag=1, average=True, reject=dict(mag=mag_reject))
    raw.add_proj(ssp_eog, remove_existing=True)

    n_fft = next_fast_len(int(round(4 * raw.info['sfreq'])))

    sfreq = raw.info['sfreq']
    
    
    
    """
    Compute forward model
    """
    src_outputs = hcp.anatomy.compute_forward_stack(subject=subject, subjects_dir=subjects_dir,
                    hcp_path=hcp_path, recordings_path=hcp_path,src_params=dict(add_dist=False),
                    info_from=dict(data_type=data_type, run_index=run_index))
    fwd = src_outputs['fwd']


    """
    Compute noise covariance
    """
    raw_noise = hcp.read_raw(subject=subject, hcp_path=hcp_path,
                         data_type='noise_empty_room')
    raw_noise.load_data()
    hcp.preprocessing.apply_ref_correction(raw_noise)
    raw_noise.add_proj(ssp_eog)
    noise_cov = compute_raw_covariance(raw_noise, method='oas')



    """
    Compute inverse operator
    """
    
    raw.info = info
    inv_op = make_inverse_operator(raw.info, forward=fwd, noise_cov=noise_cov, verbose=verbose)
     
        
        
    
    """
    Compute source activity
    """
    
    wins = [[0,win_len]]
    for i in range(n_wins):
        new_wins = [wins[0][0]+(win_len*(i+1)),
                    wins[0][1]+(win_len*(i+1))]
        wins.append(new_wins)
    

    raw_srcs = []
    for win in wins:
        res = apply_inverse_raw(raw,inv_op,lambda2=lambda2,method=inv_method,label=None, start=win[0],stop=win[1],
                                nave=1,time_func=None,pick_ori=None, buffer_size=None,prepared=False,
                                method_params=None, verbose=verbose)
        raw_srcs.append(res)



    """
    Compute source label time series
    """
    src = inv_op['src']    
    label_ts = extract_label_time_course(raw_srcs, labels, src,mode=extract_ts_mode,return_generator=False)

    
    
    
    return label_ts,sfreq



"""
Command line call behaviour
====================================================================================================
"""


if __name__ == '__main__':
    

    subject = sys.argv[1]
    
    src_label_ts = compute_src_label_ts(subject)

    source_scs = fc_mnespectral(source_label_ts,sfreq,freq_bands)


    source_aecs = [fc_aec(t,sfreq,freq_bands) for t in source_scs]


    source_aec_avgs = {freq: np.mean(np.array([a[freq] for a in source_aecs]),axis=0)
                        for freq in freq_bands.keys()}


    # save to file...
    
   
    


   