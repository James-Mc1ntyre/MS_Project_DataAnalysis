'''

Run this file to load, preprocess, epoch, and visualize data from the VR VSM experiment


'''
# Imports
import os
from cmath import log, nan
from itertools import compress
import matplotlib.pyplot as plt
plt.rcParams["figure.dpi"] = 500
import mne
import numpy as np
import pandas as pd
import pyxdf
import seaborn as sns
from BCI2kReader import BCI2kReader as b2k


def load_IDs(filepath):
    '''
    Get participant IDs from filepath to data folders
    
    Output:
    list of participant IDs
    '''
    # Initialize list
    IDs = []
    # scan through top file for participant names
    with os.scandir(filepath) as entries:
        for entry in entries:
            # append participant name
            IDs.append(entry.name.split('_')[0])
    print('%d participants on file.' % len(IDs))
    print('IDs: ', IDs)
    return IDs

def load_EEG(filepath, IDs):
    '''
    Load EEG data from filepath and IDs

    Output:
    a list of EEG data instances with shape (Trial types, # of IDs)
    '''
    print('Loading EEG data')
    # Initialize lists
    EEG_easy_list , EEG_hard_list = [], []
    for ID in IDs:
        # scan each ID for trial dates and extrct EEG data
        with os.scandir(filepath + '\\' + ID) as entries:
            for entry in entries:
                print('Loading EEG data from %s...\nIngore warnings below' % ID)
                # extract EEG data with b2K reader
                EEG_easy_list.append(b2k.BCI2kReader(filepath + '\\' + ID + '\\' + entry.name + r'\EEG' + '\\' + ID + 'Easy.dat'))
                EEG_hard_list.append(b2k.BCI2kReader(filepath + '\\' + ID + '\\' + entry.name + r'\EEG' + '\\' + ID + 'Hard.dat'))
    print('EEG data loaded')
    both_EEG_lists = [EEG_easy_list, EEG_hard_list]
    assert len(EEG_easy_list) == len(EEG_hard_list), 'EEG lists not same length'
    return both_EEG_lists

def load_NIRS(filepath, IDs):
    '''
    Load NIRS data from filepath and IDs
    
    Output:
    a list of NIRS data instances with shape (Trial types, # of IDs)
    '''
    print('Loading NIRS data')
    # Initialize lists
    NIRS_easy_list , NIRS_hard_list = [], []
    for ID in IDs:
        # scan each ID for trial dates and extrct NIRS data
        with os.scandir(filepath + '\\' + ID) as entries:
            for entry in entries:
                print('Loading NIRS data from %s...' % ID)
                # load NIRS data directly to MNE format
                NIRS_easy_list.append(mne.io.read_raw_nirx(filepath + '\\' + ID + '\\' + entry.name + r'\NIRS' + '\\' + ID + 'Easy').load_data())
                NIRS_hard_list.append(mne.io.read_raw_nirx(filepath + '\\' + ID + '\\' + entry.name + r'\NIRS' + '\\' + ID + 'Hard').load_data())
    print('NIRS data loaded')
    both_NIRS_lists = [NIRS_easy_list, NIRS_hard_list]
    assert len(NIRS_easy_list) == len(NIRS_hard_list), 'NIRS lists not same length'
    return both_NIRS_lists

def load_bhv(filepath, IDs, offsetAngles, trialOrders):
    '''
    Load behavioral data from filepath and IDs
    get response angles and reaction times

    Output:
    a list of sequences with shape (Trial types, # of IDs)
    '''
    print('Loading behavioral data')
    # Initialize lists
    BHV_easy_list , BHV_hard_list = [], []
    response_easy_list , response_hard_list = [], []
    RT_easy_list , RT_hard_list = [], []
    endRT_easy_list , endRT_hard_list = [], []
    for ID in IDs:
        print('Loading Behavioral data from %s...\nIngore warnings below' % ID)
        # scan each ID for trial dates and extrct Behavioral data
        with os.scandir(filepath + '\\' + ID) as entries:
            for entry in entries:
                # for both trial types
                for type in ['Easy', 'Hard']:
                    # load LSL data
                    data, _ = pyxdf.load_xdf(filepath + '\\' + ID + '\\' + entry.name + r'\LSL' + '\\' + type + '.xdf')
                    # get Stream names
                    streamNames = []
                    for stream in data:
                        streamNames.append(stream['info']['name'][0])
                    # load relevent stream
                    bhv = data[streamNames.index('rotateObject')]
                    bhv_rot = bhv['time_series'].T[1]
                    bhv_tim_bool = np.diff(bhv['time_stamps']) > 0.1
                    bhv_tim_loc = [i for i, x in enumerate(bhv_tim_bool) if x]
                    bhv_tim = bhv['time_stamps'][bhv_tim_loc]
                    BHV_list = []
                    for evnt in bhv_tim:
                        evnt_loc_sample = np.where(bhv['time_stamps']==evnt)[0][0]
                        BHV_list.append(bhv_rot[evnt_loc_sample+1:evnt_loc_sample+8*90])
                    # Get correct/incorrect trials and reaction times with bhv and trial angles
                    trialOrder = trialOrders[ID]
                    offsetAngle = offsetAngles[ID]
                    offsetOrder = np.array(offsetAngle)[trialOrder]
                    response_difference = []
                    reaction_times = []
                    end_rotation_times = []
                    answer = []
                    for answer, offset in zip(BHV_list[:12], offsetOrder[:12]): 
                        answered_angle = offset - (answer[0] - answer[-1])
                        if answered_angle > 180: 
                            answered_angle = answered_angle-360
                        response_difference.append(answered_angle)
                        if np.all((np.diff(answer) == 0)):
                            reaction = nan
                            endOfRotation = nan
                        else:
                            reaction = np.nonzero(np.diff(answer))[0][0]/90
                            endOfRotation = np.nonzero(np.diff(answer))[0][-1]/90
                        reaction_times.append(reaction)
                        end_rotation_times.append(endOfRotation)
                    # Append to lists for returning vars
                    if type == 'Easy':
                        BHV_easy_list.append(BHV_list)
                        response_easy_list.append(response_difference)
                        RT_easy_list.append(reaction_times)
                        endRT_easy_list.append(end_rotation_times)
                    if type == 'Hard':
                        BHV_hard_list.append(BHV_list)
                        response_hard_list.append(response_difference)
                        RT_hard_list.append(reaction_times)
                        endRT_hard_list.append(end_rotation_times)
    print('Behavioral data loaded')
    both_BHV_lists = [BHV_easy_list, BHV_hard_list]
    both_response_lists = [response_easy_list, response_hard_list]
    both_RT_lists = [RT_easy_list, RT_hard_list]
    both_endRT_lists = [endRT_easy_list, endRT_hard_list]
    assert len(BHV_easy_list) == len(BHV_hard_list), 'Behavioral data lists not same length'
    return both_BHV_lists, both_response_lists, both_RT_lists, both_endRT_lists

def preprocess_EEG(both_EEG_lists, IDs, offsetAngles, trialOrders):
    '''
    Preprocess EEG data in MNE
    Add annotations to raw, but epoch in another function
    Filter data differerntly for spectral and temporal analysis
    Changing upper passband of the filter and saving them as different variables
    
    Output:
    2 lists of preprocessed MNE raw data instances with shape (Trial types, # of IDs)
    the first list is filtered with an upper bandpass of 20 Hz and the second is 35 Hz
    '''
    print('Starting EEG data preprocessing')
    # Initialize some variables
    proc_tempo_EEG_lists = []
    proc_spect_EEG_lists = []
    # different preprocessing steps for each analysis
    for filt_type in ['temporal', 'spectral']:
        proc_EEG_easy = []
        proc_EEG_hard = []
        for trials in range(len(both_EEG_lists)):
            assert len(IDs) == len(both_EEG_lists[0]) == len(both_EEG_lists[1]), 'ID list should equal length of data lists'
            for IDcount, ID in enumerate(IDs):
                with os.scandir(filepath + '\\' + ID) as entries:
                    for entry in entries:
                        print('Processing trial %d (%s) data from %s...' % (trials+1, filt_type, ID))
                        EEG_data = both_EEG_lists[trials][IDcount].signals
                        EEG_ch_names = both_EEG_lists[trials][IDcount].parameters['ChannelNames']
                        EEG_sfreq = both_EEG_lists[trials][IDcount].parameters['SamplingRate']
                        # MNE info
                        EEG_info = mne.create_info(ch_names=EEG_ch_names, ch_types='eeg', sfreq=EEG_sfreq)
                        EEG_raw = mne.io.RawArray(EEG_data, EEG_info, verbose=False)
                        # Set up montage
                        mont1020 = mne.channels.make_standard_montage('standard_1020')
                        ind = [i for (i, channel) in enumerate(mont1020.ch_names) if channel in EEG_ch_names]
                        mont1020.ch_names = [mont1020.ch_names[x] for x in ind]
                        kept_channel_info = [mont1020.dig[x+3] for x in ind]
                        mont1020.dig = mont1020.dig[0:3]+kept_channel_info
                        EEG_raw = EEG_raw.set_montage(mont1020)
                        # Triggers
                        EEG_TargetCode = both_EEG_lists[trials][IDcount].states['TargetCode'].squeeze()
                        state1 = np.diff(EEG_TargetCode) == 1
                        state1loc = [i for i, x in enumerate(state1) if x]
                        state1sec = np.array(state1loc)/EEG_sfreq
                        state3 = np.diff(EEG_TargetCode) == 3
                        state3loc = [i for i, x in enumerate(state3) if x]
                        state3sec = np.array(state3loc)/EEG_sfreq
                        # Label triggers (with offsets)
                        trialOrder = trialOrders[ID]
                        offsetAngle = offsetAngles[ID]
                        offsetOrder = np.array(offsetAngle)[trialOrder]
                        #check if > 12 trials otherwise just take 12 after first
                        assert len(state1sec) >= 12, 'state1 < 12 times'
                        assert len(state3sec) >= 12, 'state3 < 12 times'
                        if len(state1sec) > 12: # usually is
                            StimName = ['fakeFirst']
                            RotationName = ['fakeFirst']
                        else:
                            StimName = []
                            RotationName = []
                        for i in np.absolute(offsetOrder[:12]):
                            StimName.append('Stimulus_%d' % i)
                            RotationName.append('Rotation_%d' % i)
                        # add anotations to raw
                        annotations = mne.Annotations(onset=np.concatenate([state1sec[:len(StimName)],state3sec[:len(RotationName)]]), duration=0, description=StimName + RotationName)
                        EEG_raw.set_annotations(annotations)
                        # drop known bad channels
                        EEG_raw.drop_channels(['C2', 'Pz'])
                        # Set up filtering
                        if filt_type == 'temporal':
                            upper_passband = 20
                            lower_passband = 0.1
                        elif filt_type == 'spectral':
                            upper_passband = 35
                            lower_passband = 2
                        # Filter data
                        iir_params = mne.filter.create_filter(EEG_raw.get_data(), EEG_raw.info['sfreq'], lower_passband, upper_passband, method='iir', verbose=False, iir_params=dict(order=4, ftype='butter'))
                        EEG_raw.filter(lower_passband, upper_passband, method='iir', verbose=False, iir_params=iir_params)
                        # # Plot filter
                        # plotPath = os.getcwd() + r'\VR_Data' + '\\' + ID + '\\' + entry.name
                        # fig = mne.viz.plot_filter(iir_params, sfreq=EEG_raw.info['sfreq'], compensate=True, show=False, title='Butterworth IIR Order=4')
                        # if filt_type == 'temporal':
                        #     fig.savefig(fname=plotPath+'\\'+'EEG_temp_filter.png', format='png')
                        # elif filt_type == 'spectral':
                        #     fig.savefig(fname=plotPath+'\\'+'EEG_spect_filter.png', format='png')
                        EEG_raw.notch_filter(60, method='iir', verbose=False, iir_params=iir_params)
                        # # Visually drop bad channels
                        # EEG_raw.plot(scalings='auto')
                        # input('Visually mark bad channels\nPress any key to continue')
                        # EEG_raw.drop_channels(EEG_raw.info['bads'])
                        if filt_type == 'temporal':
                            # CAR filter for temporal analysis
                            EEG_raw.set_eeg_reference()
                        # append easy or hard trials
                        if trials == 0:
                            proc_EEG_easy.append(EEG_raw)
                        elif trials == 1:
                            proc_EEG_hard.append(EEG_raw)
        assert len(proc_EEG_hard) == len(proc_EEG_easy), 'procesed easy and hard trials are not the same len'
        if filt_type == 'temporal':
            proc_tempo_EEG_lists = [proc_EEG_easy, proc_EEG_hard]
        elif filt_type == 'spectral':
            proc_spect_EEG_lists = [proc_EEG_easy, proc_EEG_hard]
    print('EEG data has been preprocessed')
    assert len(proc_tempo_EEG_lists) == 2, 'EEG tempo list (trials) is > 2'
    assert len(proc_spect_EEG_lists) == 2, 'EEG spect list (trials) is > 2'
    return proc_tempo_EEG_lists, proc_spect_EEG_lists

def preprocess_NIRS(both_NIRS_lists, IDs, offsetAngles, trialOrders):
    '''
    Load EEG data to MNE and preprocess
    Filtering included
    Add annotations to raw, but epoch in another function

    Output:
    a list of preprocessed NIRS MNE raw instances with shape (Trial types, # of IDs)
    '''
    print('Starting NIRS data preprocessing')
    proc_NIRS_lists=[]
    proc_NIRS_easy=[]
    proc_NIRS_hard=[]
    for trials in range(len(both_NIRS_lists)):
        assert len(IDs) == len(both_NIRS_lists[0]) == len(both_NIRS_lists[1]), 'ID list should equal length of data lists'
        for IDcount, ID in enumerate(IDs):
            with os.scandir(filepath + '\\' + ID) as entries:
                for entry in entries:
                    print('Processing trial %d data from %s...' % (trials+1, ID))
                    # Optical density
                    od = mne.preprocessing.nirs.optical_density(both_NIRS_lists[trials][IDcount])
                    # Scalp coupling index, mark some channels as bad
                    sci = mne.preprocessing.nirs.scalp_coupling_index(od)
                    od.info['bads'] = list(compress(od.ch_names, sci < 0.15))
                    print('%d channels marked bad: (Scalp Coupling Index < 0.15)' % len(od.info['bads']))
                    print(od.info['bads'])
                    # Filter
                    iir_params = mne.filter.create_filter(od.get_data(), od.info['sfreq'], 0.05, 0.7, method='iir', verbose=False, iir_params=dict(order=5, ftype='butter'))
                    od_filtered = od.filter(0.05, 0.7, h_trans_bandwidth=0.5, l_trans_bandwidth=0.02, method='iir', verbose=False, iir_params=iir_params)
                    # # Plot filter
                    # plotPath = os.getcwd() + r'\VR_Data' + '\\' + ID + '\\' + entry.name
                    # fig = mne.viz.plot_filter(iir_params, sfreq=od.info['sfreq'], compensate=True, show=False, title='Butterworth IIR Order=5')
                    # fig.savefig(fname=plotPath+'\\'+'NIRS_filter.png', format='png')
                    # beer lambert law
                    haemo = mne.preprocessing.nirs.beer_lambert_law(od_filtered, ppf=6)
                    # Annotations
                    events, event_id = mne.events_from_annotations(haemo)
                    state1sec = events[:,0][events[:,2]==1]/haemo.info['sfreq']
                    state3sec = events[:,0][events[:,2]==2]/haemo.info['sfreq']
                    # Label triggers (with offsets)
                    trialOrder = trialOrders[ID]
                    offsetAngle = offsetAngles[ID]
                    offsetOrder = np.array(offsetAngle)[trialOrder]
                    # Check if > 12 trials otherwise just take 12 after first
                    assert len(state1sec) >= 12, 'state1 < 12 times'
                    assert len(state3sec) >= 12, 'state3 < 12 times'
                    if len(state1sec) > 12: # usually is
                        StimName = ['fakeFirst']
                        RotationName = ['fakeFirst']
                    else:
                        StimName = []
                        RotationName = []
                    for i in np.absolute(offsetOrder[:12]):
                        StimName.append('Stimulus_%d' % i)
                        RotationName.append('Rotation_%d' % i)
                    # add anotations to raw
                    annotations = mne.Annotations(onset=np.concatenate([state1sec[:len(StimName)],state3sec[:len(RotationName)]]), duration=0, description=StimName + RotationName)
                    haemo.set_annotations(annotations)
                    if trials == 0:
                        proc_NIRS_easy.append(haemo)
                    elif trials == 1:
                        proc_NIRS_hard.append(haemo)
    assert len(proc_NIRS_easy) == len(proc_NIRS_hard), 'procesed easy and hard trials are not the same len'
    proc_NIRS_lists = [proc_NIRS_easy, proc_NIRS_hard]
    assert len(proc_NIRS_lists[0]) == len(both_NIRS_lists[0]), 'procesed list not the same len as raw list'
    assert len(proc_NIRS_lists) == len(both_NIRS_lists), 'procesed list not the same len as raw list'
    print('NIRS data has been preprocessed')
    assert len(proc_NIRS_lists) == 2, 'NIRS list (trials) is > 2'
    return proc_NIRS_lists

def epoch_EEG(proc_tempo_EEG_lists, proc_spect_EEG_lists):
    '''
    Epoch EEG data from MNE raw

    Output:
    2 lists of MNE Epoch objects
    '''
    for typeCount, EEG_list in enumerate([proc_tempo_EEG_lists, proc_spect_EEG_lists]):
        easyEpochs = []
        hardEpochs = []
        for trials in range(len(EEG_list)):
            for ID in range(len(EEG_list[trials])):
                EEG_raw = EEG_list[trials][ID]
                events, event_id = mne.events_from_annotations(EEG_raw, verbose=False)
                if 'fakeFirst' in event_id:
                    del event_id['fakeFirst']
                # reject_criteria = dict(eeg=50)
                Epochs = mne.Epochs(EEG_raw, events=events, event_id=event_id, tmin=-0.4, tmax=1.2, baseline=(-0.4,0), preload = True, verbose=False)
                if trials == 0:
                    easyEpochs.append(Epochs)
                elif trials == 1:
                    hardEpochs.append(Epochs)
        if typeCount == 0: #temporal
            tempo_EEG_epochs_list = [easyEpochs, hardEpochs]
        elif typeCount == 1: #spectral
            spect_EEG_epochs_list = [easyEpochs, hardEpochs]
    print('Finished creating EEG Epochs')
    return tempo_EEG_epochs_list, spect_EEG_epochs_list

def epoch_NIRS(proc_NIRS_lists):
    '''
    Epoch NIRS data from MNE raw

    Output:
    2 lists of MNE Epoch objects
    '''
    easyEpochs = []
    hardEpochs = []
    for trials in range(len(proc_NIRS_lists)):
        for IDcount in range(len(proc_NIRS_lists[trials])):
            NIRS_raw = proc_NIRS_lists[trials][IDcount]
            events, event_id = mne.events_from_annotations(NIRS_raw)
            if 'fakeFirst' in event_id:
                del event_id['fakeFirst']
            # reject_criteria = dict(hbo=80e-5)
            tmin, tmax = -30, 28
            Epochs = mne.Epochs(NIRS_raw, events, event_id=event_id,
                                tmin=tmin, tmax=tmax,
                                # reject=reject_criteria, 
                                reject_by_annotation=False,
                                proj=True, baseline=(None, 0), preload=True,
                                detrend=None, verbose=False)
            if trials == 0:
                easyEpochs.append(Epochs)
            elif trials == 1:
                hardEpochs.append(Epochs)
    NIRS_epoch_list = [easyEpochs, hardEpochs]
    print('Finished creating NIRS Epochs')
    return NIRS_epoch_list

def plot_EEG_ERPs(filepath, IDs, tempo_EEG_epochs_list):
    '''
    Create EEG ERP plots
    save them to the appropriate directory
    '''
    assert len(IDs) == len(tempo_EEG_epochs_list[0]) == len(tempo_EEG_epochs_list[1]), 'ID list should equal length of data lists'
    for IDcount, ID in enumerate(IDs):
        with os.scandir(filepath + '\\' + ID) as entries:
            for entry in entries:
                print('Plotting EEG ERP figures for %s' % ID)
                plotPath = filepath + '\\' + ID + '\\' + entry.name
                epochs = mne.concatenate_epochs([tempo_EEG_epochs_list[0][IDcount], tempo_EEG_epochs_list[1][IDcount]])
                
                # isolate rotation and stimuli epochs
                rotEpochs = epochs[[s for s in epochs.event_id if "Rotation" in s]]
                stiEpochs = epochs[[s for s in epochs.event_id if "Stimulus" in s]]
                
                # plot topo_ERP (joint)
                fig = rotEpochs.copy().average().plot_joint(title=ID + ' all rotation trials averaged', show=False, times=[0,0.2,0.4,0.6])
                fig.savefig(fname=plotPath+'\\'+'eegRotationTopo.png', format='png')
                fig = stiEpochs.copy().average().plot_joint(title=ID + ' all stimulus trials averaged', show=False, times=[0,0.2,0.4,0.6])
                fig.savefig(fname=plotPath+'\\'+'eegStimulusTopo.png', format='png')
                evokeds = dict(
                    Small_angle = list(epochs['Rotation_0', 'Rotation_45', 'Rotation_30'].iter_evoked()),
                    Large_angle = list(epochs['Rotation_180', 'Rotation_130'].iter_evoked()))
                fig = mne.viz.plot_compare_evokeds(evokeds=evokeds, combine='mean', picks = ['P3', 'P4', 'P5', 'P6'], ci=True, vlines='auto', show=False, title= ID+' P3, P4, P5, P6 ')#,ylim=dict(eeg=[-8e6, 8e6]))
                fig[0].savefig(fname=plotPath+'\\'+'eegAngleComparison.png', format='png')
                
                # Plot topomaps
                topomap_args = dict(extrapolate='local')
                fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(9, 3),
                                        gridspec_kw=dict(width_ratios=[1, 1, 1, 0.1]))
                vmin, vmax, ts = None, None, 0.2
                evoked_low = epochs['Rotation_0', 'Rotation_45', 'Rotation_30'].average()
                evoked_high = epochs['Rotation_180', 'Rotation_130'].average()
                evoked_low.plot_topomap(times=ts, axes=axes[0],
                                        vmin=vmin, vmax=vmax, colorbar=False,
                                        **topomap_args, show=False)

                evoked_high.plot_topomap(times=ts, axes=axes[1],
                                        vmin=vmin, vmax=vmax, colorbar=False,
                                        **topomap_args, show=False)
                evoked_diff = mne.combine_evoked([evoked_low, evoked_high], weights=[1, -1])
                evoked_diff.plot_topomap(times=ts, axes=axes[2:],
                                        vmin=vmin, vmax=vmax, colorbar=True,
                                        **topomap_args, show=False)
                for column, condition in enumerate(['Low angle', 'High Angle', 'Low - High']):
                        axes[column].set_title('{}'.format(condition))
                fig.tight_layout()
                fig.savefig(fname=plotPath+'\\'+'EEG_topoplots.png', format='png')
                
                # Quantify and compare ERPs with windowed integrals 
                smallEpochs = epochs['Rotation_0', 'Rotation_45', 'Rotation_30'].copy().pick(['P3', 'P4', 'P5', 'P6'])
                largeEpochs = epochs['Rotation_180', 'Rotation_130'].copy().pick(['P3', 'P4', 'P5', 'P6'])
                Vals = []
                Types = []
                Feat = []
                sfreq = smallEpochs.info['sfreq']
                for epoch in smallEpochs:
                    Vals.append(epoch[:,int(sfreq*(0.4+0)):int(sfreq*(0.4+0.1))].sum(axis=1).mean())
                    Types.append('small')
                    Feat.append('P1')
                    Vals.append((epoch[:,int(sfreq*(0.4+0.1)):int(sfreq*(0.4+0.3))]/2).sum(axis=1).mean())
                    Types.append('small')
                    Feat.append('N1')
                    Vals.append((epoch[:,int(sfreq*(0.4+0.3)):int(sfreq*(0.4+0.6))]/3).sum(axis=1).mean())
                    Types.append('small')
                    Feat.append('P3')
                for epoch in largeEpochs:
                    Vals.append(epoch[:,int(sfreq*(0.4+0)):int(sfreq*(0.4+0.1))].sum(axis=1).mean())
                    Types.append('large')
                    Feat.append('P1')
                    Vals.append((epoch[:,int(sfreq*(0.4+0.1)):int(sfreq*(0.4+0.3))]/2).sum(axis=1).mean())
                    Types.append('large')
                    Feat.append('N1')
                    Vals.append((epoch[:,int(sfreq*(0.4+0.3)):int(sfreq*(0.4+0.6))]/3).sum(axis=1).mean())
                    Types.append('large')
                    Feat.append('P3')
                erp_df = pd.DataFrame(np.vstack((Vals,Types,Feat)).T, columns = ['Value', 'Angle', 'Feature'], index=range(len(Vals)))
                erp_df['Value'] = erp_df['Value'].astype(float)
                fig, ax = plt.subplots()
                sns.boxplot(data=erp_df, x='Feature',y='Value', hue='Angle', ax=ax)
                fig.savefig(fname=plotPath+'\\'+'EEG_ERP_boxplots.png', format='png')
                print('Figures saved to: ' + plotPath)
                plt.close('all')
    return None

def plot_EEG_TFRs(filepath, IDs, spect_EEG_epochs_list):
    '''
    Create EEG TFR plots
    save them to the appropriate directory
    '''
    assert len(IDs) == len(spect_EEG_epochs_list[0]) == len(spect_EEG_epochs_list[1]), 'ID list should equal length of data lists'
    for IDcount, ID in enumerate(IDs):
        with os.scandir(filepath + '\\' + ID) as entries:
            for entry in entries:
                print('Plotting EEG TFR figures for %s' % ID)
                plotPath = filepath + '\\' + ID + '\\' + entry.name
                epochs = mne.concatenate_epochs([spect_EEG_epochs_list[0][IDcount], spect_EEG_epochs_list[1][IDcount]])
                # endRTs = both_endRT_lists[0][IDcount] + both_endRT_lists[1][IDcount]
                
                # isolate rotation and stimuli epochs
                rotEpochs = epochs[[s for s in epochs.event_id if "Rotation" in s]]
                stiEpochs = epochs[[s for s in epochs.event_id if "Stimulus" in s]]
                
                # plot spectrograms
                freqs = np.arange(3,40, 0.5)
                Lpower, itc = mne.time_frequency.tfr_multitaper(epochs['Rotation_0', 'Rotation_45', 'Rotation_30'], freqs=freqs, n_cycles=freqs/2)
                Hpower, itc = mne.time_frequency.tfr_multitaper(epochs['Rotation_180', 'Rotation_130'], freqs=freqs, n_cycles=freqs/2)
                fig = Lpower.plot(baseline = (None,0), combine='mean', picks=['P3', 'P4', 'P5', 'P6'], vmin=-500, vmax=500, title=ID + ' Low angle trials averaged', show=False, verbose=False)
                fig[0].savefig(fname=plotPath+'\\'+'lowTFR.png', format='png')
                fig = Hpower.plot(baseline = (None,0), combine='mean', picks=['P3', 'P4', 'P5', 'P6'], vmin=-500, vmax=500, title=ID + ' High angle trials averaged', show=False, verbose=False)
                fig[0].savefig(fname=plotPath+'\\'+'highTFR.png', format='png')
                power = Hpower - Lpower
                fig = power.plot(baseline = (None,0), combine='mean', picks=['P3', 'P4', 'P5', 'P6'], vmin=-500, vmax=500, title=ID + ' High-Low angle trials', show=False, verbose=False)
                fig[0].savefig(fname=plotPath+'\\'+'highLowDiffTFR.png', format='png')
                
                # rotation and stimuli spectrograms
                rotPower, itc = mne.time_frequency.tfr_multitaper(rotEpochs, freqs=freqs, n_cycles=freqs/2)
                stiPower, itc = mne.time_frequency.tfr_multitaper(stiEpochs, freqs=freqs, n_cycles=freqs/2)
                fig = rotPower.plot(baseline = (None,0), combine='mean', picks=['P3', 'P4', 'P5', 'P6'], vmin=-500, vmax=500, title=ID + ' Rotation trials averaged', show=False, verbose=False)
                fig[0].savefig(fname=plotPath+'\\'+'rotationTFR.png', format='png')
                fig = stiPower.plot(baseline = (None,0), combine='mean', picks=['P3', 'P4', 'P5', 'P6'], vmin=-500, vmax=500, title=ID + ' Stimuli trials averaged', show=False, verbose=False)
                fig[0].savefig(fname=plotPath+'\\'+'stimuliTFR.png', format='png')

                # epochsToCat = []
                # for epoch in rotEpochs:
                #     epoch
                
                print('Figures saved to: ' + plotPath)
                plt.close('all')
    return None

def plot_NIRS(filepath, IDs, NIRS_epoch_list):
    '''
    Create hemodynamic response plots

    save them to the appropriate directory
    '''
    assert len(IDs) == len(NIRS_epoch_list[0]) == len(NIRS_epoch_list[1]), 'ID list should equal length of data lists'
    for IDcount, ID in enumerate(IDs):
        with os.scandir(filepath + '\\' + ID) as entries:
            for entry in entries:
                print('Plotting NIRS figures for %s' % ID)
                plotPath = filepath + '\\' + ID + '\\' + entry.name
                
                # Join epochs
                epochs_Hard_copy = NIRS_epoch_list[1][IDcount].copy()
                epochs_Easy_copy = NIRS_epoch_list[0][IDcount].copy()
                epochs_Hard_copy.info['bads'] = epochs_Easy_copy.info['bads']
                epochs = mne.concatenate_epochs([epochs_Easy_copy, epochs_Hard_copy])
                
                # isolate rotation and stimuli epochs
                rotEpochs = epochs[[s for s in epochs.event_id if "Rotation" in s]]
                stiEpochs = epochs[[s for s in epochs.event_id if "Stimulus" in s]]
                
                # plot joint (with topo)
                fig = rotEpochs.copy().average().plot_joint(title=ID + ' all rotation trials averaged (HbO)', show=False, times=[0,5,10,18], picks='hbo')
                fig.savefig(fname=plotPath+'\\'+'NIRS_AllRotation_HbO_topo.png', format='png')
                fig = stiEpochs.copy().average().plot_joint(title=ID + ' all stimulus trials averaged (HbR)', show=False, times=[0,5,10,18], picks='hbo')
                fig.savefig(fname=plotPath+'\\'+'NIRS_AllStimulus_HbO_topo.png', format='png')
                fig = rotEpochs.copy().average().plot_joint(title=ID + ' all rotation trials averaged (HbO)', show=False, times=[0,5,10,18], picks='hbr')
                fig.savefig(fname=plotPath+'\\'+'NIRS_AllRotation_HbR_topo.png', format='png')
                fig = stiEpochs.copy().average().plot_joint(title=ID + ' all stimulus trials averaged (HbR)', show=False, times=[0,5,10,18], picks='hbr')
                fig.savefig(fname=plotPath+'\\'+'NIRS_AllStimulus_HbR_topo.png', format='png')

                # Plot image (stacked ERP) for consistency check
                fig = rotEpochs.plot_image(combine='mean', title=ID+ ' all trials', show=False, ts_args=dict(vlines=[-19, -16, -13, -5, 0]))
                fig[0].axes[1].set_xticks([-20, -19, -16, -13, -10, -5, 0, 10, 20])
                fig[0].axes[1].set_xticklabels(['-20s', '', 'audio', 'object1', '-10s', 'audio', 'object2 (0s)', '10s', '20s'], rotation='vertical')
                fig[0].tight_layout()
                fig[0].savefig(fname=plotPath+'\\'+'NIRS_all_HbO_image.png', format='png')
                fig[1].axes[1].set_xticks([-20, -19, -16, -13, -10, -5, 0, 10, 20])
                fig[1].axes[1].set_xticklabels(['-20s', '', 'audio', 'object1', '-10s', 'audio', 'object2 (0s)', '10s', '20s'], rotation='vertical')
                fig[1].tight_layout()
                fig[1].savefig(fname=plotPath+'\\'+'NIRS_all_HbR_image.png', format='png')
                fig = rotEpochs['Rotation_0', 'Rotation_45', 'Rotation_30'].plot_image(combine='mean', title=ID+ ' low angle', show=False, ts_args=dict(vlines=[-19, -16, -13, -5, 0]))
                fig[0].axes[1].set_xticks([-20, -19, -16, -13, -10, -5, 0, 10, 20])
                fig[0].axes[1].set_xticklabels(['-20s', '', 'audio', 'object1', '-10s', 'audio', 'object2 (0s)', '10s', '20s'], rotation='vertical')
                fig[0].tight_layout()
                fig[0].savefig(fname=plotPath+'\\'+'NIRS_low_HbO_image.png', format='png')
                fig[1].axes[1].set_xticks([-20, -19, -16, -13, -10, -5, 0, 10, 20])
                fig[1].axes[1].set_xticklabels(['-20s', '', 'audio', 'object1', '-10s', 'audio', 'object2 (0s)', '10s', '20s'], rotation='vertical')
                fig[1].tight_layout()
                fig[1].savefig(fname=plotPath+'\\'+'NIRS_low_HbR_image.png', format='png')
                fig = rotEpochs['Rotation_180', 'Rotation_130'].plot_image(combine='mean', title=ID+ ' high angle', show=False, ts_args=dict(vlines=[-19, -16, -13, -5, 0]))
                fig[0].axes[1].set_xticks([-20, -19, -16, -13, -10, -5, 0, 10, 20])
                fig[0].axes[1].set_xticklabels(['-20s', '', 'audio', 'object1', '-10s', 'audio', 'object2 (0s)', '10s', '20s'], rotation='vertical')
                fig[0].tight_layout()
                fig[0].savefig(fname=plotPath+'\\'+'NIRS_high_HbO_image.png', format='png')
                fig[1].axes[1].set_xticks([-20, -19, -16, -13, -10, -5, 0, 10, 20])
                fig[1].axes[1].set_xticklabels(['-20s', '', 'audio', 'object1', '-10s', 'audio', 'object2 (0s)', '10s', '20s'], rotation='vertical')
                fig[1].tight_layout()
                fig[1].savefig(fname=plotPath+'\\'+'NIRS_high_HbR_image.png', format='png')

                # Plot classic comparison
                fig, ax = plt.subplots(2,1, figsize=[9,10])
                evoked_dict = { 'Low/HbO': list(epochs['Rotation_0', 'Rotation_45', 'Rotation_30'].copy().pick('hbo').iter_evoked()),
                        'High/HbO' : list(epochs['Rotation_180', 'Rotation_130'].copy().pick('hbo').iter_evoked())}
                color_dict = dict(HbO='#AA3377')
                styles_dict = dict(Low=dict(linestyle='dashed'))
                mne.viz.plot_compare_evokeds(evoked_dict, combine="mean", ci=0.5 ,colors=color_dict, styles=styles_dict, show=False, title=ID+' HbO', vlines=[-19, -16, -13, -5, 0], axes=ax[0])
                
                # Plot classic hbr comparison
                evoked_dict = { 'Low/HbR': list(epochs['Rotation_0', 'Rotation_45', 'Rotation_30'].copy().pick('hbr').iter_evoked()),
                                'High/HbR' : list(epochs['Rotation_180', 'Rotation_130'].copy().pick('hbr').iter_evoked())}
                color_dict = dict(HbR='b')
                styles_dict = dict(Low=dict(linestyle='dashed'))
                mne.viz.plot_compare_evokeds(evoked_dict, combine="mean", ci=0.5 ,colors=color_dict, styles=styles_dict, show=False, title=ID+' HbR', vlines=[-19, -16, -13, -5, 0], axes=ax[1])
                
                # Tidy up the axes
                ax[1].set_xticklabels(['-20s', '', 'audio', 'object1', '-10s', 'audio', 'object2 (0s)', '10s', '20s'], rotation='vertical')
                ax[0].set_xticklabels('')
                for x in ax: x.axvspan(1, 4, alpha=0.3)
                plt.tight_layout()
                fig.savefig(fname=plotPath+'\\'+'NIRS_Both_comparison.png', format='png')

                # plot classic Stimulus comparison
                evoked_dict = { 'Stimulus/HbO': stiEpochs.average(picks='hbo'),
                                'Stimulus/HbR': stiEpochs.average(picks='hbr')}
                
                # Rename channels until the encoding of frequency in ch_name is fixed
                for condition in evoked_dict:
                    evoked_dict[condition].rename_channels(lambda x: x[:-4])
                color_dict = dict(HbO='#AA3377', HbR='b')
                fig = mne.viz.plot_compare_evokeds(evoked_dict, combine="mean", ci=True ,colors=color_dict, styles=None, show=False, title=ID)
                fig[0].savefig(fname=plotPath+'\\'+'NIRS_Stimulus.png', format='png')
                
                # Plot topomaps
                topomap_args = dict(extrapolate='local')
                fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(9, 3),
                                        gridspec_kw=dict(width_ratios=[1, 1, 1, 0.1]))
                vmin, vmax, ts = None, None, 10
                evoked_low = epochs['Rotation_0', 'Rotation_45', 'Rotation_30'].average()
                evoked_high = epochs['Rotation_180', 'Rotation_130'].average()
                evoked_low.plot_topomap(ch_type='hbo', times=ts, axes=axes[0, 0],
                                        vmin=vmin, vmax=vmax, colorbar=False,
                                        **topomap_args, show=False)
                evoked_low.plot_topomap(ch_type='hbr', times=ts, axes=axes[1, 0],
                                        vmin=vmin, vmax=vmax, colorbar=False,
                                        **topomap_args, show=False)
                evoked_high.plot_topomap(ch_type='hbo', times=ts, axes=axes[0, 1],
                                        vmin=vmin, vmax=vmax, colorbar=False,
                                        **topomap_args, show=False)
                evoked_high.plot_topomap(ch_type='hbr', times=ts, axes=axes[1, 1],
                                        vmin=vmin, vmax=vmax, colorbar=False,
                                        **topomap_args, show=False)
                evoked_diff = mne.combine_evoked([evoked_low, evoked_high], weights=[1, -1])
                evoked_diff.plot_topomap(ch_type='hbo', times=ts, axes=axes[0, 2:],
                                        vmin=vmin, vmax=vmax, colorbar=True,
                                        **topomap_args, show=False)
                evoked_diff.plot_topomap(ch_type='hbr', times=ts, axes=axes[1, 2:],
                                        vmin=vmin, vmax=vmax, colorbar=True,
                                        **topomap_args, show=False)
                for column, condition in enumerate(
                        ['Low angle', 'High Angle', 'Low - High']):
                    for row, chroma in enumerate(['HbO', 'HbR']):
                        axes[row, column].set_title('{}: {}'.format(chroma, condition))
                fig.tight_layout()
                fig.savefig(fname=plotPath+'\\'+'NIRS_topoplots.png', format='png')
                
                # Plot topo ERPs for HbO
                evoked = epochs.copy().pick(picks='hbo').average()
                from mne.channels.layout import find_layout
                layout = find_layout(evoked.info)
                pos = layout.pos.copy()
                f = plt.figure(figsize=[12,8])
                evokeds = dict(
                    Small_angle = list(epochs['Rotation_0', 'Rotation_45', 'Rotation_30'].pick(picks='hbo').crop(tmin=-10, tmax=20).iter_evoked()),
                    Large_angle = list(epochs['Rotation_180', 'Rotation_130'].pick(picks='hbo').crop(tmin=-10, tmax=20).iter_evoked()))
                ylims = (evoked.data.min() * 1e6,
                         evoked.data.max() * 1e6)
                ymax = np.min(np.abs(np.array(ylims)))
                for pick, (pos_, ch_name) in enumerate(zip(pos, evoked.ch_names)):
                    pos_scaler = .8
                    pos_offset = .05
                    plot_size_x_scaler = 2
                    plot_size_y_scaler = 1
                    pos_ = [pos_[0]*pos_scaler+pos_offset, pos_[1]*pos_scaler+pos_offset, pos_[2]*plot_size_x_scaler, pos_[3]*plot_size_y_scaler]
                    ax = plt.axes(pos_)
                    if ch_name not in evoked.info['bads']:
                        mne.viz.plot_compare_evokeds(evokeds, picks=pick, axes=ax,
                                            ylim=dict(hbo=ylims),
                                            show=False,
                                            show_sensors=False,
                                            legend=False,
                                            title='', ci=0.5)
                    ax.set_xticklabels(())
                    ax.set_ylabel('')
                    ax.set_xlabel('')
                    ax.set_yticks((-ymax, ymax))
                    ax.spines["left"].set_bounds(-ymax, ymax)
                    ax.set_ylim(ylims)
                    ax.set_yticklabels('')
                    ax.text(-.1, 1, ch_name, transform=ax.transAxes)
                ax_l = plt.axes([.45, .4, .07*plot_size_x_scaler, .05*plot_size_y_scaler])
                mne.viz.plot_compare_evokeds(evokeds, ylim=dict(hbo=ylims), title='', show_sensors=False,
                                    picks=0, axes=ax_l, ci=None,
                                            show=False)
                ax_l.set_yticks((-ymax, ymax))
                ax_l.spines["left"].set_bounds(-ymax, ymax)
                ax_l.set_ylim(ylims)
                ax_l.lines.clear()
                ax_l.patches.clear()
                ax_l.get_legend().remove()
                f.legend(['Small angle', 'Large Angle'], loc=8)
                f.suptitle(ID + ' comparison (HbO)')
                f.savefig(fname=plotPath+'\\'+'NIRS_HbO_ERP_topolayout.png', format='png')
                
                # Plot topo ERPs for HbR
                evoked = epochs.copy().pick(picks='hbr').average()
                from mne.channels.layout import find_layout
                layout = find_layout(evoked.info)
                pos = layout.pos.copy()
                f = plt.figure(figsize=[12,8])
                evokeds = dict(
                    Small_angle = list(epochs['Rotation_0', 'Rotation_45', 'Rotation_30'].pick(picks='hbr').crop(tmin=-10, tmax=20).iter_evoked()),
                    Large_angle = list(epochs['Rotation_180', 'Rotation_130'].pick(picks='hbr').crop(tmin=-10, tmax=20).iter_evoked()))
                ylims = (evoked.data.min() * 1e6,
                         evoked.data.max() * 1e6)
                ymax = np.min(np.abs(np.array(ylims)))
                for pick, (pos_, ch_name) in enumerate(zip(pos, evoked.ch_names)):
                    pos_scaler = .8
                    pos_offset = .05
                    plot_size_x_scaler = 2
                    plot_size_y_scaler = 1
                    pos_ = [pos_[0]*pos_scaler+pos_offset, pos_[1]*pos_scaler+pos_offset, pos_[2]*plot_size_x_scaler, pos_[3]*plot_size_y_scaler]
                    ax = plt.axes(pos_)
                    if ch_name not in evoked.info['bads']:
                        mne.viz.plot_compare_evokeds(evokeds, picks=pick, axes=ax,
                                            ylim=dict(hbr=ylims),
                                            show=False,
                                            show_sensors=False,
                                            legend=False,
                                            title='', ci=0.5)
                    ax.set_xticklabels(())
                    ax.set_ylabel('')
                    ax.set_xlabel('')
                    ax.set_yticks((-ymax, ymax))
                    ax.spines["left"].set_bounds(-ymax, ymax)
                    ax.set_ylim(ylims)
                    ax.set_yticklabels('')
                    ax.text(-.1, 1, ch_name, transform=ax.transAxes)
                ax_l = plt.axes([.45, .4, .07*plot_size_x_scaler, .05*plot_size_y_scaler])
                mne.viz.plot_compare_evokeds(evokeds, ylim=dict(hbr=ylims), title='', show_sensors=False,
                                        picks=0, axes=ax_l, ci=None,
                                            show=False)
                ax_l.set_yticks((-ymax, ymax))
                ax_l.spines["left"].set_bounds(-ymax, ymax)
                ax_l.set_ylim(ylims)
                ax_l.lines.clear()
                ax_l.patches.clear()
                ax_l.get_legend().remove()
                f.legend(['Small angle', 'Large Angle'], loc=8)
                f.suptitle(ID + ' comparison (hbr)')
                f.savefig(fname=plotPath+'\\'+'NIRS_HbR_ERP_topolayout.png', format='png')

                print('Figures saved to: ' + plotPath)
                plt.close('all')
    return None

def plot_NIRS_wBHV(filepath, IDs, NIRS_epoch_list, both_bhv_lists, both_RT_lists, both_endRT_lists):
    '''
    Create hemodynamic response plots with behavioral responses included
    Also plot hemodynamic responses in ROIs

    save them to the appropriate directory
    '''
    assert len(IDs) == len(NIRS_epoch_list[0]) == len(NIRS_epoch_list[1]), 'ID list should equal length of data lists'
    for IDcount, ID in enumerate(IDs):
        with os.scandir(filepath + '\\' + ID) as entries:
            for entry in entries:
                print('Plotting Behvaioral + more NIRS figures for %s' % ID)
                plotPath = filepath + '\\' + ID + '\\' + entry.name
                
                # Join epochs
                epochs_Hard_copy = NIRS_epoch_list[1][IDcount].copy()
                epochs_Easy_copy = NIRS_epoch_list[0][IDcount].copy()
                epochs_Hard_copy.info['bads'] = epochs_Easy_copy.info['bads']
                epochs = mne.concatenate_epochs([epochs_Easy_copy, epochs_Hard_copy])
                endRTs = both_endRT_lists[0][IDcount] + both_endRT_lists[1][IDcount]
                
                # isolate rotation and stimuli epochs
                rotEpochs = epochs[[s for s in epochs.event_id if "Rotation" in s]]
                stiEpochs = epochs[[s for s in epochs.event_id if "Stimulus" in s]]
                
                # Plot behavioral data
                for x in both_bhv_lists[1][IDcount]+both_bhv_lists[0][IDcount]: 
                    plt.plot(x)
                plt.xticks(ticks=np.arange(9)*90, labels=np.arange(9))
                plt.xlabel('Seconds')
                plt.ylabel('Degrees')
                plt.suptitle('Object orientation (in degrees)')
                plt.savefig(fname=plotPath+'\\'+'BehavioralData.png', format='png')
                
                # Image plots but with Reaciton times
                fig = rotEpochs.plot_image(combine='mean', title=ID+ ' all trials', show=False, ts_args=dict(vlines=[-19, -16, -13, -5, 0]), overlay_times=endRTs)
                fig[0].axes[1].set_xticks([-20, -19, -16, -13, -10, -5, 0, 10, 20])
                fig[0].axes[1].set_xticklabels(['-20s', '', 'audio', 'object1', '-10s', 'audio', 'object2 (0s)', '10s', '20s'], rotation='vertical')
                fig[0].tight_layout()
                fig[0].savefig(fname=plotPath+'\\'+'NIRS_endRT_all_HbO_image.png', format='png')
                fig[1].axes[1].set_xticks([-20, -19, -16, -13, -10, -5, 0, 10, 20])
                fig[1].axes[1].set_xticklabels(['-20s', '', 'audio', 'object1', '-10s', 'audio', 'object2 (0s)', '10s', '20s'], rotation='vertical')
                fig[1].tight_layout()
                fig[1].savefig(fname=plotPath+'\\'+'NIRS_endRT_all_HbR_image.png', format='png')

                # Pick ROIs
                PreFrontalEpochs = rotEpochs.copy().pick([s for s in rotEpochs.ch_names if 'S5_D4' in s]+[s for s in rotEpochs.ch_names if 'S5_D5' in s])
                FrontalEpochs    = rotEpochs.copy().pick([s for s in rotEpochs.ch_names if 'S3_D6' in s]+[s for s in rotEpochs.ch_names if 'S6_D6' in s])
                ParietalEpochs   = rotEpochs.copy().pick([s for s in rotEpochs.ch_names if 'S1_D2' in s]+[s for s in rotEpochs.ch_names if 'S8_D2' in s])
                DlpfcEpochs      = rotEpochs.copy().pick([s for s in rotEpochs.ch_names if 'S2_D3' in s]+[s for s in rotEpochs.ch_names if 'S7_D7' in s])
                
                fig, ax = plt.subplots(4,2, figsize=[9,9])
                # Plot PreFrontal comparison
                evoked_dict = { 'Low/HbO': list(PreFrontalEpochs['Rotation_0', 'Rotation_45', 'Rotation_30'].copy().pick('hbo').iter_evoked()),
                              'High/HbO' : list(PreFrontalEpochs['Rotation_180', 'Rotation_130'].copy().pick('hbo').iter_evoked())}
                color_dict = dict(HbO='#AA3377')
                styles_dict = dict(Low=dict(linestyle='dashed'))
                mne.viz.plot_compare_evokeds(evoked_dict, combine="mean", ci=0.5 ,colors=color_dict, styles=styles_dict, show=False, title=ID+' Pre-Frontal HbO', vlines=[-19, -16, -13, -5, 0], axes=ax[0,0])
                
                # Plot classic hbr comparison
                evoked_dict = { 'Low/HbR': list(PreFrontalEpochs['Rotation_0', 'Rotation_45', 'Rotation_30'].copy().pick('hbr').iter_evoked()),
                                'High/HbR': list(PreFrontalEpochs['Rotation_180', 'Rotation_130'].copy().pick('hbr').iter_evoked())}
                color_dict = dict(HbR='b')
                styles_dict = dict(Low=dict(linestyle='dashed'))
                mne.viz.plot_compare_evokeds(evoked_dict, combine="mean", ci=0.5 ,colors=color_dict, styles=styles_dict, show=False, title=ID+' Pre-Frontal HbR', vlines=[-19, -16, -13, -5, 0], axes=ax[0,1])
                
                # Tidy up the axes
                ax[0,1].set_xticklabels('')
                ax[0,0].set_xticklabels('')

                # Plot Frontal comparison
                evoked_dict = { 'Low/HbO': list(FrontalEpochs['Rotation_0', 'Rotation_45', 'Rotation_30'].copy().pick('hbo').iter_evoked()),
                              'High/HbO' : list(FrontalEpochs['Rotation_180', 'Rotation_130'].copy().pick('hbo').iter_evoked())}
                color_dict = dict(HbO='#AA3377')
                styles_dict = dict(Low=dict(linestyle='dashed'))
                mne.viz.plot_compare_evokeds(evoked_dict, combine="mean", ci=0.5 ,colors=color_dict, styles=styles_dict, show=False, title=ID+' Frontal HbO', vlines=[-19, -16, -13, -5, 0], axes=ax[1,0])
                
                # Plot classic hbr comparison
                evoked_dict = { 'Low/HbR': list(FrontalEpochs['Rotation_0', 'Rotation_45', 'Rotation_30'].copy().pick('hbr').iter_evoked()),
                                'High/HbR': list(FrontalEpochs['Rotation_180', 'Rotation_130'].copy().pick('hbr').iter_evoked())}
                color_dict = dict(HbR='b')
                styles_dict = dict(Low=dict(linestyle='dashed'))
                mne.viz.plot_compare_evokeds(evoked_dict, combine="mean", ci=0.5 ,colors=color_dict, styles=styles_dict, show=False, title=ID+' Frontal HbR', vlines=[-19, -16, -13, -5, 0], axes=ax[1,1])
                
                # Tidy up the axes
                ax[1,1].set_xticklabels('')
                ax[1,0].set_xticklabels('')

                # Plot DLPFC comparison
                evoked_dict = { 'Low/HbO': list(DlpfcEpochs['Rotation_0', 'Rotation_45', 'Rotation_30'].copy().pick('hbo').iter_evoked()),
                              'High/HbO' : list(DlpfcEpochs['Rotation_180', 'Rotation_130'].copy().pick('hbo').iter_evoked())}
                color_dict = dict(HbO='#AA3377')
                styles_dict = dict(Low=dict(linestyle='dashed'))
                mne.viz.plot_compare_evokeds(evoked_dict, combine="mean", ci=0.5 ,colors=color_dict, styles=styles_dict, show=False, title=ID+' DLPFC HbO', vlines=[-19, -16, -13, -5, 0], axes=ax[2,0])
                
                # Plot classic hbr comparison
                evoked_dict = { 'Low/HbR': list(DlpfcEpochs['Rotation_0', 'Rotation_45', 'Rotation_30'].copy().pick('hbr').iter_evoked()),
                                'High/HbR': list(DlpfcEpochs['Rotation_180', 'Rotation_130'].copy().pick('hbr').iter_evoked())}
                color_dict = dict(HbR='b')
                styles_dict = dict(Low=dict(linestyle='dashed'))
                mne.viz.plot_compare_evokeds(evoked_dict, combine="mean", ci=0.5 ,colors=color_dict, styles=styles_dict, show=False, title=ID+' DLPFC HbR', vlines=[-19, -16, -13, -5, 0], axes=ax[2,1])
                
                # Tidy up the axes
                ax[2,1].set_xticklabels('')
                ax[2,0].set_xticklabels('')
                
                # Plot Parietal comparison
                evoked_dict = { 'Low/HbO': list(ParietalEpochs['Rotation_0', 'Rotation_45', 'Rotation_30'].copy().pick('hbo').iter_evoked()),
                              'High/HbO' : list(ParietalEpochs['Rotation_180', 'Rotation_130'].copy().pick('hbo').iter_evoked())}
                color_dict = dict(HbO='#AA3377')
                styles_dict = dict(Low=dict(linestyle='dashed'))
                mne.viz.plot_compare_evokeds(evoked_dict, combine="mean", ci=0.5 ,colors=color_dict, styles=styles_dict, show=False, title=ID+' Parietal HbO', vlines=[-19, -16, -13, -5, 0], axes=ax[3,0])
                
                # Plot classic hbr comparison
                evoked_dict = { 'Low/HbR': list(ParietalEpochs['Rotation_0', 'Rotation_45', 'Rotation_30'].copy().pick('hbr').iter_evoked()),
                                'High/HbR': list(ParietalEpochs['Rotation_180', 'Rotation_130'].copy().pick('hbr').iter_evoked())}
                color_dict = dict(HbR='b')
                styles_dict = dict(Low=dict(linestyle='dashed'))
                mne.viz.plot_compare_evokeds(evoked_dict, combine="mean", ci=0.5 ,colors=color_dict, styles=styles_dict, show=False, title=ID+' Parietal HbR', vlines=[-19, -16, -13, -5, 0], axes=ax[3,1])
                
                # Tidy up the axes
                ax[3,1].set_xticklabels(['-20s', '', 'audio', 'object1', '-10s', 'audio', 'object2 (0s)', '10s', '20s'], rotation='vertical')
                ax[3,0].set_xticklabels(['-20s', '', 'audio', 'object1', '-10s', 'audio', 'object2 (0s)', '10s', '20s'], rotation='vertical')
                for x in range(ax.shape[0]):
                    for y in range(ax.shape[1]): 
                        ax[x,y].axvspan(1, 4, alpha=0.3)
                plt.tight_layout()
                fig.savefig(fname=plotPath+'\\'+'NIRS_all_ROIs.png', format='png')

                # Get end of rotation time for small and large angles
                largeLocsRTs, smallLocsRTs = [], []
                for lergeSel in rotEpochs['Rotation_180', 'Rotation_130'].copy().selection:
                    largeLocsRTs.append(np.where(rotEpochs.copy().selection == lergeSel)[0])
                for smallSel in rotEpochs['Rotation_0', 'Rotation_45', 'Rotation_30'].copy().selection:
                    smallLocsRTs.append(np.where(rotEpochs.copy().selection == smallSel)[0])

                # Quantify and compare ERPs with windowed integrals 
                for ox in ['hbr', 'hbo']:
                    smallEpochs = epochs['Rotation_0', 'Rotation_45', 'Rotation_30'].copy().pick(ox)
                    largeEpochs = epochs['Rotation_180', 'Rotation_130'].copy().pick(ox)
                    
                    smallEpochs_PreFrontal = PreFrontalEpochs['Rotation_0', 'Rotation_45', 'Rotation_30'].copy().pick(ox)
                    largeEpochs_PreFrontal = PreFrontalEpochs['Rotation_180', 'Rotation_130'].copy().pick(ox)
                    
                    smallEpochs_Frontal = FrontalEpochs['Rotation_0', 'Rotation_45', 'Rotation_30'].copy().pick(ox)
                    largeEpochs_Frontal = FrontalEpochs['Rotation_180', 'Rotation_130'].copy().pick(ox)

                    smallEpochs_Parietal = ParietalEpochs['Rotation_0', 'Rotation_45', 'Rotation_30'].copy().pick(ox)
                    largeEpochs_Parietal = ParietalEpochs['Rotation_180', 'Rotation_130'].copy().pick(ox)
                    
                    smallEpochs_Dlpfc = DlpfcEpochs['Rotation_0', 'Rotation_45', 'Rotation_30'].copy().pick(ox)
                    largeEpochs_Dlpfc = DlpfcEpochs['Rotation_180', 'Rotation_130'].copy().pick(ox)

                    Vals = []
                    Types = []
                    Feat = []
                    sfreq = smallEpochs.info['sfreq']
                    for epoch in smallEpochs_PreFrontal:
                        Vals.append(epoch[:,int(sfreq*(30+5)):int(sfreq*(30+15))].sum(axis=1).mean())
                        Types.append('small')
                        Feat.append('PreFrontal')
                    for epoch in smallEpochs_Frontal:
                        Vals.append(epoch[:,int(sfreq*(30+5)):int(sfreq*(30+15))].sum(axis=1).mean())
                        Types.append('small')
                        Feat.append('Frontal')
                    for epoch in smallEpochs_Parietal:
                        Vals.append(epoch[:,int(sfreq*(30+5)):int(sfreq*(30+15))].sum(axis=1).mean())
                        Types.append('small')
                        Feat.append('Parietal')
                    for epoch in smallEpochs_Dlpfc:
                        Vals.append(epoch[:,int(sfreq*(30+5)):int(sfreq*(30+15))].sum(axis=1).mean())
                        Types.append('small')
                        Feat.append('dlPFC')
                    for epoch in largeEpochs_PreFrontal:
                        Vals.append(epoch[:,int(sfreq*(30+5)):int(sfreq*(30+15))].sum(axis=1).mean())
                        Types.append('large')
                        Feat.append('PreFrontal')
                    for epoch in largeEpochs_Frontal:
                        Vals.append(epoch[:,int(sfreq*(30+5)):int(sfreq*(30+15))].sum(axis=1).mean())
                        Types.append('large')
                        Feat.append('Frontal')
                    for epoch in largeEpochs_Parietal:
                        Vals.append(epoch[:,int(sfreq*(30+5)):int(sfreq*(30+15))].sum(axis=1).mean())
                        Types.append('large')
                        Feat.append('Parietal')
                    for epoch in largeEpochs_Dlpfc:
                        Vals.append(epoch[:,int(sfreq*(30+5)):int(sfreq*(30+15))].sum(axis=1).mean())
                        Types.append('large')
                        Feat.append('dlPFC')
                    erp_df = pd.DataFrame(np.vstack((Vals,Types,Feat)).T, columns = ['Value', 'Angle', 'Feature'], index=range(len(Vals)))
                    erp_df['Value'] = erp_df['Value'].astype(float)
                    fig, ax = plt.subplots()
                    sns.boxplot(data=erp_df, x='Feature',y='Value', hue='Angle', ax=ax)
                    fig.suptitle(ox + ' 5-15s ' + ID)
                    fig.savefig(fname=plotPath+'\\'+'NIRS_'+ox+'_boxplots.png', format='png')
                    
                    # Same box plots but for a window just after the rotation
                    Vals = []
                    Types = []
                    Feat = []
                    sfreq = smallEpochs.info['sfreq']
                    for count,epoch in enumerate(smallEpochs_PreFrontal):
                        RTsec = smallLocsRTs[count]
                        Vals.append(epoch[:,int(sfreq*(30+RTsec)):int(sfreq*(30+RTsec+10))].sum(axis=1).mean())
                        Types.append('small')
                        Feat.append('PreFrontal')
                    for count,epoch in enumerate(smallEpochs_Frontal):
                        RTsec = smallLocsRTs[count]
                        Vals.append(epoch[:,int(sfreq*(30+RTsec)):int(sfreq*(30+RTsec+10))].sum(axis=1).mean())
                        Types.append('small')
                        Feat.append('Frontal')
                    for count,epoch in enumerate(smallEpochs_Parietal):
                        RTsec = smallLocsRTs[count]
                        Vals.append(epoch[:,int(sfreq*(30+RTsec)):int(sfreq*(30+RTsec+10))].sum(axis=1).mean())
                        Types.append('small')
                        Feat.append('Parietal')
                    for count,epoch in enumerate(smallEpochs_Dlpfc):
                        RTsec = smallLocsRTs[count]
                        Vals.append(epoch[:,int(sfreq*(30+RTsec)):int(sfreq*(30+RTsec+10))].sum(axis=1).mean())
                        Types.append('small')
                        Feat.append('dlPFC')
                    for count,epoch in enumerate(largeEpochs_PreFrontal):
                        RTsec = largeLocsRTs[count]
                        Vals.append(epoch[:,int(sfreq*(30+RTsec)):int(sfreq*(30+RTsec+10))].sum(axis=1).mean())
                        Types.append('large')
                        Feat.append('PreFrontal')
                    for count,epoch in enumerate(largeEpochs_Frontal):
                        RTsec = largeLocsRTs[count]
                        Vals.append(epoch[:,int(sfreq*(30+RTsec)):int(sfreq*(30+RTsec+10))].sum(axis=1).mean())
                        Types.append('large')
                        Feat.append('Frontal')
                    for count,epoch in enumerate(largeEpochs_Parietal):
                        RTsec = largeLocsRTs[count]
                        Vals.append(epoch[:,int(sfreq*(30+RTsec)):int(sfreq*(30+RTsec+10))].sum(axis=1).mean())
                        Types.append('large')
                        Feat.append('Parietal')
                    for count,epoch in enumerate(largeEpochs_Dlpfc):
                        RTsec = largeLocsRTs[count]
                        Vals.append(epoch[:,int(sfreq*(30+RTsec)):int(sfreq*(30+RTsec+10))].sum(axis=1).mean())
                        Types.append('large')
                        Feat.append('dlPFC')
                    erp_df = pd.DataFrame(np.vstack((Vals,Types,Feat)).T, columns = ['Value', 'Angle', 'Feature'], index=range(len(Vals)))
                    erp_df['Value'] = erp_df['Value'].astype(float)
                    fig, ax = plt.subplots()
                    sns.boxplot(data=erp_df, x='Feature',y='Value', hue='Angle', ax=ax)
                    fig.suptitle(ox + ' 0-10s after rotation ends ' + ID)
                    fig.savefig(fname=plotPath+'\\'+'NIRS_RTs_'+ox+'_boxplots.png', format='png')

                print('Figures saved to: ' + plotPath)
                plt.close('all')
    return None



if __name__ == "__main__":
    # Initialize the trial orders
    stimAngles = {
        "James": [45, 45, 45,
                  90, 90, 90,
                  135, 135, 135,
                  180, 180, 180],
        "Subject1": [0, 0, 0, 0, 0, 0, 0, 0,
                     45, 45, 45, 45, 45, 45, 45, 45,
                     90, 90, 90, 90, 90, 90, 90, 90,
                     130, 130, 130, 130, 130, 130, 130, 130,
                     180, 180, 180, 180, 180, 180, 180, 180,
                     225, 225, 225, 225, 225, 225, 225, 225,
                     270, 270, 270, 270, 270, 270, 270, 270,
                     315, 315, 315, 315, 315, 315, 315, 315]}
    offsetAngles = {
        "James": [30, 120, 180,
                  30, 120, 180,
                  30, 120, 180,
                  30, 120, 180],
        "Subject1": [0, 45, 90, 130, 180, -45, -90, -130,
                     0, 45, 90, 130, 180, -45, -90, -130,
                     0, 45, 90, 130, 180, -45, -90, -130,
                     0, 45, 90, 130, 180, -45, -90, -130,
                     0, 45, 90, 130, 180, -45, -90, -130,
                     0, 45, 90, 130, 180, -45, -90, -130,
                     0, 45, 90, 130, 180, -45, -90, -130,
                     0, 45, 90, 130, 180, -45, -90, -130]}
    trialOrders = {
        "James": [10, 6, 0, 2, 3, 1, 4, 9, 7, 11, 5, 8],
        "Subject1": [46, 0, 18, 15, 1, 44, 54, 35, 59, 37, 41, 48, 58, 26, 10, 57, 14,
                     51, 29, 9, 19, 34, 31, 55, 50, 63, 28, 61, 21, 32, 3, 22, 42, 6,
                     47, 45, 17, 27, 52, 60, 7, 38, 30, 33, 4, 40, 25, 36, 12, 8, 39,
                     11, 53, 43, 13, 62, 16, 23, 20, 2, 5, 49, 24, 56]} 
    mne.set_log_level(False)
    # Initialize filepath and IDs
    filepath = os.getcwd() + r'\Data'
    IDs = load_IDs(filepath)
    # Load data
    both_EEG_lists = load_EEG(filepath, IDs)
    both_NIRS_lists = load_NIRS(filepath, IDs)
    both_BHV_lists, both_response_lists, both_RT_lists, both_endRT_lists = load_bhv(filepath, IDs, offsetAngles, trialOrders)
    # Preprocess data
    proc_tempo_EEG_lists, proc_spect_EEG_lists = preprocess_EEG(both_EEG_lists, IDs, offsetAngles, trialOrders)
    proc_NIRS_lists = preprocess_NIRS(both_NIRS_lists, IDs, offsetAngles, trialOrders)
    # Epoch data
    tempo_EEG_epochs_list, spect_EEG_epochs_list = epoch_EEG(proc_tempo_EEG_lists, proc_spect_EEG_lists)
    NIRS_epoch_list = epoch_NIRS(proc_NIRS_lists)
    # Plots
    plot_EEG_ERPs(filepath, IDs, tempo_EEG_epochs_list)
    plot_EEG_TFRs(filepath, IDs, spect_EEG_epochs_list)
    plot_NIRS(filepath, IDs, NIRS_epoch_list)
    plot_NIRS_wBHV(filepath, IDs, NIRS_epoch_list, both_BHV_lists, both_RT_lists, both_endRT_lists)
