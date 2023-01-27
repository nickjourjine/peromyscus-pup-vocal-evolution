#this file contains functions related to calculating, aggregating, and organizing features of 
#vocalizations and pups who made them

#file system
import os
import numpy as np
from numpy.lib.stride_tricks import as_strided
import pandas as pd
from tqdm import tqdm

#data
from scipy.io import wavfile
from scipy.signal import stft
from scipy.signal import hilbert
from datetime import date, datetime



def get_filename_keys(dataset):
    """
    Get the category names used to label each file in a dataset

    Arguments:
        dataset (str): the dataset the pup comes from. Must be one of 'development', 'bw_po_f1', 'bw_po_cf', 'bw_po_f2'

    Returns:
       filename_keys (list): a list of keys to name the information in each section of each recording's file name
    """
    
    assert dataset in ['development', 'bw_po_f1', 'bw_po_cf', 'bw_po_f2']

    #these are keys for each dataset that correspond to information in each file name
    development_keys = ['species', 'breeding_pair','litter_number', 'pup_number', 'mic_channel', 'weight_mg', 'sex', 'start_temp', 'end_temp', 
                        'removal_flag', 'age', 'date', 'time']
    bw_po_cf_keys = ['species', 'breeding_pair', 'pup_number', 'age', 'weight_mg', 'sex', 'litter_size', 'date', 'time']
    bw_po_f1_keys = development_keys.copy()
    bw_po_f2_keys = ['mic_chanel_prefix', 'cross_prefix', 'breeding_pair','family', 'litter_number', 'pup_number', 'mic_channel', 'weight_mg', 
                     'sex', 'start_temp', 'end_temp', 'removal_flag', 'age', 'date', 'time']

    if dataset == 'development':
        keys = development_keys    
    elif dataset == 'bw_po_cf':
        keys = bw_po_cf_keys
    elif dataset == 'bw_po_f1':
        keys = bw_po_f1_keys
    elif dataset == 'bw_po_f2':
        keys = bw_po_f2_keys

    return keys


def get_pup_metadata(source_path, dataset):
    """
    Get meta data for each pup in a dataset: species, dam/sire, litterID, littersize, weight, sex, age, temperatures.

    Arguments:
        source_path (str): the full path to the recording for which you want metdata
        dataset (str): the dataset the pup comes from. Must be one of 'development', 'bw_po_f1', 'bw_po_cf', 'bw_po_f2'

    Returns:
        meta_data (dict): a dictionary of the metadata, which can be further aggregated into a dataframe with multiple pups
    """
    #initialize dict and assert dataset is correct
    #"source_file" can refer to a raw recording from a pup (1 pup for each raw recording)
    #or it can refer to a single clip from a single vocalization from a pup
    #here it refers to the latter, but you should fix this potential ambiguity

    pup_dict = {}
    source_file = os.path.split(source_path)[-1]
    assert dataset in ['development', 'bw_po_f1', 'bw_po_cf', 'bw_po_f2']

    #get the keys to use for this dataset
    keys = get_filename_keys(dataset)

    #get the values from the file name and add to dict
    values = source_file.split('.')[0].split('_')
    assert len(keys) == len(values), "keys and values don't match"
    for key, value in zip(keys,values):
        pup_dict[key]=value
        pup_dict['pup'] = source_file

    return pup_dict


def aggregate_pup(source_path, features, features_df):
    """
    Aggregate warbleR acoustic features for a single pup

    Arguments:
        source_path (str): the full path to the raw recording for which you want metdata
        features (list): list of warbleR features you want to aggregate
        features_df (dataframe): dataframe with the features you want to use

    Returns:
        aggregate_data (dict): a dictionary of the metadata, which can be further aggregated into a dataframe with multiple pups

    """

    #get the pup name and assert dataset is correct
    pup = os.path.split(source_path)[-1]
    print(pup)
    assert len(features) == 26

    #get the path to the warbleR features and make sure it's what you expect
    if features_path.endswith('.csv'):
        feats = pd.read_csv(features_path)
    elif features_path.endswith('.feather'):
        feats = pd.read_feather(features_path)

    assert 'BW_hdbscan_label0_withspace.wav.wav' not in feats['source_file'] #TODO fix this for the development dataset
    assert np.any(feats.duplicated()) == 0

    #get the vocalizations from this pup
    assert pup.split('.')[0] in list(feats['pup']), "the pup printed above isn't in the 'pup' column in the features_path file"
    feats = feats.loc[feats['pup'] == pup.split('.')[0]]

    #get the summary stats: cry count, USV count, scratch count, and mean, median, min, max, and standard deviation of each vocalization of each type

    #count data
    feats_dict = {}
    feats_dict['pup'] = pup
    feats_dict['cry_count'] = len(feats.loc[feats['predicted_label'] == 'cry'])
    feats_dict['USV_count'] = len(feats.loc[feats['predicted_label'] == 'USV'])
    feats_dict['scratch_count'] = len(feats.loc[feats['predicted_label'] == 'scratch'])
    feats_dict['total_sounds_detected'] = feats_dict['cry_count']+feats_dict['USV_count']+feats_dict['scratch_count']
    feats_dict['total_vocalizations_detected'] = feats_dict['cry_count']+feats_dict['USV_count']

    #use groupby to get the aggregate feature data - note it is not strictly necessayr to group by pup since there is only one pup
    #keeping anyway because it works and I don't have time to break it
    feat_means = feats.groupby(by=['predicted_label']).mean()
    feat_vars = feats.groupby(by=['predicted_label']).aggregate(np.var)
    feat_mins = feats.groupby(by=['predicted_label']).aggregate(np.min)  
    feat_maxs = feats.groupby(by=['predicted_label']).aggregate(np.max)
    feat_meds = feats.groupby(by=['predicted_label']).aggregate(np.median) 

    for feature in features:
        
        variances = feat_vars[feature]
        means = feat_means[feature]
        mins = feat_mins[feature]
        maxs = feat_maxs[feature]
        meds = feat_meds[feature]

        try: cry_mean = means.loc['cry']
        except: cry_mean = float('NaN')

        try: cry_variance = variances.loc['cry']
        except:cry_variance = float('NaN')

        try: cry_min = mins.loc['cry']
        except:cry_min = float('NaN')

        try: cry_max = maxs.loc['cry']
        except:cry_max = float('NaN')

        try: cry_med = meds.loc['cry']
        except:cry_med = float('NaN')

        try: USV_mean = means.loc['USV']
        except: USV_mean = float('NaN')

        try: USV_variance = variances.loc['USV']
        except: USV_variance = float('NaN')

        try: USV_min = mins.loc['USV']
        except: USV_min = float('NaN')

        try: USV_max = maxs.loc['USV']
        except: USV_max = float('NaN')

        try: USV_med = meds.loc['USV']
        except: USV_med = float('NaN')

        try: scratch_mean = means.loc['scratch']
        except: scratch_mean = float('NaN')

        try: scratch_variance = variances.loc['scratch']
        except: scratch_variance = float('NaN')

        try: scratch_min = mins.loc['scratch']
        except: scratch_min = float('NaN')

        try: scratch_max = maxs.loc['scratch']
        except: scratch_max = float('NaN')

        try: scratch_med = meds.loc['scratch']
        except: scratch_med = float('NaN')

        feats_dict[f"cry_{feature}_mean"] = cry_mean
        feats_dict[f"USV_{feature}_mean"] = USV_mean
        feats_dict[f"scratch_{feature}_mean"] = scratch_mean

        feats_dict[f"cry_{feature}_variance"] = cry_variance
        feats_dict[f"USV_{feature}_variance"] = USV_variance
        feats_dict[f"scratch_{feature}_variance"] = scratch_variance

        feats_dict[f"cry_{feature}_min"] = cry_min
        feats_dict[f"USV_{feature}_min"] = USV_min
        feats_dict[f"scratch_{feature}_min"] = scratch_min

        feats_dict[f"cry_{feature}_max"] = cry_max
        feats_dict[f"USV_{feature}_max"] = USV_max
        feats_dict[f"scratch_{feature}_max"] = scratch_max

        feats_dict[f"cry_{feature}_med"] = cry_med
        feats_dict[f"USV_{feature}_med"] = USV_med
        feats_dict[f"scratch_{feature}_med"] = scratch_med

    return feats_dict



def aggregate_all_pups(source_list, dataset, features_df):
    """
    For each pup in source_list, aggregate warbleR acoustic features by pup, get pup metdata, then combine them into a single dataframe

    Arguments:
        source_list (list): list of full paths to source_files (one per pup) you want to process
        dataset (str): the dataset the pups come from. Must be one of 'development', 'bw_po_f1', 'bw_po_cf', 'bw_po_f2'
        features_df (list): dataframe of the features to aggregate

    Returns:
        all_pup_data (dataframe): a dictionary of the metadata, which can be further aggregated into a dataframe with multiple pups

    """

    #check inputs 
    assert dataset in ['development', 'bw_po_f1', 'bw_po_cf', 'bw_po_f2']
    assert isinstance(save, bool), "save must be True or False"
    assert os.path.exists(save_dir), "save_dir doesn't exist"

    #get the metadata for every pup
    all_pup_metadata = []
    for source_path in source_list:
        pup_metadata = get_pup_metadata(source_path=source_path, dataset=dataset)
        pup_metadata = pd.DataFrame.from_records([pup_metadata])
        all_pup_metadata.append(pup_metadata)
    print('done collecting metadata...')

    #get the aggregate features for every pup
    all_pup_features = []
    for source_path in source_list:
        pup_features = aggregate_pup(source_path=source_path, features=features, features_path=features_path)
        pup_features = pd.DataFrame.from_records([pup_features])
        all_pup_features.append(pup_features)
    print('done collecting pup features...')

    #add all the pups toegther
    all_pup_features_df = pd.concat(all_pup_features)
    all_pup_metadata_df = pd.concat(all_pup_metadata)
    
    return all_pup_features_df, all_pup_metadata_df
        


def librosa_utils_frame(x, *, frame_length, hop_length, axis=-1, writeable=False, subok=False):
    """
    This is a copy of librosa.features.rms() from v0.9.2
    
    Slice a data array into (overlapping) frames.

    This implementation uses low-level stride manipulation to avoid
    making a copy of the data.  The resulting frame representation
    is a new view of the same input data.

    For example, a one-dimensional input ``x = [0, 1, 2, 3, 4, 5, 6]``
    can be framed with frame length 3 and hop length 2 in two ways.
    The first (``axis=-1``), results in the array ``x_frames``::

        [[0, 2, 4],
         [1, 3, 5],
         [2, 4, 6]]

    where each column ``x_frames[:, i]`` contains a contiguous slice of
    the input ``x[i * hop_length : i * hop_length + frame_length]``.

    The second way (``axis=0``) results in the array ``x_frames``::

        [[0, 1, 2],
         [2, 3, 4],
         [4, 5, 6]]

    where each row ``x_frames[i]`` contains a contiguous slice of the input.

    This generalizes to higher dimensional inputs, as shown in the examples below.
    In general, the framing operation increments by 1 the number of dimensions,
    adding a new "frame axis" either before the framing axis (if ``axis < 0``)
    or after the framing axis (if ``axis >= 0``).

    Parameters
    ----------
    x : np.ndarray
        Array to frame
    frame_length : int > 0 [scalar]
        Length of the frame
    hop_length : int > 0 [scalar]
        Number of steps to advance between frames
    axis : int
        The axis along which to frame.
    writeable : bool
        If ``True``, then the framed view of ``x`` is read-only.
        If ``False``, then the framed view is read-write.  Note that writing to the framed view
        will also write to the input array ``x`` in this case.
    subok : bool
        If True, sub-classes will be passed-through, otherwise the returned array will be
        forced to be a base-class array (default).

    Returns
    -------
    x_frames : np.ndarray [shape=(..., frame_length, N_FRAMES, ...)]
        A framed view of ``x``, for example with ``axis=-1`` (framing on the last dimension)::

            x_frames[..., j] == x[..., j * hop_length : j * hop_length + frame_length]

        If ``axis=0`` (framing on the first dimension), then::

            x_frames[j] = x[j * hop_length : j * hop_length + frame_length]

    Raises
    ------
    ParameterError
        If ``x.shape[axis] < frame_length``, there is not enough data to fill one frame.

        If ``hop_length < 1``, frames cannot advance.

    See Also
    --------
    numpy.lib.stride_tricks.as_strided

    Examples
    --------
    Extract 2048-sample frames from monophonic signal with a hop of 64 samples per frame

    >>> y, sr = librosa.load(librosa.ex('trumpet'))
    >>> frames = librosa.util.frame(y, frame_length=2048, hop_length=64)
    >>> frames
    array([[-1.407e-03, -2.604e-02, ..., -1.795e-05, -8.108e-06],
           [-4.461e-04, -3.721e-02, ..., -1.573e-05, -1.652e-05],
           ...,
           [ 7.960e-02, -2.335e-01, ..., -6.815e-06,  1.266e-05],
           [ 9.568e-02, -1.252e-01, ...,  7.397e-06, -1.921e-05]],
          dtype=float32)
    >>> y.shape
    (117601,)

    >>> frames.shape
    (2048, 1806)

    Or frame along the first axis instead of the last:

    >>> frames = librosa.util.frame(y, frame_length=2048, hop_length=64, axis=0)
    >>> frames.shape
    (1806, 2048)

    Frame a stereo signal:

    >>> y, sr = librosa.load(librosa.ex('trumpet', hq=True), mono=False)
    >>> y.shape
    (2, 117601)
    >>> frames = librosa.util.frame(y, frame_length=2048, hop_length=64)
    (2, 2048, 1806)

    Carve an STFT into fixed-length patches of 32 frames with 50% overlap

    >>> y, sr = librosa.load(librosa.ex('trumpet'))
    >>> S = np.abs(librosa.stft(y))
    >>> S.shape
    (1025, 230)
    >>> S_patch = librosa.util.frame(S, frame_length=32, hop_length=16)
    >>> S_patch.shape
    (1025, 32, 13)
    >>> # The first patch contains the first 32 frames of S
    >>> np.allclose(S_patch[:, :, 0], S[:, :32])
    True
    >>> # The second patch contains frames 16 to 16+32=48, and so on
    >>> np.allclose(S_patch[:, :, 1], S[:, 16:48])
    True
    """

    # This implementation is derived from numpy.lib.stride_tricks.sliding_window_view (1.20.0)
    # https://numpy.org/doc/stable/reference/generated/numpy.lib.stride_tricks.sliding_window_view.html

    x = np.array(x, copy=False, subok=subok)

    if x.shape[axis] < frame_length:
        raise ParameterError(
            "Input is too short (n={:d})"
            " for frame_length={:d}".format(x.shape[axis], frame_length)
        )

    if hop_length < 1:
        raise ParameterError("Invalid hop_length: {:d}".format(hop_length))

    # put our new within-frame axis at the end for now
    out_strides = x.strides + tuple([x.strides[axis]])

    # Reduce the shape on the framing axis
    x_shape_trimmed = list(x.shape)
    x_shape_trimmed[axis] -= frame_length - 1

    out_shape = tuple(x_shape_trimmed) + tuple([frame_length])
    xw = as_strided(
        x, strides=out_strides, shape=out_shape, subok=subok, writeable=writeable
    )

    if axis < 0:
        target_axis = axis - 1
    else:
        target_axis = axis + 1

    xw = np.moveaxis(xw, -1, target_axis)

    # Downsample along the target axis
    slices = [slice(None)] * xw.ndim
    slices[axis] = slice(0, None, hop_length)
    return xw[tuple(slices)]
    
def get_rms(*, y=None, S=None, frame_length=2048, hop_length=512, center=True, pad_mode="constant",):
    """
    This is a copy of librosa.features.rms() from v0.9.2
    Copied here unaltered to solve a version incompatibility (error "libffi.so.7: cannot open shared object file: No such file or directory")
    that would have broken the virtual environment.
    
    From https://librosa.org/doc/0.9.2/_modules/librosa/feature/spectral.html#rms:
    
    Compute root-mean-square (RMS) value for each frame, either from the
    audio samples ``y`` or from a spectrogram ``S``.

    Computing the RMS value from audio samples is faster as it doesn't require
    a STFT calculation. However, using a spectrogram will give a more accurate
    representation of energy over time because its frames can be windowed,
    thus prefer using ``S`` if it's already available.

    Parameters
    ----------
    y : np.ndarray [shape=(..., n)] or None
        (optional) audio time series. Required if ``S`` is not input.
        Multi-channel is supported.

    S : np.ndarray [shape=(..., d, t)] or None
        (optional) spectrogram magnitude. Required if ``y`` is not input.

    frame_length : int > 0 [scalar]
        length of analysis frame (in samples) for energy calculation

    hop_length : int > 0 [scalar]
        hop length for STFT. See `librosa.stft` for details.

    center : bool
        If `True` and operating on time-domain input (``y``), pad the signal
        by ``frame_length//2`` on either side.

        If operating on spectrogram input, this has no effect.

    pad_mode : str
        Padding mode for centered analysis.  See `numpy.pad` for valid
        values.

    Returns
    -------
    rms : np.ndarray [shape=(..., 1, t)]
        RMS value for each frame

    Examples
    --------
    >>> y, sr = librosa.load(librosa.ex('trumpet'))
    >>> librosa.feature.rms(y=y)
    array([[1.248e-01, 1.259e-01, ..., 1.845e-05, 1.796e-05]],
          dtype=float32)

    Or from spectrogram input

    >>> S, phase = librosa.magphase(librosa.stft(y))
    >>> rms = librosa.feature.rms(S=S)

    >>> import matplotlib.pyplot as plt
    >>> fig, ax = plt.subplots(nrows=2, sharex=True)
    >>> times = librosa.times_like(rms)
    >>> ax[0].semilogy(times, rms[0], label='RMS Energy')
    >>> ax[0].set(xticks=[])
    >>> ax[0].legend()
    >>> ax[0].label_outer()
    >>> librosa.display.specshow(librosa.amplitude_to_db(S, ref=np.max),
    ...                          y_axis='log', x_axis='time', ax=ax[1])
    >>> ax[1].set(title='log Power spectrogram')

    Use a STFT window of constant ones and no frame centering to get consistent
    results with the RMS computed from the audio samples ``y``

    >>> S = librosa.magphase(librosa.stft(y, window=np.ones, center=False))[0]
    >>> librosa.feature.rms(S=S)
    >>> plt.show()

    """
    if y is not None:
        if center:
            padding = [(0, 0) for _ in range(y.ndim)]
            padding[-1] = (int(frame_length // 2), int(frame_length // 2))
            y = np.pad(y, padding, mode=pad_mode)

        x = librosa_utils_frame(y, frame_length=frame_length, hop_length=hop_length)

        # Calculate power
        power = np.mean(np.abs(x) ** 2, axis=-2, keepdims=True)
    elif S is not None:
        # Check the frame length
        if S.shape[-2] != frame_length // 2 + 1:
            raise ParameterError(
                "Since S.shape[-2] is {}, "
                "frame_length is expected to be {} or {}; "
                "found {}".format(
                    S.shape[-2], S.shape[-2] * 2 - 2, S.shape[-2] * 2 - 1, frame_length
                )
            )

        # power spectrogram
        x = np.abs(S) ** 2

        # Adjust the DC and sr/2 component
        x[..., 0, :] *= 0.5
        if frame_length % 2 == 0:
            x[..., -1, :] *= 0.5

        # Calculate power
        power = 2 * np.sum(x, axis=-2, keepdims=True) / frame_length ** 2
    else:
        raise ParameterError("Either `y` or `S` must be input.")

    return np.sqrt(power)
    
    
def get_snr(clip_path, noise_path, algorithm = 1):
    """
    Use noise clips generated by get_silence() to calculate signal to noise

    Arguments:
        clip_path (str): the wav clip you want to calculate snr on
        noise_path (str): the background clip for the recording that the vocalization came from (generated by annotation.get_noise_clip())
        algorithm (int): Must be one of 1, 2, or 3 (see comment below for description of each)
    Returns:
        snr (floar): the signal to noise ratio
    """
    #use noise clips generated by annotation.get_noise_clip() to calculate signal to noise ratio
    #based on snr calculations from warlbeR (https://github.com/maRce10/warbleR)
    # 1: ratio of S mean amplitude envelope to N mean amplitude envelope (mean(env(S))/mean(env(N)))
    # 2: ratio of S amplitude envelope RMS (root mean square) to N amplitude envelope RMS (rms(env(S))/rms(env(N)))
    # 3: ratio of the difference between S amplitude envelope RMS and N amplitude envelope RMS to N amplitude envelope RMS ((rms(env(S)) - rms(env(N)))/rms(env(N)))
    fs, noise = wavfile.read(noise_path)
    _, signal = wavfile.read(clip_path)

    if algorithm == 1:	
        signal_amplitude_envelope = np.abs(hilbert(signal))
        noise_amplitdue_envelope = np.abs(hilbert(noise))
        snr = np.mean(signal_amplitude_envelope)/np.mean(noise_amplitdue_envelope)

    elif algorithm == 2:

        signal_rms = get_rms(y=np.array(signal, 'float'))
        noise_rms = get_rms(y=np.array(noise, 'float'))
        snr = np.mean(signal_rms)/np.mean(noise_rms)

    elif algorithm == 3:
        signal_amplitude_envelope = np.abs(hilbert(signal))
        noise_amplitdue_envelope = np.abs(hilbert(noise))
        snr = (rms(y=signal_amplitude_envelope) - rms(y=noise_amplitdue_envelope))/rms(y=noise_amplitdue_envelope)

    return snr

#same as above but iterate through a directory - right now only deal with algorithms 1 and 2
def get_snr_batch(clips_dir, noise_dir, species, algorithm):

    """
    Run get_snr() on a batch of vocalization clips in a directory

    Arguments:
        clips_path (str): the path to the directory containing the wav clips for which you want to get signal to noise
        noise_path (str): the path to the directory containing the background clip for each wav in clip_dir (one noise clip per recording)
        algorithm (int): Must be one of 1, 2, or 3 (see comment below for description of each)
    Returns:
        snr_df (dataframe): a dataframe where each row is a vocalization and columns are path to vocalization file, snr, and algorithm
    """

    #get paths to vocalizations
    if species != None:
        vocs = [i for i in os.listdir(clips_dir) if i.startswith(species)]
    else:
        vocs = [i for i in os.listdir(clips_dir) if not i.startswith('.')]

    sig2noise_list = []
    source_files = []

    #iterate through vocalizations
    for voc in tqdm(vocs):
        
        #get the audio
        clip_path = os.path.join(clips_dir,voc)
        noise_path = os.path.join(noise_dir,voc.split('_clip')[0]+'_noiseclip.wav')

        #get signal to noise
        snr = get_snr(clip_path=clip_path, noise_path=noise_path, algorithm=algorithm)

        #update
        sig2noise_list.append(snr)
        source_files.append(voc)

    #write data to dataframe
    snr_df = pd.DataFrame()
    snr_df['source_file'] = source_files
    snr_df['snr'] = sig2noise_list
    snr_df['algorithm'] = algorithm

    return snr_df
	
	
#calculate how much of the audio clip is clipped
#test for clipping
#See https://dsp.stackexchange.com/questions/61996/what-are-the-semantics-of-wav-file-sample-values#:~:text=The%2016%20bit%20values%20in,but%20this%20is%20an%20interpretation.
#for where the threshold comes from (our audio is 16-bit and values range from -32768 to 32767)
#threshold is a percent of the maximum possible value the microphone can handle in decimals (ie, .95)
def get_clipping(clip_path, threshold):
    
    """
    Get the percent of wav file that is clipped

    Arguments:
        clip_path (str): the path to the wav file
        threshold (float): percent of maximum value (32767) possible to be read with 16-bit encoding above which you will call audio "clipped"
    Returns:
        percent_clipped (float): percent of frames in the wav file that exceed the clipping threshold
    """

    #check inputs
    assert os.path.exists(clip_path)
    assert 0 < threshold < 1, "Threshold must be a value between 0 and 1"
    
    #calculate percent clipped
    fs, audio = wavfile.read(clip_path)
    rect_wav = np.abs(audio)
    clipping_limit = threshold*32767
    clipped = rect_wav[rect_wav>clipping_limit]
    percent_clipped = len(clipped)/len(audio)

    return percent_clipped
    
#same as above but iterate through a directory of clips
#returns two lists: the clipping percents and the corres
def get_clipping_batch(wav_dir, threshold, species = None):
    
    """
    Run get_clipping() on a batch of vocalization clips in a directory

    Arguments:
        wav_dir (str): the path to the directory containing the wav clips for which you want to evaluate clipping
        threshold (float): percent of maximum value (32767) possible to be read with 16-bit encoding above which you will call audio "clipped"
        species (str): optional 2 letter code for species if you only want to process one species at a time
    Returns:
        clipping_df (dataframe): a dataframe where each row is wav file (eg vocalization) and columns are path to wav file, % clipped, and threshold
    """

    if species != None:
        to_process = [os.path.join(wav_dir,i) for i in os.listdir(wav_dir) if i.startswith(species) and i.endswith('.wav')]

    else:
        to_process = [os.path.join(wav_dir,i) for i in os.listdir(wav_dir) if not i.startswith('.') and i.endswith('.wav')]

    source_files = []
    clipping_percents = []
    for wav in tqdm(to_process):
        #get clipping percent
        percent_clipped = get_clipping(clip_path=wav, threshold=threshold)
        
        #update 
        source_files.append(wav.split('/')[-1])
        clipping_percents.append(percent_clipped)

    clipping_df = pd.DataFrame()
    clipping_df['source_file'] = source_files
    clipping_df['percent_clipped'] = clipping_percents
    clipping_df['clipping_threshold'] = threshold*32767
    
    return clipping_df

def write_warbleR_job_scripts(dataset, save_root, wav_root, script_dir, path_to_warbleR_extract):
    """
    Write sbatch job files to run warbleR_feature_extraction.R on a computing cluster 
    
    Required processing steps:
        1. You have a csv of all pups to process with a column called species, which will be used to group the features into directories
        2. You have a directory containing one wav clip for every vocalization in the above csv (no subdirectories)
    
    Arguments:
        dataset (str): one of ['bw_po_cf', 'bw_po_f1', 'bw_po_f2', development] (cross foster, F1, F2, development)
        save_root (str): the place where csv of acoustic features will be saved
        wav_root (str): the place containing the wav files (one per vocalization) to get features from
        script_dir (str): the place to save the sbatch scripts (one per species)
        path_to_warbleR_extract (str): path to the file warbleR_extract.R
    
    Returns
        None
    """
    
    #check inputs
    assert dataset in ['bw_po_cf', 'bw_po_f1', 'bw_po_f2', 'development']
    assert os.path.exists(save_root)
    assert os.path.exists(wav_root)

    #get the species that you have segments for - note that for the non_development data sets these are not strictly species but some other 
    #useful way of grouping the recordings (treatment/mic channel)
    if dataset == 'bw_po_cf':

        species_list = sorted(['BW', 'PO', 'CF-BW', 'CF-PO'])

    elif dataset == 'bw_po_f1':
       
        species_list = sorted(['BW-PO-cross-F1', 'cross-BW', 'cross-PO'])

    elif dataset == 'bw_po_f2':
        
        species_list = sorted(['ch1', 'ch2', 'ch3', 'ch4', 'ch5', 'ch6', 'ch7', 'ch8'])
        
    elif dataset == 'development':
        
        species_list = sorted(['BW', 'GO', 'LL', 'LO', 'MU', 'MZ', 'NB', 'PO', 'SW', 'BK'])

    #make a dictionary for paths
    paths_dict = {}
    for species in species_list:
        paths_dict[species] = {}

    #make the path to the directory where features will be saved
    today = str(date.today())
    today = ('').join(today.split('-'))
    now = str(datetime.now())
    time = now.split(' ')[-1]
    time = ('').join(time.split('.')[0].split(':'))
    save_path = os.path.join(save_root,('_').join([today,time]))
    os.mkdir(save_path)
    
    #populate the dictionary
    for species in species_list:

        #get the path to the raw clips for which features will be calculated
        wav_path = os.path.join(wav_root, species)

        #add the the paths to dictions
        paths_dict[species]['wav_path'] = wav_path
        paths_dict[species]['save_path'] = save_path
        #for each species, write an .sbatch file (no job array here) with
        lines = [
        '#!/bin/bash\n', 
        '#\n', 
        '#SBATCH -J warb # A single job name for the array\n', 
        '#SBATCH -p hoekstra,shared\n', 
        '#SBATCH -c 1 # one core\n', 
        '#SBATCH -t 0-8:00 # Running time of 8 hours\n', 
        '#SBATCH --mem 16000 # Memory request of 24 GB\n', 
        '#SBATCH -o '+species+'_warblExtract_%A_%a.out # Standard output\n', 
        '#SBATCH -e '+species+'_warblExtract_%A_%a.err # Standard error\n',  
        '\n', 
        '#load R\n',
        'module load intel/2017c impi/2017.4.239 FFTW/3.3.8-fasrc01\n',
        'module load R/4.0.2-fasrc01\n',
        'export R_LIBS_USER=$HOME/apps/R_4.0.2:$R_LIBS_USER\n',
        '\n',
        '#set directories\n',
        'SPECIES='+ species + '\n',
        'WAVSDIR='+ paths_dict[species]['wav_path'] +'\n',
        'SAVEDIR='+ paths_dict[species]['save_path'] +'\n',
        '\n',
        '#run warbleR extract\n',
        'Rscript /n/hoekstra_lab_tier1/Users/njourjine/manuscript/notebooks/00_manuscript/warbleR_extract.R'+" $SPECIES"+" $WAVSDIR"+" $SAVEDIR"
        ]

        #write lines
        sbatch_name = 'warbleR_extract_'+species+'.sbatch'
        sbatch_save_path = os.path.join(script_dir, sbatch_name)
        with open(sbatch_save_path, 'w') as f:
            f.writelines(lines)
    
    
    #write a parent .sh script to run all of the species' sbatch scripts
    parent_lines =  []
    for species in species_list:
        parent_lines.extend(['echo ', species, '\n',
                           'sbatch warbleR_extract_', species, '.sbatch\n',
                           'sleep 5\n'])
        
    sh_name = 'warbleR_extract_parent.sh'
    sh_save_path = os.path.join(script_dir, sh_name)
    with open(sh_save_path, 'w') as f:
            f.writelines(parent_lines)
            
    print('wrote job scripts to:\n\t', script_dir)

        




   