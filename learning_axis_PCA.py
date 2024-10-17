#!/usr/bin/env python
# coding: utf-8

# In[27]:


import numpy as np
from sklearn.decomposition import PCA
import scipy.io
import scipy.io as sio
import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import t
from scipy.stats import uniform
from scipy.integrate import cumtrapz
from scipy.interpolate import interp1d
from sklearn.linear_model import LinearRegression
from scipy.signal import resample
from scipy.ndimage import gaussian_filter1d
from scipy.signal import convolve
from statsmodels.formula.api import ols


# In[1]:


from IPython.display import display, HTML
display(HTML("<style>.container { width:90% !important; }</style>"))


# In[5]:


# Function to load and process the reorganized MAT files
def load_reorganized_mat_file(file_path, file_name):
    mat_data = scipy.io.loadmat(file_path)
    neurons = mat_data['neurons'][0]  # Access the structured array

    processed_data = []
    
    for neuron in neurons:
#         free50fr0 = neuron['free50FR0']
#         tone50fr0 = neuron['tone50FR0']
        free100fr0 = neuron['free100FR0']
        tone100fr0 = neuron['tone100FR0']
        outcome = neuron['Outcome'].flatten()
        rt = neuron['RT'].flatten()
        freet = neuron['FreeT'].flatten()
        tonet = neuron['ToneT'].flatten()
        info = file_name[-17:-4]
        
        # Organize data into a dictionary or DataFrame
        neuron_data = {
#             'free50FR0': [fr0_entry[0] for fr0_entry in free50fr0],
#             'tone50FR0': [fr0_entry[0] for fr0_entry in tone50fr0],
            'free100FR0': [fr0_entry[0] for fr0_entry in free100fr0],
            'tone100FR0': [fr0_entry[0] for fr0_entry in tone100fr0],
            'Outcome': outcome,
            'RT': rt,
            'FreeT': freet,
            'ToneT': tonet,
            'info': info
        }
        
        processed_data.append(neuron_data)
    
    return processed_data

# Example usage
directory_path = r'G:\Google Takeout\t001\Takeout\Drive\Response Inhibition Project\2021-22_Attention\NP 2024-7\session firing rate trace struct full mixed window - Copy'
# root_path = r'G:\Google Takeout\t001\Takeout\Drive\Response Inhibition Project\2021-22_Attention\NP 2024-7\anticipation function'
all_sessions_data = []

for file_name in os.listdir(directory_path):
    file_path = os.path.join(directory_path, file_name)
    session_data = load_reorganized_mat_file(file_path, file_name)
    all_sessions_data.append(session_data)

print("Num of sessions:", len(all_sessions_data))
print("Num of neurons in the 1st session:", len(all_sessions_data[0]))
print('Num of trials in the 1st session:', len(all_sessions_data[0][0]['RT']))
print('Duration of withholding in the 1st trial: ' + str(all_sessions_data[0][0]['tone100FR0'][0].shape[1]/10) + ' s')
print('Max and min tone duration in the 1st session: ' + str(np.max(all_sessions_data[0][0]['ToneT'])) + ' and ' + str(np.min(all_sessions_data[0][0]['ToneT'])) + ' s')
print('Max and min free period in the 1st session: ' + str(np.max(all_sessions_data[0][0]['FreeT'])) + ' and ' + str(np.min(all_sessions_data[0][0]['FreeT'])) + ' s')


# In[39]:


# Function to extract last 50 and first 50 samples
def create_session_matrices(session_data):
    freeLast5 = []
    toneFirst5 = []
    
    for neuron_data in session_data:
        free_trials = neuron_data['free100FR0']
        tone_trials = neuron_data['tone100FR0']
        
        # Prepare matrices for storing the extracted trial data
        free_matrix = []
        tone_matrix = []
        
        # Loop through each trial and extract last 50 samples for free and first 50 for tone
        for free_trial, tone_trial in zip(free_trials, tone_trials):
            if free_trial.shape[1] > 50:
                free_matrix.append(free_trial[:, -50:].flatten())  # Last 50 samples of free period
            if tone_trial.shape[1] > 50:
                tone_matrix.append(tone_trial[:, :50].flatten())   # First 50 samples of tone period
        
        # Convert lists to numpy arrays and append to session list
        freeLast5.append(np.array(free_matrix))
        toneFirst5.append(np.array(tone_matrix))
    
    # Convert to 3D arrays (Neurons x Trials x Time Points)
    freeLast5 = np.stack(freeLast5)
    toneFirst5 = np.stack(toneFirst5)
    
    # Convert to 3D arrays (Neurons x Trials x Time Points) and remove redundant dimensions
    freeLast5 = np.squeeze(np.stack(freeLast5))  # Removes redundant dimensions if any
    toneFirst5 = np.squeeze(np.stack(toneFirst5))  # Removes redundant dimensions if any
    
    return freeLast5, toneFirst5

# Function to perform PCA and calculate variance explained
def percentage_pcs_70_variance(data_3d):
    n_neurons, n_trials, n_timepoints = data_3d.shape
    pcs_percentage_neuron_space = []
    pcs_percentage_trial_space = []
    
    # Loop through each time point for neuron space PCA
    for t in range(n_timepoints):
        print(f"Processing sample {t} / {n_timepoints}")
        # Neuron space: Neurons x Trials at each time point
        data_neuron_space = data_3d[:, :, t]  # Neurons x Trials
        pca_neuron = PCA()
        pca_neuron.fit(data_neuron_space.T)  # Transpose to treat neurons as variables
        explained_var_neuron = np.cumsum(pca_neuron.explained_variance_ratio_)
        
        
        # Find the percentage of PCs that explain 70% variance
        n_pcs_70_neuron = np.argmax(explained_var_neuron >= 0.7) + 1  # Add 1 since it's zero-indexed
        print(n_pcs_70_neuron)
        percentage_pcs_neuron = n_pcs_70_neuron / n_neurons * 100  # Percentage of total PCs
        print(percentage_pcs_neuron)
        pcs_percentage_neuron_space.append(percentage_pcs_neuron)

        # Trial space: Trials x Neurons at each time point
        data_trial_space = data_3d[:, :, t].T  # Trials x Neurons (transpose to match dimensions)
        pca_trial = PCA()
        pca_trial.fit(data_trial_space)
        explained_var_trial = np.cumsum(pca_trial.explained_variance_ratio_)
        
        # Find the percentage of PCs that explain 70% variance
        n_pcs_70_trial = np.argmax(explained_var_trial >= 0.7) + 1
        percentage_pcs_trial = n_pcs_70_trial / n_trials * 100  # Percentage of total PCs
        pcs_percentage_trial_space.append(percentage_pcs_trial)

    return pcs_percentage_neuron_space, pcs_percentage_trial_space


# In[19]:


all_sessions_data[0][0]['tone100FR0'][0][:, 50:].shape[1]


# In[42]:


# Main process
percentage_neuron_free = []
percentage_neuron_tone = []
percentage_trial_free = []
percentage_trial_tone = []

# load data
directory_path = r'G:\Google Takeout\t001\Takeout\Drive\Response Inhibition Project\2021-22_Attention\NP 2024-7\session firing rate trace struct full mixed window - Copy'
root_path1 = r'G:\Google Takeout\t001\Takeout\Drive\Response Inhibition Project\2021-22_Attention\NP 2024-7\PCA results\Npc'
root_path2 = r'G:\Google Takeout\t001\Takeout\Drive\Response Inhibition Project\2021-22_Attention\NP 2024-7\PCA results\Tpc'
all_sessions_data = []
session_indices = [x-1 for x in list(range(1, 92))]
# session_indices = [0]

for file_name in os.listdir(directory_path):
    file_path = os.path.join(directory_path, file_name)
    session_data = load_reorganized_mat_file(file_path, file_name)
    all_sessions_data.append(session_data)

for session_index in session_indices:
    session_data = all_sessions_data[session_index]
    print(f"Processing Session {session_index}")
#     print(session_data)
    freeLast5, toneFirst5 = create_session_matrices(session_data)
    print(toneFirst5.shape)
    
    # Perform PCA for tone and free periods in neuron space
    neuron_pcs_free, trial_pcs_free = percentage_pcs_70_variance(freeLast5)
    neuron_pcs_tone, trial_pcs_tone = percentage_pcs_70_variance(toneFirst5)
    
#     # Store the results for each session
#     percentage_neuron_free.append(neuron_pcs_free)
#     percentage_neuron_tone.append(neuron_pcs_tone)
#     percentage_trial_free.append(trial_pcs_free)
#     percentage_trial_tone.append(trial_pcs_tone)
    
    # Save results into .MAT files
    Npc_results_dict = {
        'percentage_neuron_free': neuron_pcs_free,
        'percentage_neuron_tone': neuron_pcs_tone
    }
    mat_filename = 'Session_trajNpc70Var_' + all_sessions_data[session_index][0]['info'] + '.mat'
    mat_filepath = Path(root_path1) / mat_filename
    sio.savemat(mat_filepath, Npc_results_dict)
    
    Tpc_results_dict = {
        'percentage_trial_free': trial_pcs_free,
        'percentage_trial_tone': trial_pcs_tone
    }
    mat_filename = 'Session_trajTpc70Var_' + all_sessions_data[session_index][0]['info'] + '.mat'
    mat_filepath = Path(root_path2) / mat_filename
    sio.savemat(mat_filepath, Tpc_results_dict)


# In[ ]:




