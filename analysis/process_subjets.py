'''
Script to load nsubs for pp vs AA. It requires installing heppy. For perlmutter: https://github.com/alicernc/info/blob/main/heppy_at_perlmutter.md
'''


import os
import time
import numpy as np
import math 
import sys
import glob
from sklearn.preprocessing import StandardScaler
from collections import defaultdict

import utils
# Fastjet via python (from external library heppy)
import fastjet as fj
import fjcontrib
import fjext

import logging
from pathlib import Path
import h5py
import awkward as ak
import hist
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import uproot
import pandas as pd
from tqdm import tqdm
import energyflow as ef

class process_subjets():

    def __init__(self, base_dir, use_SR=False, n_events=10000, N_cluster_list=[10], only_obs=False, classification_task = 'rnd', compute_subjets_and_global_obs=False, unsupervised=False): # Added new flag
        # Determine the filename based on whether use_SR is True or False.
        filename = 'SR.h5' if use_SR else 'SB.h5'
        # Create the full file path by joining the base directory and filename.
        file_path = os.path.join(os.path.abspath(base_dir), filename)
        self.file_path = file_path  # Save the full file path for later use.
        if unsupervised and not use_SR:
            sys.exit("Unsupervised mode is only supported for use_SR=True. Exiting.")
        self.unsupervised = unsupervised    
        self.use_SR = use_SR
        self.n_events = n_events
        self.N_cluster_list = N_cluster_list
        self.only_obs = only_obs

        # classification_task
        self.classification_task = classification_task

        # Store the new flag
        self.compute_subjets_and_global_obs = compute_subjets_and_global_obs

        # Ensure flags are not contradictory
        if self.only_obs and self.compute_subjets_and_global_obs:
            print("Warning: Both only_obs and compute_subjets_and_global_obs are True. Prioritizing compute_subjets_and_global_obs=True.")
            self.only_obs = False # Prioritize calculating both

        self.output = defaultdict(list)

        self.init_data(use_SR=use_SR, n_events=n_events,unsupervised = self.unsupervised) # Pass jets_raw to init_data if needed later

        # Ensure the parent directory exists.
        parent_dir = os.path.dirname(self.file_path)
        if not os.path.exists(parent_dir): os.makedirs(parent_dir)

        # Write jet arrays to file
        with h5py.File(self.file_path, 'w') as hf:
            print('-------------------------------------')
            hf.create_dataset(f'labels', data=self.Y)
            print(f'labels: {self.Y.shape}')

            # Write numpy arrays
            for key,val in self.output_np.items():
                hf.create_dataset(key, data=val)
                print(f'{key}: {val.shape}')
            print('-------------------------------------\n')

    #---------------------------------------------------------------
    def init_data(self, use_SR, n_events, unsupervised = False):
        if unsupervised:
            nbkg = n_events
            nsig = 30000
            jets_raw, X, mjj, self.Y = utils.class_loader(nbkg=nbkg, nsig=nsig, unsupervised=True)

        else: 
            if not use_SR:
                X, jets_raw, mjj = utils.DataLoader(n_events=n_events, unsupervised=False) # Renamed jets to jets_raw
                self.Y = np.zeros(len(X), dtype=int)
            else:
                n_dataset = 50000
                # Ensure class_loader also returns X and jets_raw correctly
                jets_raw, X, mjj, self.Y = utils.class_loader(use_SR=True, nbkg=n_dataset, nsig=n_dataset, unsupervised=False)

        n_loaded_events = X.shape[0] # Number of events loaded
        print(f'Raw jets shape: {jets_raw.shape}') # Should be (n_events, 2, 5)
        print(f'Particle array X shape: {X.shape}') # Shape depends on n_particles

        # --- Fastjet Particle Preparation using X ---
        # (Assuming X is [n_events, 2, n_particles, features] and contains pt, y, phi for fastjet)
        columns = ['pt', 'y', 'phi']
        if X is None or X.shape[1] != 2 or X.ndim != 4 or X.shape[3] < 3:
            raise ValueError(f"Particle array X has unexpected shape {X.shape if X is not None else None}. Expected (n_events, 2, n_particles, >=3)")
        particles_for_fj = X[:, :, :, :3] # Select pt, y, phi from X

        n_particles_per_jet = particles_for_fj.shape[2]
        df_particles = pd.DataFrame(particles_for_fj.reshape(-1, 3), columns=columns)
        # Create jet_id index: repeats 1, 1, ..., N*2, N*2, ...
        df_particles.index = np.repeat(np.arange(n_loaded_events * 2), n_particles_per_jet) + 1
        df_particles.index.name = 'jet_id'

        df_fjparticles_grouped = df_particles.groupby('jet_id')

        print('Converting particle dataframe (X) to fastjet::PseudoJets...')
        try:
            if df_fjparticles_grouped.ngroups == 0:
                raise ValueError("No groups found in particle DataFrame.")
            self.df_fjparticles = df_fjparticles_grouped.apply(self.get_fjparticles).dropna()
            if len(self.df_fjparticles) != n_loaded_events * 2:
                print(f"Warning: Number of valid PseudoJet vectors ({len(self.df_fjparticles)}) doesn't match expected ({n_loaded_events * 2}).")
        except Exception as e:
            print(f"Error during fastjet particle conversion: {e}")
            raise
        print('Done.')
        print()

        # --- Run Fastjet Analysis on X ---
        fj.ClusterSequence.print_banner()
        print('Finding jets and computing N-subjettiness and n_particles from X...')
        # This loop calls analyze_event -> compute_jet_observables
        # which populates self.output['observables_fj']
        result = [self.analyze_event(fj_particles)
                for index, fj_particles in tqdm(enumerate(self.df_fjparticles), total=len(self.df_fjparticles))]


        # --- Process and Combine Output ---
        self.output_np = {} # Initialize final output dict
        temp_output_np = {key: np.array(value) for key, value in self.output.items() if value} # Convert intermediate lists

        #print(f'Intermediate output_np keys: {temp_output_np.keys()}')
        for key, arr in temp_output_np.items():
            print(f'Intermediate output_np[{key}].shape: {arr.shape}')

        # Reshape fastjet observables and combine with raw pt/mass
        fj_key = 'observables_fj'
        final_obs_key = 'observables' # Final key to be saved

        if fj_key in temp_output_np:
            arr_fj = temp_output_np[fj_key]
            n_jets_processed = arr_fj.shape[0]
            n_features_fj = arr_fj.shape[1] # Should be 3 (tau21, tau32, n_particles)

            if n_jets_processed != n_loaded_events * 2:
                 print(f"Warning: Jet count mismatch for '{fj_key}'. Processed ({n_jets_processed}) vs Expected ({n_loaded_events * 2}). Final array might be incorrect.")
                 # Attempt reshape anyway based on n_loaded_events, but be cautious
                 # A more robust method would track successful analyze_event calls

            # Reshape fastjet results to (n_events, 2, 3)
            try:
                fj_observables_reshaped = arr_fj.reshape(n_loaded_events, 2, n_features_fj)
                #print(f"Reshaped key '{fj_key}' to: {fj_observables_reshaped.shape}")
            except ValueError as e:
                print(f"Error reshaping '{fj_key}': {e}. Skipping combination.")
                # Store original unreshaped array if reshape fails? Or skip key?
                # self.output_np[fj_key] = arr_fj # Option: store unreshaped
                return # Or exit, depending on desired behavior

            # --- Get original pt and mass from jets_raw ---
            # Ensure jets_raw has the correct shape
            if jets_raw.shape[0] != n_loaded_events or jets_raw.shape[1] != 2:
                 raise ValueError(f"Original jets_raw array shape mismatch. Expected ({n_loaded_events}, 2, 5), got {jets_raw.shape}")

            # Extract original pt (index 0) -> shape: (n_events, 2)
            jet_pts_orig = jets_raw[:, :, 0]
            # Extract original mass (index 3) -> shape: (n_events, 2)
            jet_masses_orig = jets_raw[:, :, 3]

            # Reshape pt and mass to (n_events, 2, 1) for concatenation
            jet_pts_reshaped = jet_pts_orig.reshape(n_loaded_events, 2, 1)
            jet_masses_reshaped = jet_masses_orig.reshape(n_loaded_events, 2, 1)

            # Concatenate along the last axis (features)
            # Order: [original_mass, original_pt, tau21, tau32, n_particles]
            final_observables = np.concatenate(
                (jet_masses_reshaped, jet_pts_reshaped, fj_observables_reshaped), axis=2
            )
            
            # Apply StandardScaler on final_observables with region-dependent scaling
            scaler_file = os.path.join(os.path.dirname(self.file_path), 'scaler.pkl')
            n_events_final = final_observables.shape[0]
            
            # Sort the two jets per event based on jet mass (first column) in descending order
            jet_sort_indices = np.argsort(-final_observables[:, :, 0], axis=1)
            final_observables = np.take_along_axis(final_observables, jet_sort_indices[:, :, np.newaxis], axis=1)

            final_observables_2d = final_observables.reshape(n_events_final * 2, -1)
            if not use_SR:
                scaler = StandardScaler()
                final_observables_scaled = scaler.fit_transform(final_observables_2d)
                import pickle
                os.makedirs(os.path.dirname(scaler_file), exist_ok=True)
                with open(scaler_file, 'wb') as f:
                    pickle.dump(scaler, f)
            else:
                import pickle
                with open(scaler_file, 'rb') as f:
                    scaler = pickle.load(f)
                final_observables_scaled = scaler.transform(final_observables_2d)
            final_observables = final_observables_scaled.reshape(n_events_final, 2, -1)
            print(f"Final observables shape after scaling: {final_observables.shape}") # Should be (n_events, 2, 5)
            
            # Sort the two jets per event based on jet mass (first column) in descending order
            #jet_sort_indices = np.argsort(-final_observables[:, :, 0], axis=1)
            #final_observables = np.take_along_axis(final_observables, jet_sort_indices[:, :, np.newaxis], axis=1)
            
            # Store the final combined (scaled and sorted) array under the 'observables' key
            self.output_np[final_obs_key] = final_observables
            #print(f"Stored final combined observables under key '{final_obs_key}' with shape: {self.output_np[final_obs_key].shape}") # Should be (n_events, 2, 5)

        else:
             print(f"Warning: Key '{fj_key}' not found in intermediate output. Cannot add pt/mass.")

        # Update subjet arrays: apply the same jet sorting as in final_observables
        for key, arr in temp_output_np.items():
             if key.isdigit():  # Assumes subjet keys are digits like '5'
                  n_jets_processed = arr.shape[0]
                  if n_jets_processed == n_loaded_events * 2:
                       N_cluster = int(key)
                       n_subjet_features = arr.shape[2]
                       try:
                            subjets_array = arr.reshape(n_loaded_events, 2, N_cluster, n_subjet_features)
                            # Apply the same jet_sort_indices from final_observables
                            if 'jet_sort_indices' in locals():
                                subjets_array = np.take_along_axis(subjets_array, jet_sort_indices[:, :, np.newaxis, np.newaxis], axis=1)
                            self.output_np[key] = subjets_array
                            print(f"Reshaped and sorted subjet key '{key}' to: {self.output_np[key].shape}")
                       except ValueError as e:
                            print(f"Error reshaping subjet key '{key}': {e}. Storing unreshaped.")
                            self.output_np[key] = arr  # Store unreshaped if error
                  else:
                       print(f"Warning: Jet count mismatch for subjet key '{key}'. Storing unreshaped.")
                       self.output_np[key] = arr  # Store unreshaped

        return # End of init_data

        

    #---------------------------------------------------------------
    # Transform particles to fastjet::PseudoJets
    #---------------------------------------------------------------
    def get_fjparticles(self, df_particles_grouped):

        user_index_offset = 0
        return fjext.vectorize_pt_eta_phi(df_particles_grouped['pt'].values,
                                          df_particles_grouped['y'].values,
                                          df_particles_grouped['phi'].values,
                                          )

    #---------------------------------------------------------------
    # Process an event
    #---------------------------------------------------------------
    def analyze_event(self, fj_particles, index=-1):
    
        # Check that the entries exist appropriately
        if fj_particles and type(fj_particles) != fj.vectorPJ:
            print('fj_particles type mismatch -- skipping event')
            return

        # Find jets -- one jet per "event". We only use antikt for the Jet Clustering
        jet_def = fj.JetDefinition(fj.antikt_algorithm, fj.JetDefinition.max_allowable_R)

        cs = fj.ClusterSequence(fj_particles, jet_def)
        jet_selected = fj.sorted_by_pt(cs.inclusive_jets())[0]
  
        # Compute jet quantities and store in our data structures
        if self.compute_subjets_and_global_obs:
            # Calculate and store BOTH subjets and global observables
            self.fill_subjets(jet_selected, index = index)
            self.compute_jet_observables(jet_selected) # Stores under 'global_observables'
        elif self.only_obs:
            # Calculate and store ONLY global observables
            self.compute_jet_observables(jet_selected) # Stores under 'global_observables'
        else:
            # Calculate and store ONLY subjets (original only_obs=False behavior)
            self.fill_subjets(jet_selected, index = index)


    #---------------------------------------------------------------
    # Compute subjet kinematics...
    #---------------------------------------------------------------
    def fill_subjets(self, jet, index=-1):
        subjet_def = fj.JetDefinition(fj.kt_algorithm, fj.JetDefinition.max_allowable_R)
        cs_subjet = fj.ClusterSequence(jet.constituents(), subjet_def)
            
        for N_cluster in self.N_cluster_list:

            subjets = fj.sorted_by_pt(cs_subjet.exclusive_jets_up_to(N_cluster))
            # Create a list to hold the subjet kinematic information
            subjet_features = []
            for subj in subjets:
                # Access the subjet observables; assuming they are accessible as .pt, .rap, .phi.
                pt  = subj.pt()
                rap = subj.rap()
                phi = subj.phi()
                if phi >= math.pi: phi -= 2*math.pi
                subjet_features.append([pt, rap, phi, 1])
            # Pad the list if there are fewer than N_cluster subjets.
            if len(subjet_features) < N_cluster:
                #print(f"Warning: For index: {index}, N_cluster {N_cluster}, only {len(subjet_features)} subjets found. Padding the remainder.")
                # Determine how many rows to pad.
                pad_rows = N_cluster - len(subjet_features)
                # Create a pad array, for example filled with zeros.
                pad_array = [[0, 0, 0, 0]] * pad_rows
                subjet_features.extend(pad_array)
   
            # Store the result in the output dictionary.
            # Since self.output is a defaultdict(list), we append the array for this N_cluster.
            self.output[f'{N_cluster}'].append(subjet_features)
    
    #---------------------------------------------------------------
    # Compute subjet kinematics...
    #---------------------------------------------------------------
    def compute_jet_observables(self, jet):
        # Get the jet mass
        #mass = jet.m()
        # Get jet pt
        #pt = jet.pt()
        
        # Number of particles (constituents) in the jet
        n_particles = len(jet.constituents())
        # Setup N-subjettiness calculators
        tau1_calc = fjcontrib.Nsubjettiness(1, fjcontrib.OnePass_KT_Axes(), fjcontrib.UnnormalizedMeasure(1.0))
        tau2_calc = fjcontrib.Nsubjettiness(2, fjcontrib.OnePass_KT_Axes(), fjcontrib.UnnormalizedMeasure(1.0))
        tau3_calc = fjcontrib.Nsubjettiness(3, fjcontrib.OnePass_KT_Axes(), fjcontrib.UnnormalizedMeasure(1.0))

        # Compute τ values for the jet.
        tau1 = tau1_calc.result(jet)
        tau2 = tau2_calc.result(jet)
        tau3 = tau3_calc.result(jet)

        # Safeguard against division by zero.
        tau21 = (tau2 / tau1) if tau1 > 0 else 0 # Calculated via fastjet
        tau32 = (tau3 / tau2) if tau2 > 0 else 0 # Calculated via fastjet

        # Use a temporary key for these fastjet-derived observables
        key = 'observables_fj'
        if key not in self.output:
            self.output[key] = []

        # Store only the fastjet-derived observables
        # Order: [tau21, tau32, n_particles]
        obs = [tau21, tau32, n_particles]
        self.output[key].append(obs)
        



if __name__ == '__main__':
    t_st = time.time()
    
    # classification_task
    classification_task = 'rnd' # rnd (default) or qvsg

    n_events= 126000
    N_cluster_list=[30, 35, 40, 50, 75] # Example N_cluster for subjets

    # --- Options ---
    unsupervised = True
    run_only_obs = False          # Calculate only global observables
    run_only_subjets = True       # Calculate only subjets
    run_subjets_and_global = False # Calculate both subjets and global observables

    # Determine flags and directory based on options
    if unsupervised: 
        run_only_subjets = True
        only_obs_flag = False
        compute_subjets_and_global_obs_flag = False
        # Use a directory name indicating unsupervised content
        dir_suffix = f'unsupervised_{n_events}'
    else:
        if run_subjets_and_global:
            only_obs_flag = False
            compute_subjets_and_global_obs_flag = True
            # Use a directory name indicating combined content
            dir_suffix = f'subjets_and_global_{n_events}'
        elif run_only_obs:
            only_obs_flag = True
            compute_subjets_and_global_obs_flag = False
            dir_suffix = f'jet_obs_{n_events}'
        elif run_only_subjets:
            only_obs_flag = False
            compute_subjets_and_global_obs_flag = False
            # Use a directory name indicating only subjets
            dir_suffix = f'subjets_only_{n_events}'
        else:
            print("No run option selected. Exiting.")
            sys.exit()

    # Define base directory
    base_output_dir = f'/pscratch/sd/d/dimathan/LHCO/Data/subjets/{classification_task}_{dir_suffix}'

    print(f"\nRunning with flags: only_obs={only_obs_flag}, compute_subjets_and_global_obs={compute_subjets_and_global_obs_flag}")
    print(f"Output directory: {base_output_dir}\n")
    
    if classification_task == 'rnd':
        # Run for Sideband (Background)
        print("--- Processing Sideband (SB) ---")
        if not unsupervised:
            process_subjets(base_output_dir,
                            use_SR=False,
                            n_events=n_events,
                            N_cluster_list=N_cluster_list,
                            only_obs=only_obs_flag,
                            compute_subjets_and_global_obs=compute_subjets_and_global_obs_flag,)

        # Run for Signal Region (Signal + Background)
        print("\n--- Processing Signal Region (SR) ---")
        print(f'unsupervised: {unsupervised}')
        process_subjets(base_output_dir,
                        use_SR=True,
                        n_events=n_events, # Adjust SR event count if needed
                        N_cluster_list=N_cluster_list,
                        only_obs=only_obs_flag,
                        compute_subjets_and_global_obs=compute_subjets_and_global_obs_flag, 
                        unsupervised=unsupervised, )
        
    print(f'\nTotal time: {time.time()-t_st:.2f} seconds')

    #125k is 5,10,15,20, 25
    #126k is 30,35,40,50,75