# Graph Autoencoder for Model Agnostic Anomaly Detection
This repo builds the Graph Autoencoder from the Paper 'Graph theory inspired anomaly detection at the LHC', along with a couple of others simple architectures, for model-agnostic anomaly detection at particle collisions. It's currently set up to work out of the box using the LHC Olympics RnD ["dataset"](https://zenodo.org/records/6466204) if you download it locally. If you want to use subjets as the input, you'll have to cluster the jets using the process_subjet.py, using the library ["heppy"](https://github.com/cbernet/heppy) that requires a careful environment initialization, as seen by init_perlmutter_heppy.sh. 
The autoencoder can be trained either in an unsupervised manner or in a weakly supervised one using the sideband region (SB). For more info look at the config script and the steer_analysis.py parser arguments.

To run the model, change the parameters of the config.yaml file and run: 
```bash
python python -u analysis/steer_analysis.py -c config/config.yaml
``` 

The general pipeline is: steer analysis -> ml_analysis -> gae_train, ml_anonaly. Utils.py contain many useful functions, including for loading and preprocessing the data. 