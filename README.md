# Graph Autoencoders for Model Agnostic Anomaly Detection


This repo construct various Graph Autoencoders (with and without attention) for model-agnostic anomaly detection at particle collisions. It's currently set up to work out of the box using the LHC Olympics RnD ["dataset"](https://zenodo.org/records/6466204). It's trained using the sideband  region (SB), effectively learning the dof of the background data (QCD), and then tested on separating signal vs background in the signal region (SR). 

To run the model, change the parameters of the config.yaml file accordingly and run: 
```bash
python analysis/steer_analysis.py --regenerate_graphs
``` 
where the --regenerate_graphs choice is optional once the graphs (PyG) have been constructed and saved. 

Check the steer_analysis script for more details.