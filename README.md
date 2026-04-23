# STESP
This project presents an open-source reproduction of the STESP algorithm, implementing the full core workflow detailed in the original paper. Key procedures include dynamic community network construction using the Spain-COVID dataset, STESP model training, epidemic infection scale prediction, and threshold analysis of the basic reproduction number.

The project mainly contains the following files:

stesp/config.py defines the main configuration parameters for STESP experiments and serves as the unified configuration entry of the project.

stesp/data.py loads the raw data, constructs dynamic community network sequences, and generates the input tensors required by the main model described in the paper.

stesp/model.py implements the core STESP model. It corresponds to the main methodological structure of the paper and is used to predict the infection probability of each community at the current time step.

stesp/mean_field.py maps the infection probabilities output by STESP to infection-scale prediction results.

train_stesp.py is used to train the main STESP model.

threshold_analysis.py is used to perform the threshold analysis experiments.
