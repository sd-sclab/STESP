This project provides a public reimplementation of STESP, covering the core workflow described in the paper, including dynamic community network construction based on the Spain-COVID dataset, training of the STESP model, infection-scale inference, and basic reproduction number threshold analysis.

The project mainly contains the following files:

1. stesp/config.py defines the main configuration parameters for STESP experiments and serves as the unified configuration entry of the project.

2. stesp/data.py loads the raw data, constructs dynamic community network sequences, and generates the input tensors required by the main model described in the paper.

3. stesp/model.py implements the core STESP model. It corresponds to the main methodological structure of the paper and is used to predict the infection probability of each community at the current time step.

4. stesp/mean_field.py maps the infection probabilities output by STESP to infection-scale prediction results.

5. train_stesp.py is used to train the main STESP model.

6. threshold_analysis.py is used to perform the threshold analysis experiments.