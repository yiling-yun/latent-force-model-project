# Force model estimation
Run `main_force_model_est.m`. This estimates the force parameters based on input trajectories.

# Predict human similarity judgments
- **Force model:** Run `get_force_dmat.m`
- **Kinematic Feature model:** Run `get_kinematic_feat_dmat.m`
These files compute pairwise Euclidean distances between videos based on descriptive histograms of model features.

# Generate trajectories
Run `main_gen_exp_traj.m` This generate new trajectories based on the estimated force parameters.
