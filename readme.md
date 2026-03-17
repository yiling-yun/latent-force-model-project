# Latent Force Representation Model for Social Perception

This project presents a parametric model of social perception based on the force dynamics of attraction and repulsion between two agents. The model is designed to capture the kind of social interactions famously described by Heider and Simmel (1944), where simple geometric shapes moving together evoke rich social narratives (e.g., chasing, fighting, helping).

This repository includes both the modeling code and human behavioral data from labeling and similarity judgment experiments.

## Project Structure
project/
├── utils/                        # shared config (video order, dataset EDA)
├── human/
│   ├── behavioralExpCode/        # experiment code in HTML and JavaScript
│   │   ├── exp1/			# labeling experiment of 1156 animations 
│   │   ├── exp2/			# odd-one-out similarity judgment task
│   │   └── exp3/			# labeling experiment on force-generated animations
│   └── behavioralExpDataAndAnalysis/  # human data and analysis
│       ├── exp1/
│       ├── exp2/
│       └── exp3/
└── models/
    ├── force/                    # latent force model
    ├── lstm/                     # LSTM model
    └── readme.md

