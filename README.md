# TCAV ECG

## Overview
This repository contains code for performing TCAV (Testing with Concept Activation Vectors) analysis on ECG data. The project leverages the Captum library to interpret model predictions and understand the influence of different concepts on the model's decision-making process.

## Features
- Plotting TCAV scores for different experimental sets.
- Preparing and processing data from the Physionet Challenge Dataset.
- Performing statistical significance tests on TCAV results.
- Visualizing results with heatmaps and bar plots.

## Installation

### Prerequisites
- Python 3.10 or later
- A virtual environment is recommended

### Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/tcav-ecg.git
   cd tcav-ecg
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Data Preparation
- Ensure the Physionet Challenge Dataset is available in the specified format.
- Use the provided Jupyter notebooks to preprocess and prepare the data for analysis.

### Running TCAV Analysis
- USE TCAV.ipynb
- in notebook, USER must define following code for their research setting
- get_ecg_tensor, TCAV_dataset,model_inference 
- USER must define target_classifier and target_label dataframe

## Contributing
Contributions are welcome! Please read the [contributing guidelines](CONTRIBUTING.md) for more details.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact
For questions or feedback, please contact [yourname@domain.com](mailto:yourname@domain.com).
