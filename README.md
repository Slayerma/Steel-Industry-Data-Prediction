# Steel Industry Data Prediction

## Overview

The Steel Industry Data Prediction repository contains tools and methodologies for predicting energy usage and efficiency in the steel industry using machine learning. The project encompasses data preparation, model development, and evaluation stages, providing insights that can help optimize energy consumption and reduce costs.

## Repository Structure

- **Jupyter Notebook**: `Non-Refactored-Analysis.ipynb`  
  Contains initial analysis, data preprocessing, exploratory data analysis (EDA), and model evaluation.

- **Python Scripts**: 
  - `energy_usage_prediction.py`  
    A refactored file containing the structured machine learning pipeline for predicting energy usage in the steel industry. This script implements best practices in Python, ensuring modularity and scalability.

## Features

- **Data Preprocessing**: Clean and prepare the dataset by handling missing values, encoding categorical variables, and normalizing numerical features.
- **Exploratory Data Analysis (EDA)**: Analyze data distribution and relationships, visualizing key insights.
- **Machine Learning Models**: Implemented using scikit-learn and XGBoost, focusing on performance and accuracy.
- **Performance Metrics**: Evaluation of models using accuracy, precision, recall, and F1-score.
- **Recommendations**: Output includes model performance metrics and suggestions for improvement.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/Steel-Industry-Data-Prediction.git
   ```

2. Navigate to the project directory:
   ```bash
   cd Steel-Industry-Data-Prediction
   ```

3. Install the required libraries:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. **Jupyter Notebook**: Open `Non-Refactored-Analysis.ipynb` for a comprehensive analysis workflow, including EDA and initial model training.
2. **Python Script**: Run `energy_usage_prediction.py` for a modular pipeline that can be easily integrated into larger projects.

## Contributing

Contributions are welcome! Please create a pull request or submit issues for any suggestions or improvements.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgements

- [scikit-learn](https://scikit-learn.org/stable/) for machine learning tools.
- [XGBoost](https://xgboost.readthedocs.io/en/latest/) for gradient boosting framework.
