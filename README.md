# Forecasting Air Pollutants Using Neural Networks  

## Objective and Data Analysis  
Air pollution remains a critical environmental and public health issue, particularly in metropolitan areas like **Delhi**. This project aims to develop a robust **neural network-based forecasting model** to predict air pollutant concentrations with high accuracy.  

The study focuses on air quality in **Shadipur, Delhi**, across different phases of the **COVID-19 lockdown**, analyzing pollutants such as:  

- **PM2.5 (Particulate Matter)**  
- **NO (Nitric Oxide)**  
- **NO₂ (Nitrogen Dioxide)**  
- **NOx (Nitrogen Oxides)**  
- **SO₂ (Sulfur Dioxide)**  
- **Ozone (O₃)**  
- **Benzene (C₆H₆)**  
- **Ethylbenzene (ETH)**  
- **m/p-Xylene**  
- **Toluene (C₆H₅CH₃)**  

### Preprocessing and Data Handling  
To ensure high-quality input for the model, multiple data processing techniques were implemented:  

- **Time-Series Decomposition**: Seasonal-trend decomposition using **Fourier Transforms (FFT)** and **Seasonal-Trend decomposition using LOESS (STL)** to extract periodic patterns.  
- **Missing Data Handling**: **Linear interpolation** and **KNN imputation** to maintain dataset consistency.  
- **Exploratory Data Analysis (EDA)**: Correlation heatmaps, rolling statistics, and trend analysis to study seasonal variations and relationships among pollutants.  

## Model Development and Evaluation  
Given the complex nature of air pollution trends, various **machine learning (ML) and artificial intelligence (AI) models** were explored. The models developed and tested include:  

- **Multi-Layer Perceptron (MLP)** – Standard feedforward neural network.  
- **Backpropagation Neural Network (BPNN)** – Optimized for error correction through gradient descent.  
- **Radial Basis Function Neural Network (RBFNN)** – Effective for pattern recognition in non-linear time-series data.  
- **Support Vector Regression (SVR)** – Non-neural model used as a benchmark for comparison.  
- **Wavelet Neural Network (WNN)** – A hybrid model combining wavelet transforms with neural networks for superior feature extraction.  

### Model Evaluation Metrics  
To assess the models' performance, standard time-series regression metrics were used:  

- **Root Mean Square Error (RMSE)** – Measures the model’s predictive accuracy.  
- **Mean Squared Error (MSE)** – Penalizes larger deviations in predictions.  
- **Mean Absolute Error (MAE)** – Evaluates absolute deviations from true values.  
- **R² Score (Coefficient of Determination)** – Represents the variance explained by the model.  
- **Taylor Diagrams** – Used for graphical performance comparison of models.  

### Key Findings  
The **Wavelet Neural Network (WNN)** emerged as the best-performing model, outperforming others in capturing pollutant variations due to its **multi-resolution analysis capability**. It effectively handled **short-term fluctuations and long-term trends**, making it the most reliable forecasting tool for this problem.  

## Validation and Uncertainty Analysis  
To ensure robustness, the following validation techniques were applied:  

- **Temporal Cross-Validation**: Splitting the dataset into multiple time-based folds to optimize generalization across unseen data.  
- **Sensitivity Analysis**: Evaluating the effect of different hyperparameters and data configurations on model performance.  
- **Monte Carlo Simulations**: Running multiple simulations to quantify prediction uncertainties and establish confidence intervals.  

## Impact of COVID-19 Lockdown on Air Quality  
The project also investigated the influence of lockdown policies on air pollution levels by dividing 2020 into five phases:  

1. **Pre-lockdown (January – March 2020)** – Baseline pollution levels before restrictions.  
2. **Strict Lockdown (March – May 2020)** – Significant reduction in emissions due to halted transportation and industrial activity.  
3. **Relaxed Lockdown (June – August 2020)** – Partial reopening, leading to a gradual rise in pollutants.  
4. **Liberal Lockdown (September – October 2020)** – Increased mobility and industrial activity.  
5. **After-Lockdown (November – December 2020)** – Pollution levels nearly returning to pre-pandemic conditions.  

### Observations  
- **Sharp Decline in NO₂ and NOx**: A drastic drop in vehicular emissions was recorded during the strict lockdown.  
- **Increase in Ozone (O₃)**: Due to reduced NOx emissions, ozone formation mechanisms shifted, leading to an unexpected rise in O₃ levels.  
- **Resurgence of PM2.5**: Post-lockdown industrial activities contributed to the rapid return of particulate pollution.  
- **Volatile Organic Compounds (Benzene, Toluene, Ethylbenzene, Xylene – BTEX Group)**:  
  - These pollutants showed mixed trends, with a **decrease in early lockdown phases** but a **rapid resurgence post-lockdown** due to the reopening of fuel stations and industrial activities.  
  - **Xylene and Toluene levels spiked** during relaxed lockdown phases, indicating their association with **vehicular emissions and industrial solvents**.  

Statistical analysis and visualizations (e.g., heatmaps, line graphs, and bar charts) were used to clearly depict these changes.  

## Insights and Applications  
### Key Takeaways  
- **WNN is highly effective** for time-series forecasting of air pollutants.  
- The lockdown provided a **real-world experiment** on pollution control, revealing actionable insights.  
- **Long-term emission reduction strategies** can be developed based on these findings.  

### Real-World Applications  
The results from this study have direct implications in:  
- **Environmental Policy-Making** – Assisting governments in designing effective emission control policies.  
- **Public Health Initiatives** – Understanding air quality trends to mitigate respiratory and cardiovascular diseases.  
- **Urban Planning** – Using pollution forecasts to guide city planning for sustainability.  

## Conclusion  
This project successfully leveraged **deep learning and wavelet-based neural networks** to forecast air pollutant concentrations in **Delhi** with high accuracy. The study not only demonstrated the **superiority of WNNs** in time-series forecasting but also provided critical insights into how **policy changes can significantly impact air quality**.  

The findings serve as a **foundation for future research in predictive environmental analytics**, contributing to global efforts in combating air pollution.  
