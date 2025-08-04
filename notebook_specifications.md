
# Jupyter Notebook Specification: PD/LGD Model Backtesting Simulator

## 1. Notebook Overview

**Learning Goals:**

*   Understand the principles and methodology behind backtesting PD/LGD models.
*   Learn how to implement a rolling-origin forecast approach for model validation.
*   Compute and interpret key forecast error metrics and stability indices.
*   Evaluate the impact of macroeconomic factors on model performance.
*   Apply model governance and validation principles.

**Expected Outcomes:**

*   A functional Jupyter Notebook that allows users to backtest PD/LGD models using historical data.
*   Clear visualizations that present model performance and stability analysis.
*   Implement automated alert systems to notify users of potential model issues.
*   Generate a validation pack documenting the backtesting process and results.
*   A well-documented notebook that is reproducible and adheres to regulatory requirements.

## 2. Mathematical and Theoretical Foundations

This section provides the theoretical background for backtesting PD/LGD models.

*   **Probability of Default (PD):** The probability that a borrower will default on their debt obligations within a specified time horizon.
    *   Definition:  $PD = P(Default)$ where $P$ denotes probability.
    *   Real-world applications: Credit risk assessment, loan pricing, regulatory capital calculation.
*   **Loss Given Default (LGD):**  The proportion of exposure lost on a loan if a default occurs.
    *   Definition: $LGD = \frac{Exposure\,at\,Default - Recovery}{Exposure\,at\,Default}$
    *   Real-world applications: Credit risk assessment, loan pricing, regulatory capital calculation.
*   **Backtesting:** Evaluating the performance of a model by applying it to historical data and comparing the predicted outcomes with the actual outcomes.
*   **Rolling-Origin Forecast:** A backtesting approach where the model is repeatedly trained on a historical dataset and then used to predict outcomes for a future period. After the actual outcomes are observed, the training window is shifted forward in time, and the process is repeated.  This simulates how the model would perform in real-time.
*   **Forecast Error:** The difference between the predicted outcome and the actual outcome.

    *   Definition: $Error = Predicted\,Value - Actual\,Value$.
    *   Common Error Metrics:
        *   **Mean Error (ME):** The average forecast error. $$ME = \frac{1}{n}\sum_{i=1}^{n}(Predicted_i - Actual_i)$$
        *   **Mean Absolute Error (MAE):** The average of the absolute values of the forecast errors. $$MAE = \frac{1}{n}\sum_{i=1}^{n}|Predicted_i - Actual_i|$$
        *   **Root Mean Squared Error (RMSE):**  The square root of the average of the squared forecast errors. $$RMSE = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(Predicted_i - Actual_i)^2}$$
*   **Population Stability Index (PSI):** Measures the shift in the distribution of a model's scores between two samples (e.g., development and validation samples). It helps to identify if the model's scoring behavior has changed over time.

    *   Formula: $$PSI = \sum_{i=1}^{N} (Actual\%_i - Expected\%_i) * ln(\frac{Actual\%_i}{Expected\%_i})$$
    where:
    *   $N$ is the number of score ranges (bins).
    *   $Actual\%_i$ is the percentage of accounts in bin $i$ in the validation sample.
    *   $Expected\%_i$ is the percentage of accounts in bin $i$ in the development sample.
    *   Typical Interpretation:
        *   PSI < 0.1: Insignificant change in population.
        *   0.1 <= PSI < 0.2: Moderate change in population.
        *   PSI >= 0.2: Significant change in population.
*   **Parameter Drift:**  Changes in model parameters over time. This can indicate model instability or overfitting. Parameter drift can be monitored by tracking the values of the model coefficients over the rolling-origin backtesting intervals.
*   **Volatility Analysis:**  Comparing the volatility of the actual default rate changes with that of the model residuals.  Large unexplained volatility in the residuals may indicate the need for a volatility model (e.g., ARCH).
    *   Formula: Volatility is often measured as the standard deviation: $$\sigma = \sqrt{\frac{1}{N}\sum_{i=1}^{N}(x_i - \mu)^2}$$, where $x_i$ are the data points, $\mu$ is the mean, and $N$ is the number of data points.
*   **ARIMA, VAR, ARDL models:** These are time-series models used for forecasting.
    *   ARIMA stands for Autoregressive Integrated Moving Average. It is used for forecasting univariate time series data.
    *   VAR stands for Vector Autoregression. It is used for forecasting multiple time series data.
    *   ARDL stands for Autoregressive Distributed Lag. It is used for forecasting time series data with lagged values of the dependent and independent variables.

## 3. Code Requirements

*   **Expected Libraries:**
    *   `pandas`: Data manipulation and analysis. Used for handling time-series data, creating tables, and performing statistical calculations.
    *   `numpy`: Numerical computing. Used for mathematical operations, array manipulation, and random number generation.
    *   `matplotlib`: Data visualization. Used for generating plots and charts of model performance.
    *   `seaborn`: Statistical data visualization. Used for creating more advanced and aesthetically pleasing plots.
    *   `statsmodels`: Statistical modeling. Used for implementing ARIMA, VAR, and ARDL models.
    *   `scikit-learn`: Machine learning library. Used for model evaluation and other machine learning tasks.
    *   `scipy`: Scientific computing. Used for statistical analysis.
    *   `datetime`: Handling date and time data.
    *   `warnings`: Managing warnings to prevent output cluttering.
    *   `tabulate`: Outputting data in tabular format.
    *   Libraries to enable sending Email/Slack alerts.

*   **Input/Output Expectations:**

    *   **Input:**
        *   Credit rating default rate time-series data (e.g., CSV file). The data should contain columns for:
            *   `Date`: Date of the observation (quarterly).
            *   `Rating Grade`: Credit rating grade/segment.
            *   `Default Rate`: Rolling 12-month default rate for the grade.
        *   Loan-level LGD dataset (e.g., CSV file). The data should contain columns for:
            *   `Loan ID`: Unique identifier for the loan.
            *   `Original Loan Amount`: The initial loan amount.
            *   `Term`: Loan term in months.
            *   `Coupon Rate`: Interest rate of the loan.
            *   `LendingClub Grade`: LendingClub's risk grade for the loan.
            *   `Default Status Flag`: Binary indicator of default (1 if default, 0 otherwise).
            *   `Recovery Cash Flows`: Amounts and dates of recovery payments.
            *   `Collection Fees`: Any fees paid during the collection process.
        *   Macroeconomic data (e.g., CSV file). The data should contain columns for:
            *   `Date`: Date of the observation (quarterly).
            *   `GDP Growth`: Quarterly GDP growth rate (%).
            *   `Oil Prices`: Average quarterly oil price.
            *   `Inflation`: Quarterly inflation rate (%).
            *   `Unemployment Rate`: Unemployment rate (%).
            *   `Interest Rates`: Central bank policy rate or interbank rate.
            *   `Stock Index`: Stock index level or return.
        *   Configuration file containing thresholds and data paths (e.g., JSON or YAML).
    *   **Output:**
        *   Time-series plots of predicted vs. realized default rates with confidence bands.
        *   Bar chart of cumulative forecast error.
        *   Rolling-window line chart of model coefficients (e.g., unemployment lag-1 β).
        *   PSI or KS statistic trend plot.
        *   Box-and-whisker plots of LGD recovery profiles by vintage and secured vs. unsecured status.
        *   Heat-map of segment-level absolute errors (grades × quarters).
        *   Tornado chart showing ΔPD and ΔLGD under baseline vs. adverse macro scenarios.
        *   Swim-lane (Gantt) chart of the review cycle.
        *   Alerts (email/Slack) when forecast error, parameter drift, or PSI exceeds thresholds.
        *   Validation pack containing data description, diagnostics, back-test results, sensitivity tables, governance checklist, and sign-off section.
        *   Updated model inventory in JSON/YAML format with version, datasets, performance metrics, validation date, next-review date, and links to artifacts.

*   **Algorithms and Functions to be Implemented:**
    *   **Data Loading and Preprocessing:**
        *   Function to load credit rating default rate time-series data, LGD data, and macroeconomic data from CSV files.
        *   Function to aggregate the loan-level LGD data to calculate realized LGDs for each loan.
        *   Function to align and merge the credit risk data and macroeconomic data based on the date.
        *   Function to handle missing values in the datasets.
        *   Function to create rolling 12-month default rate series.
    *   **Model Implementation:**
        *   Function to implement ARIMA, VAR, or ARDL models for PD forecasting.
        *   Function to estimate LGD point estimates.
    *   **Backtesting:**
        *   Function to implement a rolling-origin forecast approach for backtesting PD/LGD models.
        *   Function to train the model on historical data and generate predictions for a future period.
        *   Function to shift the training window forward in time and repeat the process.
    *   **Error Measurement:**
        *   Function to calculate forecast errors (e.g., difference between predicted and actual default rates).
        *   Function to calculate Mean Error (ME), Mean Absolute Error (MAE), and Root Mean Squared Error (RMSE).
    *   **Stability Analysis:**
        *   Function to calculate the Population Stability Index (PSI) to assess model stability.
        *   Function to track parameter drift over time.
    *   **Volatility Analysis:**
        *   Function to calculate the volatility (standard deviation) of actual default rate changes and model residuals.
    *   **Visualization:**
        *   Function to generate time-series plots of predicted vs. realized default rates with confidence bands.
        *   Function to generate bar charts of cumulative forecast error.
        *   Function to generate rolling-window line charts of model coefficients.
        *   Function to generate PSI trend plots.
        *   Function to generate box-and-whisker plots of LGD recovery profiles.
        *   Function to generate heat-maps of segment-level absolute errors.
        *   Function to generate tornado charts showing ΔPD and ΔLGD under different scenarios.
        *   Function to generate swim-lane charts of the review cycle.
    *   **Alerts:**
        *   Function to implement alert logic based on predefined thresholds for forecast error, parameter drift, and PSI.
        *   Function to send email/Slack alerts when thresholds are breached.
    *   **Validation Pack Generation:**
        *   Function to generate a PDF/HTML report containing the backtesting results, visualizations, and governance checklist.
    *   **Model Inventory Update:**
        *   Function to update the model inventory (JSON/YAML) with the latest model information, performance metrics, validation date, and next-review date.

*   **Visualizations:**
    *   Time-series overlay of predicted vs. realised default rates with confidence bands.
    *   Bar chart of cumulative forecast error.
    *   Rolling-window line chart of coefficients (e.g., unemployment lag-1 β).
    *   PSI or KS statistic trend plot.
    *   Box-and-whisker by vintage, segmented by secured vs. unsecured.
    *   Heat-map of segment-level absolute errors (grades × quarters).
    *   Tornado chart showing ΔPD and ΔLGD under baseline vs. adverse macro scenarios.
    *   Swim-lane (Gantt) of review cycle: development → validation → approval → monitoring flags.
    *   Tables presenting key performance metrics like ME, MAE, RMSE, PSI.

## 4. Additional Notes or Instructions

*   **Assumptions:**
    *   The historical data is representative of future conditions.
    *   The macroeconomic data is accurate and reliable.
    *   The model parameters are stable over time.
    *   The chosen models (ARIMA, VAR, ARDL) are appropriate for the data.
*   **Constraints:**
    *   The notebook should be able to handle large datasets efficiently.
    *   The calculations should be performed accurately and consistently.
    *   The visualizations should be clear and easy to understand.
    *   The alert system should be reliable and timely.
*   **Customization Instructions:**
    *   Users should be able to customize the data paths, model parameters, thresholds, and alert settings through the configuration file.
    *   Users should be able to choose which visualizations to generate.
    *   Users should be able to specify the start and end dates for the backtesting period.
*   **Regulatory Alignment:**
    *   Cross-reference each monitoring and validation step to SR 11-7 "three areas of model risk management (development/use, validation, governance)" so that auditors can map evidence to regulation.
*   **Reproducibility & Control:**
    *   Use configuration files for thresholds and data paths.
    *   Store all raw and derived data under version control.
    *   Tag Docker image of the notebook environment; include requirements.txt or environment.yml.
