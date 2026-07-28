# Weather Condition Classification using Support Vector Machine (SVM) and Open-Meteo API

**Author:** Aditya Shukla

**Registration Number:** 23BAI10155

**Application Number:** IN26011099

**Batch Number:** 1(A)

**Email ID:** aditya.23bai10155@vitbhopal.ac.in

## Objective
The objective of this project is to build an SVM classification model that predicts whether the weather is Warm or Cool, using live meteorological data (temperature, humidity, surface pressure, wind speed) fetched from the Open-Meteo API.

## API Documentation Link
- [Open-Meteo Weather Forecast API](https://open-meteo.com/)

## Libraries Used
- `requests`
- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `scikit-learn`

## Methodology
1. **Data Collection and Understanding**: Fetched 7 days of hourly weather data (temperature, relative humidity, surface pressure, and wind speed) for Bengaluru using the Open-Meteo API and converted the JSON response into a Pandas DataFrame. from the Open-Meteo API and converted the JSON response into a Pandas DataFrame. Identified the four API variables as input features and created `Weather_Class` (Warm if temperature ≥ 25°C, else Cool) as the target.
2. **Data Preprocessing**:
   - Checked for missing values.
   - Dropped the `time` column (a timestamp label, not a predictive feature).
   - Encoded the target variable (`Warm` -> 1, `Cool` -> 0).
   - Split the dataset into 80% training and 20% testing using a stratified `train_test_split`.
   - Standardized all feature values using `StandardScaler`, since SVM is a distance/margin-based algorithm.
3. **Model Development**: Trained an `SVC` classifier with an RBF kernel on the scaled training features, then predicted the weather class on the test set.
4. **Model Evaluation**: Evaluated the model using Accuracy, Precision, Recall, and F1-Score, and visualized performance with a Confusion Matrix heatmap.

## Results

- **Accuracy:** 0.9117647058823529
- **Precision:** 0.930672268907563
- **Recall:** 0.9117647058823529
- **F1-Score:** 0.9139808481532148

## Conclusion
This project used live data from the Open-Meteo API to build an SVM (RBF kernel) classifier that labels each hour's weather as Warm or Cool based on temperature, humidity, surface pressure, and wind speed. The model reaches very high accuracy, but this is expected rather than impressive on its own: the target variable is defined directly as a threshold on temperature, and temperature is also included as an input feature, so the classifier mainly needs to learn that one boundary rather than a genuinely complex pattern. Feature scaling was important for training the SVM, since it separates classes using distances and margins in feature space; an unscaled feature like surface pressure (~1000) would otherwise dominate a feature like wind speed (~10) purely due to its larger numeric range, distorting the decision boundary. A key advantage of SVM is that the RBF kernel lets it capture non-linear decision boundaries without manually engineering new features. A key limitation is that SVM does not scale well to very large datasets and offers limited interpretability compared to models like Decision Trees.
