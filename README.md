# Predictive Analysis for Manufacturing Operations

## Objective
This project demonstrates the development of a predictive analysis model for manufacturing operations. The goal is to:

1. Build a predictive model that forecasts machine downtime based on historical data.
2. Implement RESTful API endpoints to upload data, train the model, and make predictions.

---

## Project Structure

```
manufacturingOperationsPredictions
|--- uploads
|    |--- (Uploaded files and trained model will be stored here)
|
|--- newMain.py         # Main Python script containing the application code
|--- README.md          # Documentation for the project
|--- requirements.txt   # List of dependencies required for the project
```

---

## Endpoints

### 1. **Upload CSV File**
**Endpoint**: `/upload`
- **Method**: `POST`
- **Description**: Upload a CSV file containing manufacturing data.
- **Request**:
  - Form-data: `file` (The CSV file to upload)
- **Response**:
  - Success: `{ "message": "File <filename> uploaded successfully", "path": "<file_path>" }`
  - Error: `{ "error": "No file provided" }`

### 2. **Train Model**
**Endpoint**: `/train`
- **Method**: `POST`
- **Description**: Train the model using the uploaded CSV file. The model uses Logistic Regression, and hyperparameters are optimized using Optuna.
- **Response**:
  - Success: `{ "message": "Model trained successfully", "metrics": { "accuracy": <accuracy>, "f1_score": <f1_score> } }`
  - Error: `{ "error": "<error_message>" }`

### 3. **Predict Downtime**
**Endpoint**: `/predict`
- **Method**: `POST`
- **Description**: Predict machine downtime based on input parameters.
- **Request**:
  - JSON: `{ "temperature": <float>, "run_time": <float> }`
- **Response**:
  - Success: `{ "Downtime": "Yes"/"No", "Confidence": <confidence_score> }`
  - Error: `{ "error": "<error_message>" }`

---

## Features
1. **Model Training**: Uses Logistic Regression with hyperparameter tuning via Optuna.
2. **Balanced Dataset**: Applies SMOTE to handle class imbalance in the dataset.
3. **Evaluation Metrics**: Provides accuracy and F1-score for model evaluation.
4. **RESTful API**: Three endpoints to upload data, train the model, and predict outcomes.

---

## Usage Instructions

### Prerequisites
- Python 3.7 or above installed on your machine.
- Ensure the required dependencies are installed using:
  ```bash
  pip install -r requirements.txt
  ```

### Running the Application
1. **Start the Flask Server**:
   ```bash
   python newMain.py
   ```
2. **Access the API Endpoints**:
   Use tools like [Postman](https://www.postman.com/) or `curl` to test the endpoints.

### Example Workflow

#### Using Postman
1. **Upload Data**:
   - Open Postman.
   - Select `POST` method and enter the URL: `http://127.0.0.1:5000/upload`.
   - Under the `Body` tab, select `form-data` and add a key `file` with the file to upload.
   - Click `Send`.
2. **Train Model**:
   - Select `POST` method and enter the URL: `http://127.0.0.1:5000/train`.
   - Click `Send`.
3. **Predict Downtime**:
   - Select `POST` method and enter the URL: `http://127.0.0.1:5000/predict`.
   - Under the `Body` tab, select `raw` and choose JSON format. Provide the input:
     ```json
     {
       "temperature": 300,
       "run_time": 120
     }
     ```
   - Click `Send`.

#### Using cURL
1. **Upload Data**:
   ```bash
   curl -X POST -F "file=@<path_to_csv_file>" http://127.0.0.1:5000/upload
   ```
2. **Train Model**:
   ```bash
   curl -X POST http://127.0.0.1:5000/train
   ```
3. **Predict Downtime**:
   ```bash
   curl -X POST -H "Content-Type: application/json" -d '{"temperature": 300, "run_time": 120}' http://127.0.0.1:5000/predict
   ```

---

## Example Dataset Format
| Downtime flag | Temperature | Run time |
|---------------|-------------|----------|
| 0             | 13          | 13.2     |
| 0             | 40          | 14.2     |
| 0             | 42          | 15.0     |
| 0             | 37          | 12.8     |
| 0             | 50          | 16.5     |
| 0             | 20          | 11.9     |
| 1             | 55          | 19.6     |
| 1             | 60          | 20.5     |
| 1             | 58          | 19.8     |
| 1             | 62          | 20.2     |
| 1             | 59          | 20.1     |
| 1             | 61          | 20.4     |

### Notes:
- Ensure the dataset contains the required columns: `Downtime flag`, `Temperature`, and `Run time` if using your own dataset.
- Missing or incorrect columns will result in an error during training.

---

## Dependencies
The project dependencies are listed in `requirements.txt` and include:
- Flask: Web framework for creating the API.
- Pandas: Data manipulation and analysis.
- Scikit-learn: Machine learning library for Logistic Regression.
- Imbalanced-learn: To apply SMOTE for handling imbalanced datasets.
- Optuna: Hyperparameter optimization framework.
- Joblib: Model serialization and deserialization.

---

## License
This project is licensed under the [MIT License](https://github.com/Adityagupta200/Manufacturing-Operations-Predictions/blob/8c9984b7ccbe9bc7208e671ed57db0003cb705f6/LICENSE).

