# app.py - Gradio Interface for Smoking Prediction to Open with Hugging Face

This document explains how the **`app.py`** file works for Hugging Face Spaces deployment, its user interface, and model integration.

## 📁 Required Files for app.py

```
├── app.py                                    # Main Gradio application
├── optimized_random_forest_model.pkl        # Trained model (246MB)
├── preprocessor.pkl                          # Data preprocessing pipeline (2.7KB)
└── requirements.txt                          # Dependencies
```

## 🔧 How app.py Works

### 1. Model Loading Process

```python
# On startup, app.py loads both pkl files:
model = joblib.load('optimized_random_forest_model.pkl')          # Random Forest classifier
preprocessor = joblib.load('preprocessor.pkl')                    # Data preprocessing pipeline
```

### 2. Data Processing Pipeline

When user submits health data:

1. **Input Collection**: Gradio form collects 24 health parameters
2. **Data Preprocessing**: `preprocessor.pkl` transforms raw inputs (scaling + encoding)
3. **Prediction**: `optimized_random_forest_model.pkl` generates smoking probability
4. **Explanation**: LIME analyzes which features influenced the prediction

### 3. Where PKL Files Are Used

| PKL File                            | Used For                                             | Called When      |
| ----------------------------------- | ---------------------------------------------------- | ---------------- |
| `preprocessor.pkl`                  | Data transformation (StandardScaler + OneHotEncoder) | Every prediction |
| `optimized_random_forest_model.pkl` | Smoking status prediction                            | Every prediction |

## 🖥️ User Interface Overview

### Input Form (24 Health Parameters)

The Gradio interface provides:

**Demographic Inputs:**

- Age (20-85), Gender (M/F), Height (130-190cm), Weight (30-135kg), Waist (51-129cm)

**Health Measurements:**

- Blood Pressure: Systolic (71-240), Diastolic (40-146)
- Blood Chemistry: Cholesterol (55-445), Triglycerides (8-999), HDL (4-618), LDL (1-1860)
- Liver Function: AST (6-1311), ALT (1-2914), GTP (1-999)
- Other: Hemoglobin (4.9-21.1), Blood Sugar (46-505), Creatinine (0.1-11.6)
- Sensory: Vision L/R (0.1-9.9), Hearing L/R (1.0-2.0)
- Dental: Tartar (Y/N), Caries (0/1), Urine Protein (1-6)

### Output Display

**Prediction Results:**

- Primary prediction: "Smoker" or "Non-Smoker"
- Confidence percentages for both classes
- Color-coded risk assessment

**LIME Explanation:**

- Top 5 most influential features for this specific prediction
- Shows which health indicators contributed to the decision

## 🔄 Prediction Workflow

```python
def make_prediction(age, gender, height, weight, ...):
    # 1. Create DataFrame from user inputs
    input_data = pd.DataFrame({...})

    # 2. Transform data using preprocessor.pkl
    processed_data = preprocessor.transform(input_data)

    # 3. Generate prediction using model.pkl
    prediction_proba = model.predict_proba(processed_data)
    prediction = model.predict(processed_data)

    # 4. Create LIME explanation
    lime_explanation = get_instance_feature_importance_lime(processed_data)

    # 5. Return formatted results
    return prediction_result, confidence_score, lime_features
```

## 💡 Key Features of app.py

### Error Handling

- **Model Loading**: Graceful failure if pkl files are missing
- **Input Validation**: Checks for valid ranges and data types
- **LIME Fallback**: Uses global feature importance if LIME fails

### LIME Integration

- **Dynamic Explainer**: Initializes LIME explainer on first prediction
- **Feature Importance**: Shows top 5 features affecting each prediction
- **Personalized Insights**: Different explanations for different health profiles

### Cloud Optimization

- **Logging**: Comprehensive error logging for Hugging Face Spaces
- **Memory Efficient**: Loads models once, reuses for all predictions
- **Performance Tracking**: Built-in timing for prediction performance

## 📊 Interface Layout

```
┌─────────────────────────────────────────┐
│              INPUT SECTION              │
├─────────────────────────────────────────┤
│ Age: [slider]  Gender: [dropdown]       │
│ Height: [slider]  Weight: [slider]      │
│ Blood Pressure: [slider] [slider]       │
│ Cholesterol: [slider]  HDL: [slider]    │
│ ... (24 total parameters)               │
│                                         │
│      [Predict Smoking Status]           │
├─────────────────────────────────────────┤
│             OUTPUT SECTION              │
├─────────────────────────────────────────┤
│ Result: SMOKER (78.5% confidence)       │
│ Non-Smoker: 21.5% | Smoker: 78.5%      │
│                                         │
│ Most Influential Features:              │
│ 1. hemoglobin (15.2)                    │
│ 2. gender_M (1.0)                       │
│ 3. triglyceride (245)                   │
│ 4. height(cm) (175)                     │
│ 5. Gtp (45)                             │
└─────────────────────────────────────────┘
```

## ⚡ Performance

- **Model Loading**: ~5-10 seconds on first load
- **Prediction Time**: ~1-2 seconds per prediction
- **Memory Usage**: ~300MB (including both pkl files)
- **LIME Explanation**: ~2-3 seconds additional processing

## 🔧 Technical Notes

### Model Dependencies

Both pkl files must be present in the same directory as app.py for successful operation.

### Data Preprocessing

The `preprocessor.pkl` ensures that user inputs are transformed exactly as they were during model training, maintaining prediction accuracy.

### LIME Explanations

Local explanations help users understand which specific health factors led to their smoking prediction, making the AI decision transparent.
