import gradio as gr
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import os
import sys
from lime import lime_tabular

# Error handling information
print("Python version:", sys.version)
print("Current working directory:", os.getcwd())
print("Files in current directory:", os.listdir())

# Global LIME explainer variable
lime_explainer = None

# Model and preprocessor loading in a try-except block
try:
    print("Model loading attempt...")
    model = joblib.load('optimized_random_forest_model.pkl')
    print("Model loaded successfully!")
    
    print("Preprocessor loading attempt...")
    preprocessor = joblib.load('preprocessor.pkl')
    print("Preprocessor loaded successfully!")
    
    # Define numeric and categorical features
    categorical_features = ['gender', 'oral', 'tartar']
    numerical_features = ['age', 'height(cm)', 'weight(kg)', 'waist(cm)', 'eyesight(left)', 
                        'eyesight(right)', 'hearing(left)', 'hearing(right)', 'systolic', 
                        'relaxation', 'fasting blood sugar', 'Cholesterol', 'triglyceride', 
                        'HDL', 'LDL', 'hemoglobin', 'Urine protein', 'serum creatinine', 
                        'AST', 'ALT', 'Gtp', 'dental caries']

    # Preparing feature names
    try:
        print("Preparing feature names...")
        numeric_feature_names = numerical_features.copy()
        categorical_encoder = preprocessor.named_transformers_['cat']
        onehot_encoder = categorical_encoder.named_steps['onehot']
        encoded_feature_names = onehot_encoder.get_feature_names_out(categorical_features)
        all_feature_names = np.concatenate([numeric_feature_names, encoded_feature_names])
        print("Feature names prepared successfully!")
        
        # Check the values that categorical variables can take
        print("Checking categorical variable categories...")
        categories = onehot_encoder.categories_
        for i, cat in enumerate(categorical_features):
            print(f"{cat} categories: {categories[i]}")
            
        # LIME explainer creation
        # We initialize it as None and will initialize it during the first prediction
        print("LIME explainer will be initialized during first prediction")
            
    except Exception as e:
        print(f"Error preparing feature names: {e}")
        all_feature_names = numerical_features + [f"{cat}_{val}" for cat in categorical_features for val in ['M', 'F', 'Y', 'N']]
        print("Using default feature names:", all_feature_names[:5], "...")

except Exception as e:
    print(f"Critical error loading model or preprocessor: {e}")
    print("Model files may be missing or corrupted.")
    error_message = f"Error loading model: {str(e)}"

def get_instance_feature_importance_lime(processed_data, input_data):
    """
    LIME is used to calculate the feature importance values for a prediction
    """
    global lime_explainer
    
    try:
        # Initialize the LIME explainer (first use)
        if lime_explainer is None:
            print("Initializing LIME explainer for the first time...")
            # Determine feature_names and class_names for LIME
            lime_explainer = lime_tabular.LimeTabularExplainer(
                training_data=processed_data,  # Use the first example as training data
                feature_names=all_feature_names,
                class_names=["Non-Smoker", "Smoker"],
                mode="classification",
                discretize_continuous=True
            )
            print("LIME explainer initialized")

        # Generate LIME explanation
        print("Generating LIME explanation...")
        explanation = lime_explainer.explain_instance(
            data_row=processed_data[0],
            predict_fn=model.predict_proba,
            num_features=5,  # Show the top 5 features
            top_labels=1     # Only the most likely class
        )
        
        # Get the explanation for the most likely class
        most_likely_class_idx = explanation.top_labels[0]
        
        # Get the top 5 features
        feature_importance = explanation.as_list(label=most_likely_class_idx)
        top_features = [feature_name for feature_name, importance in feature_importance]
        
        print(f"LIME explanation generated: {top_features}")
        return top_features
    
    except Exception as e:
        print(f"Error calculating LIME feature importance: {e}")
        import traceback
        traceback.print_exc()
        
        # If LIME fails, use the default global feature importance
        print("Falling back to global feature importances...")
        try:
            feature_importances = model.feature_importances_
            feature_importance_df = pd.DataFrame({
                'Feature': all_feature_names,
                'Importance': feature_importances
            }).sort_values('Importance', ascending=False)
            
            return feature_importance_df.head(5)['Feature'].tolist()
        except:
            return ["Feature importance calculation failed"]

def make_prediction(age, gender, height, weight, waist, eyesight_left, eyesight_right,
                   hearing_left, hearing_right, systolic, relaxation, fasting_blood_sugar,
                   cholesterol, triglyceride, hdl, ldl, hemoglobin, urine_protein,
                   serum_creatinine, ast, alt, gtp, tartar, dental_caries):
    
    try:
        print("Prediction function called")
        
        # In the original dataset, the oral value is fixed as 'Y', so we prepare the categorical converter accordingly
        oral = 'Y'  # Oral always has the value 'Y'
        
        # Check the values of categorical variables
        try:
            categorical_encoder = preprocessor.named_transformers_['cat']
            onehot_encoder = categorical_encoder.named_steps['onehot']
            categories = onehot_encoder.categories_
            
            # Check the values used by the model
            gender_values = categories[0]  # gender categories
            tartar_values = categories[2]  # tartar categories
            
            print(f"Valid gender values for the model: {gender_values}")
            print(f"Valid tartar values for the model: {tartar_values}")
            
            # Check valid values
            if gender not in gender_values:
                print(f"WARNING: '{gender}' is not a valid 'gender' value. Using '{gender_values[0]}'.")
                gender = gender_values[0]
                
            if tartar not in tartar_values:
                print(f"WARNING: '{tartar}' is not a valid 'tartar' value. Using '{tartar_values[0]}'.")
                tartar = tartar_values[0]
        except Exception as e:
            print(f"Error during categorical variable validation: {e}")
        
        # Create a DataFrame with the input values
        input_data = pd.DataFrame({
            'age': [age],
            'gender': [gender],
            'height(cm)': [height],
            'weight(kg)': [weight],
            'waist(cm)': [waist],
            'eyesight(left)': [eyesight_left],
            'eyesight(right)': [eyesight_right],
            'hearing(left)': [hearing_left],
            'hearing(right)': [hearing_right],
            'systolic': [systolic],
            'relaxation': [relaxation],
            'fasting blood sugar': [fasting_blood_sugar],
            'Cholesterol': [cholesterol],
            'triglyceride': [triglyceride],
            'HDL': [hdl],
            'LDL': [ldl],
            'hemoglobin': [hemoglobin],
            'Urine protein': [urine_protein],
            'serum creatinine': [serum_creatinine],
            'AST': [ast],
            'ALT': [alt],
            'Gtp': [gtp],
            'oral': [oral],  # Fixed 'Y' value used
            'tartar': [tartar],
            'dental caries': [dental_caries]
        })
        
        print("Preprocessing data...")
        # Apply preprocessing
        processed_data = preprocessor.transform(input_data)
        print("Preprocessing completed!")
        
        print("Making prediction...")
        # Make prediction
        prediction_proba = model.predict_proba(processed_data)[0]
        prediction = model.predict(processed_data)[0]
        print(f"Prediction result: {prediction}, Probability: {prediction_proba}")
        
        # Prepare result message
        if prediction == 1:
            result = "Smoker"
            probability = prediction_proba[1] * 100
        else:
            result = "Non-Smoker"
            probability = prediction_proba[0] * 100
        
        # Calculate the instance-specific feature importance with LIME
        print("Calculating instance-specific feature importance with LIME...")
        top_features = get_instance_feature_importance_lime(processed_data, input_data)
        top_features_str = ", ".join(top_features)
        print(f"Top features for this instance: {top_features_str}")
        
        print("Prediction completed, results returned")
        return result, f"{probability:.2f}%", top_features_str
    
    except Exception as e:
        print(f"Unexpected error during prediction: {e}")
        return f"Error: {str(e)}", "Error occurred", "Prediction failed"

# Create Gradio interface
print("Creating Gradio interface...")
with gr.Blocks(title="Smoking Status Prediction") as app:
    gr.Markdown("# Smoking Status Prediction Model")
    gr.Markdown("""This application predicts whether a person is a smoker based on health and demographic data.
    The model uses Random Forest classifier trained on a dataset of health indicators.""")
    
    with gr.Tab("Make Prediction"):
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Demographics")
                age = gr.Slider(18, 90, step=1, label="Age", value=40)
                gender = gr.Dropdown(["M", "F"], label="Gender", value="M")
                height = gr.Slider(140, 200, step=1, label="Height (cm)", value=170)
                weight = gr.Slider(40, 150, step=1, label="Weight (kg)", value=70)
                waist = gr.Slider(50, 150, step=1, label="Waist Circumference (cm)", value=85)
                
                gr.Markdown("### Vision and Hearing")
                eyesight_left = gr.Slider(0.1, 2.0, step=0.1, label="Eyesight (Left)", value=1.0)
                eyesight_right = gr.Slider(0.1, 2.0, step=0.1, label="Eyesight (Right)", value=1.0)
                hearing_left = gr.Slider(0.0, 1.0, step=0.1, label="Hearing (Left)", value=1.0)
                hearing_right = gr.Slider(0.0, 1.0, step=0.1, label="Hearing (Right)", value=1.0)
                
            with gr.Column(scale=1):
                gr.Markdown("### Blood Pressure and Sugar")
                systolic = gr.Slider(90, 200, step=1, label="Systolic Blood Pressure", value=120)
                relaxation = gr.Slider(50, 150, step=1, label="Diastolic Blood Pressure", value=80)
                fasting_blood_sugar = gr.Slider(50, 300, step=1, label="Fasting Blood Sugar", value=95)
                
                gr.Markdown("### Cholesterol Profile")
                cholesterol = gr.Slider(100, 350, step=1, label="Total Cholesterol", value=190)
                triglyceride = gr.Slider(30, 600, step=1, label="Triglyceride", value=120)
                hdl = gr.Slider(20, 100, step=1, label="HDL Cholesterol", value=55)
                ldl = gr.Slider(30, 250, step=1, label="LDL Cholesterol", value=110)
                
            with gr.Column(scale=1):
                gr.Markdown("### Blood and Urine Tests")
                hemoglobin = gr.Slider(8, 20, step=0.1, label="Hemoglobin", value=14.0)
                urine_protein = gr.Slider(1, 6, step=1, label="Urine Protein", value=1)
                serum_creatinine = gr.Slider(0.2, 2.0, step=0.1, label="Serum Creatinine", value=0.9)
                
                gr.Markdown("### Liver Function")
                ast = gr.Slider(10, 200, step=1, label="AST", value=25)
                alt = gr.Slider(10, 200, step=1, label="ALT", value=25)
                gtp = gr.Slider(10, 300, step=1, label="GTP", value=30)
                
                gr.Markdown("### Dental Health")
                # We don't use input for the 'oral' variable because it's always 'Y' in the original dataset
                tartar = gr.Dropdown(["Y", "N"], label="Tartar", value="Y")
                dental_caries = gr.Slider(0, 3, step=1, label="Dental Caries", value=0)
        
        predict_btn = gr.Button("Predict Smoking Status")
        
        with gr.Row():
            with gr.Column():
                prediction_result = gr.Textbox(label="Prediction Result")
                probability_result = gr.Textbox(label="Confidence")
                top_features = gr.Textbox(label="Top Features for This Prediction")
    
    with gr.Tab("Model Information"):
        gr.Markdown("## Model Information")
        gr.Markdown("""
        ### Random Forest Classifier
        
        This model uses a Random Forest Classifier to predict smoking status based on health indicators. 
        
        The model was trained on a dataset containing demographic, biometric, and health-related features.
        
        ### Top Features Used by the Model
        
        The most important features for prediction include:
        - Hemoglobin
        - HDL Cholesterol (higher in non-smokers)
        - Age
        - Gender
        - Height
        - Weight
        
        ### Model Performance
        
        On the test dataset, the model achieved:
        - Accuracy: ~80%
        - Precision: ~78%
        - Recall: ~79%
        """)
    
    # Connect the prediction function
    predict_btn.click(
        fn=make_prediction,
        inputs=[age, gender, height, weight, waist, eyesight_left, eyesight_right,
               hearing_left, hearing_right, systolic, relaxation, fasting_blood_sugar,
               cholesterol, triglyceride, hdl, ldl, hemoglobin, urine_protein,
               serum_creatinine, ast, alt, gtp, tartar, dental_caries],
        outputs=[prediction_result, probability_result, top_features]
    )

print("Gradio interface created, launching application...")
# Launch the app
if __name__ == "__main__":
    app.launch() 