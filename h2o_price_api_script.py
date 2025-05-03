
# Install Libs 
# !pip install h2o pandas fastapi uvicorn pydantic

import h2o  # Import H2O for machine learning
import pandas as pd  # Pandas for data handling
from fastapi import FastAPI  # FastAPI for creating API service
from pydantic import BaseModel  # Pydantic for request validation

# Initialize FastAPI
app = FastAPI()

# Initialize H2O
h2o.init()

# Load the trained H2O AutoML model
model_path = "./best_price_model/GBM_5_AutoML_1_20250503_154925"  # Update with actual model path
model = h2o.load_model(model_path)

# Define request format using Pydantic
class POSTRequestSchema(BaseModel):
    material_name: str
    quantity: int
    month: int
    year: int
    

# Define the prediction endpoint
@app.post("/predict_price")
def predict_values(request: POSTRequestSchema):
    # Convert request to DataFrame
    input_data = pd.DataFrame([{
        "MATERIAL_LABEL": request.material_name,
        "QUANTITY": request.quantity,
        "SIM_PERIOD": request.month,
        "SIM_YEAR": request.year
    }])

    # Convert to H2OFrame
    input_h2o = h2o.H2OFrame(input_data)

    # Ensure categorical variables are properly encoded
    for col in input_h2o.columns:
        if input_h2o[col].isfactor()[0]:  # If column is categorical
            input_h2o[col] = input_h2o[col].asfactor()  # Convert it to categorical

    # Debugging: Print model features & input
    print("Model Features:", model.get_params())
    print("Input Data for Prediction:")
    print(input_h2o)

    # Predict price
    prediction = model.predict(input_h2o)
    predicted_price = prediction.as_data_frame().iloc[0, 0]  # Extract first prediction

    # Response JSON
    return {
        "predicted_price": round(predicted_price, 2),
        "material_name": request.material_name
    }

# To run
# uvicorn h2o_price_api_script:app --host 0.0.0.0 --port 8000