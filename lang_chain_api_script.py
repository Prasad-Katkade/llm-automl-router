#
#pip3 install fastapi pandas pydantic langchain langchain-community langchain-openai requests uvicorn


from fastapi import FastAPI, HTTPException
import pandas as pd
from pydantic import BaseModel
from langchain_openai import OpenAI
import os
import json
import requests
import re

# Load sales data
df = pd.read_csv("sales_data.csv")  # Ensure CSV exists

# Initialize FastAPI app
app = FastAPI()

# Define request model
class RequestModel(BaseModel):
    prompt: str

llm = OpenAI(api_key="")  # Replace with your actual API key

def get_request_type_and_details(prompt: str):
   
    full_prompt = f"""
    You are an AI that classifies user queries into three types: "price", "quantity", or "invalid".
    
    **Classification Rules**:
    - If the query asks about **price** or its synonyms (e.g., "price", "cost", "rate", "net price", "selling price"), return `"type": "price"`.
    - If the query asks about **quantity** or its synonyms (e.g., "quantity", "requirement", "stock", "demand", "supply"), return `"type": "quantity"`.
    - If the query does **not** relate to price or quantity (e.g., "growth rate", "what is X?", "explain X", "uses of X"), return `"type": "invalid"`.

    **Extract the following details**:
    - `material_name`: Extract the product name (e.g., "Blueberry Muesli").
    - `month`: Convert textual months (e.g., "February" → `2`). If missing, return `null`.
    - `year`: Extract the year as a number. If missing, return `null`.
    - `quantity`: **Only include this if `"type": "price"`**. Otherwise, exclude it.

    **JSON Response Format**:
    - **For price queries**:
      ```json
      {{
          "type": "price",
          "data": {{
              "material_name": "extracted_name",
              "quantity": quantity_value,
              "month": month_value,
              "year": year_value
          }}
      }}
      ```
    - **For quantity queries (DO NOT include "quantity")**:
      ```json
      {{
          "type": "quantity",
          "data": {{
              "material_name": "extracted_name",
              "month": month_value,
              "year": year_value
          }}
      }}
      ```
    - **For invalid queries**:
      ```json
      {{
          "type": "invalid",
          "data": {{
              "material_name": "extracted_name",
              "month": month_value,
              "year": year_value
          }}
      }}
      ```

    **User Query**: "{prompt}"
    """


    response = llm.predict(full_prompt) 

    print("openAI resp-\n",response)

    try:
        response_json = json.loads(response.strip())  # Ensure OpenAI returns valid JSON
        return response_json
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="Error parsing LLM response: Invalid JSON format")


def validate_material(material_name: str):
    df["MATERIAL_DESCRIPTION"] = df["MATERIAL_DESCRIPTION"].str.lower().str.strip()
    matching_materials = df[df["MATERIAL_DESCRIPTION"].str.contains(material_name.lower(), na=False)]
    
    return not matching_materials.empty

def call_external_api(api_url: str, data: dict):
    try:
        response = requests.post(api_url, json=data)
        response.raise_for_status()  # Ensure we get a valid response
        return response.json()
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Error calling external API: {str(e)}")
    
def process_data(data: dict):
    # Convert quantity (if exists) to a number
    if data.get("type") == "price" and "quantity" in data["data"]:
        quantity_str = str(data["data"]["quantity"]).lower().strip()
        
        # Extract numeric value and convert
        quantity_match = re.match(r"(\d+)", quantity_str)
        if quantity_match:
            data["data"]["quantity"] = int(quantity_match.group(1))  # Convert to integer
        else:
            data["data"]["quantity"] = 1  # Default if not a valid number

    # Ensure month is numeric
    month_map = {
        "january": 1, "jan": 1, "february": 2, "feb": 2, "march": 3, "mar": 3,
        "april": 4, "apr": 4, "may": 5, "june": 6, "jun": 6, "july": 7, "jul": 7,
        "august": 8, "aug": 8, "september": 9, "sep": 9, "october": 10, "oct": 10,
        "november": 11, "nov": 11, "december": 12, "dec": 12
    }
    
    if "month" in data["data"]:
        month_str = str(data["data"]["month"]).lower().strip()
        if month_str in month_map:
            data["data"]["month"] = month_map[month_str]  # Convert to numeric month
    
    # Fill missing values
    if "quantity" not in data["data"] or data["data"]["quantity"] is None:
        data["data"]["quantity"] = 1  # Default for price queries
    
    if "month" not in data["data"] or data["data"]["month"] is None:
        data["data"]["month"] = 1  # Default to January
    
    if "year" not in data["data"] or data["data"]["year"] is None:
        data["data"]["year"] = 2027  # Default year

    return data




@app.post("/ask")
def ask_question(request: RequestModel):
    prompt = request.prompt.strip()

    # Extract query details using OpenAI
    query_details = get_request_type_and_details(prompt)

    query_type = query_details.get("type")
    data = query_details.get("data", {})

    if query_type == "invalid":
        raise HTTPException(status_code=400, detail="Invalid request. Please ask about price or quantity.")

    # Validate material name
    material_name = data.get("material_name", "").lower()
    if not validate_material(material_name):
        return {"error": "No material name found in sales data."}

    # Fill missing values with defaults
    data = process_data({"type": query_type, "data": data})

    # Define API URLs
    api_urls = {
        "price": "http://127.0.0.1:8000/predict_price",
        "quantity": "http://127.0.0.1:3000/predict_quantity"
    }

    # Call the appropriate API
    api_url = api_urls.get(query_type)
    api_response = call_external_api(api_url, data["data"])

    # Return response in the required format
    return {
        "type": query_type,
        "data": api_response
    }

# To run 
# uvicorn lang_chain_api_script:app --host 0.0.0.0 --port 4000