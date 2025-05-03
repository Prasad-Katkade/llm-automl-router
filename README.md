# 🧠 LLM-Driven AutoML Orchestrator (MCP-lite)

A lightweight implementation of the Model-Context-Protocol (MCP) architecture using **FastAPI**, **OpenAI**, and **H2O AutoML**. This project allows secure and scalable interaction with in-house ML models via natural language prompts, keeping sensitive data protected while leveraging the power of LLMs.

<img width="665" alt="Image" src="https://github.com/user-attachments/assets/511a0cdb-6a02-43e6-8de5-27602c8ddeaf" />

---

## 🚀 Features

- Natural language interface to interact with ML models
- In-house AutoML model serving using H2O
- FastAPI-based microservice architecture
- No need to expose private data to third-party LLM providers
- JSON-based protocol to route requests dynamically

---

## 📦 Tech Stack

- **H2O AutoML** for training and serving ML models
- **OpenAI GPT** via LangChain for language understanding
- **FastAPI** for microservice APIs
- **Pydantic** for schema validation

---

## ⚙️ Prerequisites

> H2O AutoML requires **Java 21 or higher**. Ensure it’s installed and configured in your system’s PATH.


Install required Python packages:

```bash
pip install h2o pandas fastapi uvicorn pydantic
pip install langchain langchain-community langchain-openai
```
## 🚦 Let’s Run the Program

Follow these steps to set up and run the full pipeline.

### 1.  Train the Models

Make sure to first **train the price and quantity models**:

- Open the Jupyter notebooks located in the `notebooks/` folder:
  - `H20_Automl_Price.ipynb`
  - `H20_Automl_Quantity.ipynb`
- Run the cells in order to train the models
- Save the trained models into their respective folders.
  
### 2. Start the APIs

> Make sure to update the model paths in the scripts before running.


```bash
uvicorn h2o_quantity_api_script:app --host 0.0.0.0 --port 3000
uvicorn h2o_price_api_script:app --host 0.0.0.0 --port 8000
uvicorn lang_chain_api_script:app --host 0.0.0.0 --port 4000
```

<img width="642" alt="Image" src="https://github.com/user-attachments/assets/33aaa01b-887a-4a6f-9ed2-f6570ea5cfed" />
<img width="642" alt="Image" src="https://github.com/user-attachments/assets/7aeb8097-0796-49b2-abc1-c0637078db24" />
