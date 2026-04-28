# TripleMap Operational System

## Overview
This repository contains pipeline scripts for the **TRIPLET Challenge Subtask 2**, covering three datasets:
- Business
- Telco
- Female Celebrity

Each pipeline processes structured/semi-structured data and generates relational triples for knowledge graph construction.

---

## Prerequisites
- Python 3.11.9  
- pip (Python package manager)

---

## Installation

### 1. Create a virtual environment
```bash
python -m venv venv
```

### 2. Activate the virtual environment:
    - **Windows:**
      ```bash
      venv\Scripts\activate
      ```
    - **macOS/Linux:**
      ```bash
      source venv/bin/activate
      ```

### 3. Install dependencies:
```bash
pip install -r requirements.txt
```
### 4. OpenAI Credentials Setup
Add your OpenAI API key in the `config/openai_credentials.txt` as follows:

api_key=your_api_key_here

## Project Structure

.
├── business_pipeline.py
├── telco_pipeline.py
├── female_celebrity_pipeline.py
├── requirements.txt
├── business_data.json
├── female_celebrity_data.json
├── telco_data.json
├── data/
│   ├── business_data_test.json
│   ├── telco_data_test.json
│   ├── female_celebrity_data_test.json

## Running the Scripts

Run each dataset pipeline independently:

```bash
python business_pipeline.py 
python telco_pipeline.py 
python celebrity_pipeline.py
```

The data for each pipeline is stored in its respective JSON file.

- For `business_pipeline.py`, the `cleaned_data/business_data.json` file contains the list of documents from the business test dataset.
- For `telco_pipeline.py`, the `cleaned_data/telco_data.json` file contains the list of documents from the telco test dataset.
- For `female_celebrity_pipeline.py`, the `cleaned_data/female_celebrity_data.json` file contains the list of documents from the female celebrity test dataset.

The generated triples for each table are printed to the terminal during execution of the pipeline scripts and are not stored in external output files.