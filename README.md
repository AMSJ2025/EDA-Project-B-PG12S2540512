# EDA Project B Starter — PG12S2540512

Student: Ahmed Al Omairi  
Student ID: PG12S2540512

This repository contains a one-file Streamlit starter app for Mini Project B time-series forecasting.

## Files

- `app.py` — single-file Streamlit app
- `requirements.txt` — required Python packages
- `data/dataset_sample.csv` — cleaned and sliced dataset sample

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## OpenRouter API key

The app reads the API key in this order:

1. Streamlit Secrets: `OPENROUTER_API_KEY`
2. Environment variable: `OPENROUTER_API_KEY`
3. Password input field in the app

For Streamlit Community Cloud, add the key under **App settings → Secrets**:

```toml
OPENROUTER_API_KEY = "your_key_here"
```

## Deploy on Streamlit Community Cloud

1. Create a public GitHub repository.
2. Upload exactly these files:
   - `app.py`
   - `requirements.txt`
   - `README.md`
   - `data/dataset_sample.csv`
3. Open Streamlit Community Cloud.
4. Create a new app.
5. Connect the repository.
6. Use branch `main`.
7. Set the main file path to `app.py`.
8. Deploy.

## What to submit

Submit the following to your instructor:

- Streamlit deployed app URL
- GitHub repository URL
- Exported `submission.json`
- Exported `project_card.md`
