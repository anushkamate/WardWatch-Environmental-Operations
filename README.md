# WardWatch — Hospital Environment Monitoring

A polished Streamlit dashboard for exploring indoor air-quality measurements and demonstrating machine-learning-based review flags for simulated hospital rooms.

## What it does
- Uploads and validates a CSV dataset.
- Cleans numeric measurements and prepares a rule-labelled demonstration dataset.
- Trains a Random Forest pipeline and reports held-out evaluation metrics.
- Provides room-reading inference, data exploration, and downloadable CSV reports.
- Uses a restrained, responsive dashboard interface designed for facilities workflows.

## Required columns
`NH3`, `NO2`, `CO`, `PM2.5`, `Temp`, `Pressure`, `Humidity`, `O3`

Some common column name variants are recognized. Confirm that measurement units are consistent with your source dataset.

## Run locally
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Deploy to Streamlit Community Cloud
1. Create a GitHub repository and upload `app.py`, `requirements.txt`, `README.md`, and `.gitignore`.
2. Check that the dataset's licence allows redistribution before adding it to GitHub. The recommended setup is to keep it out of the repo and upload it through the app.
3. Visit https://share.streamlit.io/ and connect GitHub.
4. Create an app from your repository and select `app.py` as the main file.
5. Deploy, open the app URL, and upload the CSV using the sidebar.

## Important limitations
This is an educational prototype, not a certified hospital monitoring or clinical decision system. The source dataset does not contain verified labels for hospital room safety. `Normal`, `Review`, and `Priority` are generated from percentile-based pollution thresholds and illustrative temperature/humidity limits. Accuracy measures agreement with those generated rules and must not be presented as validated safety performance.

Room names in the app are simulated assignments; there is no live sensor integration or persistent cloud database. Before any real-world use, thresholds must be approved for the specific environment and the model validated on independently labelled data.

## Repository files
- `app.py` — dashboard and in-app model training
- `requirements.txt` — deployment dependencies
- `README.md` — project and deployment instructions
- `.gitignore` — excludes local datasets, model artefacts, virtual environments and secrets
- `.streamlit/config.toml` — optional Streamlit theme settings
