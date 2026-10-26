# Dual Allegiance in Practice — Streamlit app

An interactive companion to the "Dual Allegiance in Practice" analysis: trust, corruption
perception, and performance approval of traditional leaders vs. elected local government
in Ghana, built on Afrobarometer Round 9 (2022) survey data (n = 2,369).

**Tabs:** a filterable dashboard recreating the report's core charts, a live prediction
tool (logistic regression + Random Forest, trained in-session), and a data explorer for
searching/filtering/downloading the underlying CSVs.

## Run locally

```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

The app opens at `http://localhost:8501`. Data is read from `data/ghana_r9_clean.csv` and
`data/ghana_r9_raw.csv` (already included) — no external downloads or API keys needed.

## Deploy to Streamlit Community Cloud (free, public link)

1. Push this repository to GitHub (the app lives in `streamlit_app/`).
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
3. Click **New app**, pick the repository/branch, set **Main file path** to `streamlit_app/app.py`, and
   deploy. The first build takes a minute or two.
4. You'll get a public URL like `https://<your-app-name>.streamlit.app` — good for a
   portfolio link or a resume/LinkedIn callout.

No secrets or environment variables are required for this app. If you later add any
(e.g. an API key), put them in `.streamlit/secrets.toml` locally (already git-ignored)
and in the Streamlit Cloud app's **Settings → Secrets**, never committed to the repo.

## Project structure

```
app.py                  Main Streamlit app (single file, four tabs)
requirements.txt        Python dependencies
.streamlit/config.toml  Theme (kente-inspired blue/gold palette)
data/
  ghana_r9_clean.csv     Recoded, numeric-scale dataset used by the app
  ghana_r9_raw.csv        Original Afrobarometer text-label dataset (Data Explorer tab)
```

## Data source

Afrobarometer Round 9 (2022), Ghana country file, retrieved via the public
[Afrobarometer-Data-Explorer](https://github.com/MafiAtUN/Afrobarometer-Data-Explorer)
mirror of the official Afrobarometer microdata (afrobarometer.org), redistributed for
research/educational use.

## Notes & caveats

The Predict tab's models are intentionally the same ones from the original analysis
notebook — including their honest limitation: demographics alone predict who leans
traditional only barely better than chance (ROC-AUC ≈ 0.57–0.58 on held-out data). Read
predictions as illustrative of the modeling approach, not as reliable individual forecasts.
