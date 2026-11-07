# Metro Manila Barangay Heat Hazard Forecast (Dashboard Prototype)

This is the Streamlit prototype for the thesis *Explainable Hybrid Attention-LSTM and XGBoost Framework for Spatiotemporal Heat Stress Risk Forecasting in Metro Manila*. It has two modes:
- **Live 3-day forecast:** forecasts the DOST-PAGASA heat index category of all 1,710 barangays, using Open-Meteo's forecast service.
- **Replay of the 2025 test year:** shows the forecasts the model made for 2025 next to what actually happened.

It also shows SHAP and attention explanations for each barangay and a priority layer (hazard × exposure × vulnerability).

## Files

| File | What it is |
|---|---|
| `app.py` | The dashboard |
| `requirements.txt` | Python packages, pinned to the versions of the Colab training run |
| `alstm_encoder.keras` | Trained 14-day Attention-LSTM encoder |
| `xgb_hybrid.json` | Trained hybrid XGBoost classifier |
| `meta.json` | Feature names, scaling ranges and category labels |
| `barangays.csv` | Static barangay data (LST, NDVI, population density, grid cell) |
| `barangays.geojson` | Barangay boundaries (HDX COD-AB) |
| `test_forecasts.csv.gz` | 2025 forecasts and actual categories for replay mode |

## A. Deploy online (Streamlit Community Cloud, free)

1. **Make a GitHub repository.**
    1. Create a free account at github.com if you don't have one.
    2. Click **New repository**, name it, for example, `heatrisk-dashboard`, and set it to **Public**.
    3. Click **uploading an existing file** and drag in **all the files in this folder**.
    4. Click **Commit changes**.
2. **Deploy the app.**
    1. Go to share.streamlit.io and sign in with the same GitHub account.
    2. Click **Create app**, then **Deploy a public app from GitHub**.
    3. Choose the repository, branch `main`, and main file `app.py`.
    4. Optionally, set a custom address such as `heatrisk-mm`.
    5. Open **Advanced settings** and set **Python version 3.13**. This must match the training run.
    6. Click **Deploy**. The first build installs TensorFlow and takes about 5–10 minutes.
3. **Check it works.** Open the link, try both modes, and click a barangay.
4. **Before the defense:** free apps go to sleep after a few days without visitors. Open the link the morning of the defense and click **"Yes, get this app back up"** if it is asleep.

If the build fails, open **Manage app → logs** at the bottom right of the app page. Copy the last red error lines and send them to whoever is helping you. The most likely cause is a package version that isn't available for the chosen Python version.

## B. Run on your own laptop (offline backup)

1. **Install Python 3.13** from python.org. On Windows, tick **"Add Python to PATH"** during installation.
2. **Open a terminal in this folder.** On Windows, open the folder in File Explorer, type `cmd` in the address bar and press Enter.
3. **Run these commands once:**

   ```
   python -m venv venv
   venv\Scripts\activate          (Windows)   |   source venv/bin/activate   (Mac/Linux)
   pip install -r requirements.txt
   ```

4. **Start the dashboard.** Do this every time you want to use it:

   ```
   venv\Scripts\activate          (Windows)   |   source venv/bin/activate   (Mac/Linux)
   streamlit run app.py
   ```

   The dashboard opens in your browser at http://localhost:8501.

Replay mode works without internet. Live mode needs internet, because it downloads the forecast from Open-Meteo.

## Notes

- The live forecasts use Open-Meteo's forecast model. The model was trained on ERA5 reanalysis, so live forecasts are indicative.
- Categories are based on the daily-mean heat index. They are conservative compared with PAGASA's afternoon-peak advisories, and they are not official warnings.
- The data sources are Open-Meteo / ERA5 (Copernicus), Landsat 8/9 and Sentinel-2 via Google Earth Engine, PSA 2020 CPH, and HDX COD-AB boundaries (CC BY-IGO).
