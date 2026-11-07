"""Metro Manila Barangay Heat Hazard Forecast - Streamlit prototype (Chapter III, Section J).

Run locally:  pip install -r requirements.txt  then  streamlit run app.py
Deploy:       see README.md (Streamlit Community Cloud).
"""
import json, os, time
import numpy as np, pandas as pd, requests, streamlit as st
import plotly.express as px, plotly.graph_objects as go

HERE = os.path.dirname(os.path.abspath(__file__))
st.set_page_config(page_title='Metro Manila Heat Hazard Forecast', layout='wide', initial_sidebar_state='collapsed')

INK, MUTED, ACCENT, PAPER, PANEL = '#14213D', '#5A6577', '#B8441A', '#FBFBF8', '#F3EFE6'
FONT = 'Inter, Arial, sans-serif'
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=Space+Grotesk:wght@500;600&display=swap" rel="stylesheet">
<style>
html, body, [class*="css"], .stMarkdown, .stText, .stCaption, label, p, li { font-family: 'Inter', Arial, sans-serif; }
h1, h2, h3, h4 { font-family: 'Space Grotesk', 'Inter', Arial, sans-serif !important; letter-spacing: -0.01em; color: #14213D; }
.block-container { padding-top: 2.2rem; padding-bottom: 2rem; max-width: 1400px; }
.eyebrow { font-size: 0.78rem; letter-spacing: 0.12em; text-transform: uppercase; color: #B8441A; font-weight: 600; margin-bottom: 0.2rem; }
.subtitle { color: #5A6577; font-size: 0.98rem; margin-top: -0.4rem; margin-bottom: 1.2rem; max-width: 60rem; }
[data-testid="stMetric"] { background: #F3EFE6; border-radius: 10px; padding: 0.8rem 1rem; }
[data-testid="stMetricLabel"] p { color: #5A6577; font-size: 0.82rem; }
[data-testid="stMetricValue"] { font-family: 'Space Grotesk', 'Inter', sans-serif; color: #14213D; }
.foot { color: #5A6577; font-size: 0.8rem; border-top: 1px solid #E3DED3; padding-top: 0.8rem; margin-top: 1.5rem; }
.probrow { display: flex; align-items: center; gap: 0.6rem; margin: 0.25rem 0; font-size: 0.9rem; color: #14213D; }
.probrow .lab { width: 11.5rem; }
.probrow .bar { height: 0.7rem; border-radius: 3px; }
.probrow .val { color: #5A6577; font-variant-numeric: tabular-nums; }
#MainMenu, footer { visibility: hidden; }
</style>""", unsafe_allow_html=True)

# ----------------------------------------------------------------- loading
@st.cache_resource
def load_models():
    import tensorflow as tf, xgboost as xgb
    from tensorflow import keras
    from tensorflow.keras import layers

    _register = getattr(getattr(keras, 'saving', None), 'register_keras_serializable', None) or keras.utils.register_keras_serializable

    @_register(package='heatrisk')
    class AdditiveAttention(layers.Layer):
        def __init__(self, units=64, **kw):
            super().__init__(**kw); self.units = units
            self.W1 = layers.Dense(units, use_bias=False); self.W2 = layers.Dense(units); self.v = layers.Dense(1, use_bias=False)
        def call(self, h):
            s = h[:, -1, :]; e = self.v(tf.tanh(self.W1(h) + tf.expand_dims(self.W2(s), 1)))
            a = tf.nn.softmax(e, axis=1); return tf.reduce_sum(a * h, axis=1), tf.squeeze(a, -1)
        def get_config(self): return {**super().get_config(), 'units': self.units}

    full = keras.models.load_model(os.path.join(HERE, 'alstm_encoder.keras'), custom_objects={'AdditiveAttention': AdditiveAttention})
    enc = keras.Model(full.input, full.get_layer('embedding').output)
    att = keras.Model(full.input, full.get_layer('attention').output[1])
    clf = xgb.XGBClassifier(); clf.load_model(os.path.join(HERE, 'xgb_hybrid.json'))
    return enc, att, clf

@st.cache_data
def load_static():
    meta = json.load(open(os.path.join(HERE, 'meta.json')))
    brgy = pd.read_csv(os.path.join(HERE, 'barangays.csv'), index_col=0)
    geo = json.load(open(os.path.join(HERE, 'barangays.geojson'))) if os.path.exists(os.path.join(HERE, 'barangays.geojson')) else None
    f = os.path.join(HERE, 'test_forecasts.csv.gz')
    if not os.path.exists(f): f = os.path.join(HERE, 'test_forecasts.csv')
    test = pd.read_csv(f, parse_dates=['issue_date', 'target_date'],
                       dtype={'actual': 'int8', 'predicted': 'int8', 'p0': 'float32', 'p1': 'float32', 'p2': 'float32'})
    return meta, brgy, geo, test

meta, brgy, geo, test = load_static()
CLASSES, COLORS = meta['class_names'], meta['class_colors']; K = len(CLASSES); L = meta['lag_window']
COLORS = ['#7FAE8C', '#E3B341', '#C9541F'][:K]          # muted PAGASA-order palette: cool green, amber, burnt orange
CMAP = dict(zip(CLASSES, COLORS))

def nws_heat_index_c(t_c, rh):
    T = np.asarray(t_c, float) * 9/5 + 32; RH = np.asarray(rh, float)
    simple = 0.5 * (T + 61.0 + (T - 68.0) * 1.2 + RH * 0.094)
    hi = -42.379 + 2.04901523*T + 10.14333127*RH - .22475541*T*RH - .00683783*T*T - .05481717*RH*RH \
         + .00122874*T*T*RH + .00085282*T*RH*RH - .00000199*T*T*RH*RH
    adj1 = ((13 - RH) / 4) * np.sqrt(np.clip((17 - np.abs(T - 95.)) / 17, 0, None))
    hi = np.where((RH < 13) & (T >= 80) & (T <= 112), hi - adj1, hi)
    hi = np.where((RH > 85) & (T >= 80) & (T <= 87), hi + ((RH - 85) / 10) * ((87 - T) / 5), hi)
    return (np.where((simple + T) / 2 < 80, simple, hi) - 32) * 5/9

# ----------------------------------------------------------- live weather
DAILY = ['temperature_2m_mean', 'relative_humidity_2m_mean', 'wind_speed_10m_mean', 'shortwave_radiation_sum']
HOURLY = ['temperature_2m', 'relative_humidity_2m', 'wind_speed_10m', 'shortwave_radiation']

def _get(params):
    for attempt in range(4):
        r = requests.get('https://api.open-meteo.com/v1/forecast', params=params, timeout=60)
        if r.status_code == 200:
            j = r.json(); return j if isinstance(j, list) else [j]
        if r.status_code == 429: time.sleep(20 * (attempt + 1)); continue
        raise RuntimeError(f'Open-Meteo error {r.status_code}: {r.text[:200]}')
    raise RuntimeError('Open-Meteo rate limit reached; try again later.')

@st.cache_data(ttl=6 * 3600, show_spinner='Downloading the last 14 days and the next 3 days from Open-Meteo...')
def fetch_weather(lats, lons, past_days):
    """Daily means for each barangay point: past `past_days` days + today + 2 forecast days."""
    out = []
    base = dict(past_days=past_days, forecast_days=3, timezone='Asia/Manila', wind_speed_unit='ms')
    for i in range(0, len(lats), 100):
        la, lo = ','.join(f'{x:.5f}' for x in lats[i:i+100]), ','.join(f'{x:.5f}' for x in lons[i:i+100])
        try:
            res = _get({**base, 'latitude': la, 'longitude': lo, 'daily': ','.join(DAILY)})
            for r in res:
                d = r['daily']
                out.append(np.stack([d['temperature_2m_mean'], d['relative_humidity_2m_mean'], d['wind_speed_10m_mean'], d['shortwave_radiation_sum']], 1))
                days = d['time']
        except (KeyError, RuntimeError):   # fallback: aggregate hourly values to daily means / sums
            res = _get({**base, 'latitude': la, 'longitude': lo, 'hourly': ','.join(HOURLY)})
            for r in res:
                h = pd.DataFrame(r['hourly']); h['day'] = pd.to_datetime(h['time']).dt.date
                g = h.groupby('day').agg(t=('temperature_2m', 'mean'), rh=('relative_humidity_2m', 'mean'), ws=('wind_speed_10m', 'mean'),
                                         sw=('shortwave_radiation', lambda s: s.sum() * 3600 / 1e6))
                out.append(g[['t', 'rh', 'ws', 'sw']].values); days = [str(x) for x in g.index]
        if i + 100 < len(lats): time.sleep(1)
    arr = np.stack(out).astype(float)                                    # (n_brgy, n_days, 4)
    hi = nws_heat_index_c(arr[:, :, 0], arr[:, :, 1])[:, :, None]
    return np.concatenate([arr, hi], 2), days

def forecast_live():
    enc, att, clf = load_models()
    # Barangays close together share the same weather-model grid cell, so request each ~2.5 km cell once
    # (about 90 points instead of 1,710: one request, no rate limit) and give every barangay its cell's values.
    cell = pd.DataFrame({'la': (brgy.lat.values / 0.025).round() * 0.025, 'lo': (brgy.lon.values / 0.025).round() * 0.025})
    uniq, inv = np.unique(cell.values, axis=0, return_inverse=True)
    Wu, days = fetch_weather(tuple(uniq[:, 0]), tuple(uniq[:, 1]), L)
    W = Wu[np.ravel(inv)]
    if np.isnan(W).any():
        W = pd.DataFrame(W.reshape(-1, 5)).ffill().bfill().values.reshape(W.shape)
    mn, mx = np.array(meta['met_min']), np.array(meta['met_max'])
    Ws = ((W - mn) / (mx - mn)).astype(np.float32)
    S = ((brgy[meta['spatial_features']].values - np.array(meta['spa_min'])) / (np.array(meta['spa_max']) - np.array(meta['spa_min']))).astype(np.float32)
    today = len(days) - 3                                                # index of today in the returned series
    results = {}
    for h in (1, 2, 3):                                                  # target = today + h ; window ends at today + h - 1
        end = today + h - 1; X = Ws[:, end - L + 1: end + 1]
        E = enc.predict(X, verbose=0, batch_size=4096); A = att.predict(X, verbose=0, batch_size=4096)
        V = np.hstack([E, Ws[:, end], S]); P = clf.predict_proba(V, iteration_range=(0, meta['n_trees']))   # embedding + today's weather + spatial
        results[h] = dict(target=pd.Timestamp(days[today]) + pd.Timedelta(days=h), P=P, V=V, A=A, HI=W[:, end - L + 1: end + 1, 4],
                          uses_forecast_weather=h > 1)
    return results

# ----------------------------------------------------------- helpers
def vulnerability():
    """Descriptive exposure x vulnerability index (percentile ranks; no thresholds are learned)."""
    r = pd.DataFrame({'heat': brgy.lst_c.rank(pct=True), 'green': 1 - brgy.ndvi.rank(pct=True), 'exposure': brgy.pop_density_per_sqkm.rank(pct=True)})
    return r.mean(1)

def shap_groups(clf, v, k):
    import xgboost as xgb
    c = clf.get_booster().predict(xgb.DMatrix(v[None, :]), pred_contribs=True, iteration_range=(0, meta['n_trees']))
    c = c[0, k] if c.ndim == 3 else c[0]
    fn = meta['feature_names']; emb = [i for i, f in enumerate(fn) if f.startswith('Temporal')]
    g = {'Weather history (temporal embedding)': c[emb].sum()}
    g.update({f: c[i] for i, f in enumerate(fn) if not f.startswith('Temporal')})
    return pd.Series(g).sort_values()

def draw_map(df, value_col, title):
    kw = dict(data_frame=df, geojson=geo, locations='brgy_pcode', featureidkey='properties.adm4_pcode', color=value_col,
              color_discrete_map=CMAP, category_orders={value_col: CLASSES}, hover_name='brgy_name',
              hover_data={'city_name': True, 'brgy_pcode': False, 'p_top': ':.2f'}, center={'lat': 14.58, 'lon': 121.03}, zoom=9.7, opacity=0.85)
    fig = px.choropleth_map(**kw, map_style='carto-positron') if hasattr(px, 'choropleth_map') else px.choropleth_mapbox(**kw, mapbox_style='carto-positron')
    fig.update_traces(marker_line_width=0.3, marker_line_color='rgba(255,255,255,0.6)')
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), height=600, font=dict(family=FONT, color=INK), legend_title_text='',
                      legend=dict(orientation='h', yanchor='bottom', y=0.01, xanchor='left', x=0.01, bgcolor='rgba(251,251,248,0.9)'))
    return fig

# ----------------------------------------------------------- page
st.markdown('<div class="eyebrow">Barangay heat hazard forecast · Metro Manila</div>', unsafe_allow_html=True)
st.title('Which barangays face hazardous heat tomorrow?')
st.markdown('<div class="subtitle">Next-day DOST-PAGASA heat index category for 1,710 barangays, from a hybrid Attention-LSTM and XGBoost model, '
            'with the reasons behind each forecast. Categories use the daily-mean heat index, which runs lower than PAGASA\'s afternoon peak.</div>',
            unsafe_allow_html=True)

ctl1, ctl2, ctl3 = st.columns([1.3, 1.2, 1])
with ctl1:
    mode = st.radio('View', ['Live forecast', f"Replay {meta['test_year']}"], horizontal=True)
with ctl3:
    show_priority = st.toggle('Outline priority barangays', help='Top forecast category and the top 25% on a combined index of '
                              'surface temperature (LST), low vegetation (NDVI) and population density.')

if mode.startswith('Live'):
    try:
        R = forecast_live()
    except Exception as e:
        st.error(f'Live forecast unavailable ({e}). Switch to the replay view.'); st.stop()
    with ctl2:
        h = st.radio('Forecast for', [1, 2, 3], horizontal=True,
                     format_func=lambda x: f"{R[x]['target']:%a %d %b}")
    r = R[h]; P = r['P']
    df = brgy.rename_axis('brgy_pcode').reset_index()
    df['pred'] = P.argmax(1); df['p_top'] = P[:, K - 1]
    for j in range(K): df[f'p{j}'] = P[:, j]
    target_label = f"{r['target']:%A, %d %B %Y}"
    map_col = 'Forecast class'
else:
    days = sorted(test.target_date.dt.date.unique())
    with ctl2:
        day = st.selectbox('Day', days, index=int(np.argmax(test.groupby('target_date').predicted.apply(lambda s: (s == K - 1).sum()).values)),
                           format_func=lambda d: f'{d:%a %d %b %Y}')
    d = test[test.target_date.dt.date == day]
    df = d.rename(columns={'predicted': 'pred'}).copy(); df['p_top'] = d[f'p{K-1}']
    df['Actual class'] = [CLASSES[int(i)] for i in df.actual]
    r = None
    target_label = f'{day:%A, %d %B %Y}'
    map_col = 'Forecast class'

df['Forecast class'] = [CLASSES[int(i)] for i in df.pred]
vul = vulnerability().reindex(df.brgy_pcode).values
df['priority'] = (df.pred >= K - 1) & (vul >= 0.75)
counts = df['Forecast class'].value_counts().reindex(CLASSES, fill_value=0)

m1, m2, m3, m4 = st.columns(4)
m1.metric('Forecast date', target_label.split(', ', 1)[1] if ', ' in target_label else target_label)
m2.metric(f'{CLASSES[-1]}', f'{counts.iloc[-1]:,} barangays')
m3.metric('Priority barangays', f'{int(df.priority.sum()):,}')
if r is None:
    m4.metric('Forecast matched actual', f'{(df.actual == df.pred).mean():.0%}')
else:
    m4.metric(CLASSES[1], f'{counts.iloc[1]:,} barangays')

st.write('')
c1, c2 = st.columns([1.55, 1], gap='large')
with c1:
    if r is None:
        map_col = st.radio('Map shows', ['Forecast class', 'Actual class'], horizontal=True, label_visibility='collapsed')
    if geo is None:
        st.warning('barangays.geojson not found in the dashboard folder; map unavailable.')
    else:
        fig = draw_map(df, map_col, '')
        if show_priority and df.priority.any():
            pr = df[df.priority]
            fig.add_trace(go.Choroplethmap(geojson=geo, locations=pr.brgy_pcode, featureidkey='properties.adm4_pcode', z=np.ones(len(pr)),
                                           colorscale=[[0, 'rgba(0,0,0,0)'], [1, 'rgba(0,0,0,0)']], marker_line_color=INK,
                                           marker_line_width=2, showscale=False, hoverinfo='skip') if hasattr(go, 'Choroplethmap') else
                          go.Choroplethmapbox(geojson=geo, locations=pr.brgy_pcode, featureidkey='properties.adm4_pcode', z=np.ones(len(pr)),
                                              colorscale=[[0, 'rgba(0,0,0,0)'], [1, 'rgba(0,0,0,0)']], marker_line_color=INK,
                                              marker_line_width=2, showscale=False, hoverinfo='skip'))
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
    if r is not None and h > 1:
        st.caption('Days 2 and 3 use forecast weather inside the 14-day window, so uncertainty grows with lead time.')

with c2:
    names = (df.brgy_name + ', ' + df.city_name).values
    default = int(np.argmax(df.p_top.values))
    i = st.selectbox('Barangay', range(len(df)), index=default, format_func=lambda j: names[j])
    row = df.iloc[i]
    st.subheader(str(row.brgy_name))
    st.caption(str(row.city_name) + ('  ·  priority barangay' if bool(row.priority) else ''))
    line = f"Forecast: **{CLASSES[int(row.pred)]}**"
    if 'actual' in row: line += f"  ·  Actual: **{CLASSES[int(row.actual)]}**"
    st.markdown(line)
    bars = ''.join(f'<div class="probrow"><span class="lab">{c}</span><span class="bar" style="width:{max(2, row[f"p{j}"] * 160):.0f}px;'
                   f'background:{COLORS[j]}"></span><span class="val">{row[f"p{j}"]:.2f}</span></div>' for j, c in enumerate(CLASSES))
    st.markdown(bars, unsafe_allow_html=True)
    if r is not None:
        _, _, clf = load_models(); k = int(row.pred)
        s2 = shap_groups(clf, r['V'][i], k)
        f2 = go.Figure(go.Bar(x=s2.values, y=s2.index, orientation='h', marker_color=[ACCENT if v > 0 else '#2A5B9E' for v in s2.values]))
        f2.update_layout(title=dict(text=f'What pushed the forecast toward {CLASSES[k]}', font=dict(size=14)), height=290,
                         margin=dict(l=0, r=0, t=36, b=0), font=dict(family=FONT, color=INK, size=12), plot_bgcolor='rgba(0,0,0,0)',
                         paper_bgcolor='rgba(0,0,0,0)', xaxis=dict(title='SHAP contribution (log-odds)', gridcolor='#E3DED3', zerolinecolor=MUTED))
        st.plotly_chart(f2, use_container_width=True, config={'displayModeBar': False})
        lbl = [f't−{L-1-j}' if j < L - 1 else 't' for j in range(L)]
        f3 = go.Figure()
        f3.add_trace(go.Bar(x=lbl, y=r['A'][i], name='Attention weight', marker_color='rgba(184,68,26,0.30)', yaxis='y2'))
        f3.add_trace(go.Scatter(x=lbl, y=r['HI'][i], name='Heat index (°C)', line=dict(color=INK, width=2)))
        f3.update_layout(title=dict(text='Last 14 days: heat index and attention', font=dict(size=14)), height=280,
                         margin=dict(l=0, r=0, t=36, b=0), font=dict(family=FONT, color=INK, size=12), plot_bgcolor='rgba(0,0,0,0)',
                         paper_bgcolor='rgba(0,0,0,0)', yaxis=dict(title='°C', gridcolor='#E3DED3'),
                         yaxis2=dict(overlaying='y', side='right', showgrid=False, title='attention'),
                         legend=dict(orientation='h', y=-0.25))
        st.plotly_chart(f3, use_container_width=True, config={'displayModeBar': False})
    else:
        st.caption('The SHAP and attention panels appear in the live view.')

st.markdown(f"""<div class="foot">Weather: Open-Meteo (live forecast model; trained on ERA5 reanalysis). Surface temperature and vegetation: Landsat 8/9 and Sentinel-2
monthly composites as of {meta['spatial_as_of']}. Population: PSA 2020 Census. Boundaries: HDX COD-AB (CC BY-IGO).
A research prototype that complements, and does not replace, official DOST-PAGASA heat advisories.</div>""", unsafe_allow_html=True)
