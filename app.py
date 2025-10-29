import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import hashlib, io
from datetime import datetime, timedelta

st.set_page_config(
    page_title="Subsea Pipeline Dashboard (Plotly, Reactive)",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.title("Subsea Pipeline Dashboard")

SEGMENT_STRETCH = 2.0  # spacing on fallback path

# ================== UNIFIED THRESHOLDS + COLORS (Book-based) ==================
THRESHOLDS_FIXED = {
    # Risk Indices (normalized 0–1)
    "Hydrate_Risk_Index": (0.33, 0.66),
    "Wax_Deposition_Risk_Index": (0.33, 0.66),
    "Corrosion_Risk_Index": (0.33, 0.66),

    # Operating Parameters
    "Temperature_degC": (39, 70),  # Warn ≤39, Normal 40–70, Critical ≤10 or ≥80
    "Pressure_MPa": (10, 13),      # Warn >10–13, Critical ≥13.5
    "Flow_Rate_kg_s": (60, 100),   # Warn 60–99, Critical ≤50
}

COLOR_MAP = {"Low": "#16a34a", "Medium": "#facc15", "High": "#dc2626"}

def categorize_value(col, val):
    if col not in THRESHOLDS_FIXED:
        return "Low"
    low, high = THRESHOLDS_FIXED[col]
    try:
        v = float(val)
    except:
        return "Low"

    if "Risk_Index" in col:
        if v <= low: return "Low"
        elif v <= high: return "Medium"
        else: return "High"
    elif col == "Temperature_degC":
        if v <= 10 or v >= 80: return "High"
        elif v <= low: return "Medium"
        else: return "Low"
    elif col == "Pressure_MPa":
        if v >= 13.5: return "High"
        elif v > low: return "Medium"
        else: return "Low"
    elif col == "Flow_Rate_kg_s":
        if v <= 50: return "High"
        elif v < high: return "Medium"
        else: return "Low"
    return "Low"

# ================== REQUIRED COLUMNS & LABELS ==================
REQUIRED_COLUMNS = [
    "Pipe_Segment_ID",
    "Cumulative_Pipe_Length_km",
    "Temperature_degC",
    "Pressure_MPa",
    "Flow_Rate_kg_s",
    "Hydrate_Risk_Index",
    "Wax_Deposition_Risk_Index",
    "Corrosion_Risk_Index",
    "Flow_Assurance_Status"
]

COLUMN_LABELS = {
    "Pipe_Segment_ID": "Pipe Segment",
    "Cumulative_Pipe_Length_km": "Cumulative Pipe Length",
    "Temperature_degC": "Temperature",
    "Pressure_MPa": "Pressure",
    "Flow_Rate_kg_s": "Flow Rate",
    "Hydrate_Risk_Index": "Hydrate Risk Index",
    "Wax_Deposition_Risk_Index": "Wax Deposition Risk Index",
    "Corrosion_Risk_Index": "Corrosion Risk Index",
    "Flow_Assurance_Status": "Flow Assurance Status",
    "Timestamp": "Timestamp",
}

def pretty_name(col: str) -> str:
    return COLUMN_LABELS.get(col, col.replace("_", " "))

# ================== SIDEBAR ==================
def _clear_on_upload():
    try:
        st.cache_data.clear()
    except Exception:
        pass
    for k in ["df", "file_sig"]:
        if k in st.session_state:
            st.session_state.pop(k)

uploaded = st.sidebar.file_uploader(
    "Upload pipeline CSV",
    type=["csv"],
    key="uploader_unique",
    on_change=_clear_on_upload,
    help="Re-upload anytime; the 3D view and tables will auto-update."
)

# ================== HELPERS ==================
def _orthonormal_basis(t):
    t = t / (np.linalg.norm(t) + 1e-12)
    a = np.array([0., 0., 1.])
    if abs(np.dot(t, a)) > 0.9:
        a = np.array([0., 1., 0.])
    n1 = np.cross(t, a); n1 /= (np.linalg.norm(n1) + 1e-12)
    n2 = np.cross(t, n1); n2 /= (np.linalg.norm(n2) + 1e-12)
    return n1, n2

def polyline_to_tube_mesh(x, y, z, intensity_vals, radius=0.1, sides=16, hover_labels=None):
    x = np.asarray(x, float); y = np.asarray(y, float); z = np.asarray(z, float)
    n = len(x)
    if n < 2:
        return go.Mesh3d(x=x, y=y, z=z)

    T = np.zeros((n, 3))
    pts = np.column_stack([x, y, z])
    T[0] = pts[1] - pts[0]
    T[-1] = pts[-1] - pts[-2]
    T[1:-1] = pts[2:] - pts[:-2]

    theta = np.linspace(0, 2*np.pi, sides, endpoint=False)
    cos_t = np.cos(theta); sin_t = np.sin(theta)

    verts, colors, customdata = [], [], []
    for i in range(n):
        n1, n2 = _orthonormal_basis(T[i])
        for k in range(sides):
            p = pts[i] + radius * (cos_t[k]*n1 + sin_t[k]*n2)
            verts.append(p)
            colors.append(float(intensity_vals[i]))
            customdata.append(hover_labels[i] if (hover_labels is not None and i < len(hover_labels)) else "")

    verts = np.array(verts)
    vx, vy, vz = verts[:,0], verts[:,1], verts[:,2]

    I, J, K = [], [], []
    for i in range(n-1):
        ring0 = i * sides
        ring1 = (i+1) * sides
        for k in range(sides):
            a0 = ring0 + k
            a1 = ring0 + (k+1) % sides
            b0 = ring1 + k
            b1 = ring1 + (k+1) % sides
            I += [a0, a1]; J += [b0, b0]; K += [a1, b1]

    discrete_colorscale = [
        [0.00, "#16a34a"], [0.33, "#16a34a"],
        [0.34, "#facc15"], [0.66, "#facc15"],
        [0.67, "#dc2626"], [1.00, "#dc2626"],
    ]

    mesh = go.Mesh3d(
        x=vx, y=vy, z=vz, i=I, j=J, k=K,
        intensity=colors, colorscale=discrete_colorscale,
        cmin=0, cmax=2, showscale=False,
        customdata=np.array(customdata, dtype=object),
        hovertemplate="%{customdata}<extra></extra>",
        lighting=dict(ambient=0.25, diffuse=0.75, specular=0.5, roughness=0.6, fresnel=0.1),
        lightposition=dict(x=0.5, y=0.5, z=1.0),
        name="", showlegend=False, opacity=1.0
    )
    return mesh

def file_signature(b: bytes) -> str:
    import hashlib as _hashlib
    return _hashlib.md5(b).hexdigest()

@st.cache_data(show_spinner=False)
def read_csv_bytes(b: bytes) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(b))

def validate_schema(df: pd.DataFrame):
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    return missing

# ---------- Shared, synced time slider helper ----------
def synced_time_slider(options, page_key, label="Select Timestamp"):
    if not options:
        return None

    options = list(pd.to_datetime(pd.Index(options)))

    if "selected_time" not in st.session_state or st.session_state["selected_time"] not in options:
        st.session_state["selected_time"] = options[0]

    if page_key not in st.session_state:
        st.session_state[page_key] = st.session_state["selected_time"]

    def _on_change():
        st.session_state["selected_time"] = st.session_state[page_key]
        for k in ("selected_time_main", "selected_time_sensor", "selected_time_risk"):
            if k != page_key and k in st.session_state:
                st.session_state[k] = st.session_state["selected_time"]

    st.select_slider(
        label,
        options=options,
        key=page_key,
        format_func=lambda x: x.strftime("%Y-%m-%d %H:%M:%S"),
        on_change=_on_change,
        help="This slider is synced across all pages."
    )
    return st.session_state["selected_time"]

# ---------- Generic HTML table renderer ----------
def html_table(df: pd.DataFrame, *, formats: dict | None = None, newline_cols: set | None = None, per_cell_style_fn=None) -> str:
    if df is None or df.empty:
        return "<div>No data</div>"

    cols = list(df.columns)
    html = ['<table style="border-collapse:collapse; width:100%;">', "<thead><tr>"]
    for c in cols:
        html.append(f'<th style="text-align:left; padding:8px; border-bottom:1px solid #444;">{c}</th>')
    html.append("</tr></thead><tbody>")

    for i, row in df.iterrows():
        html.append("<tr>")
        for c in cols:
            v = row[c]
            if formats and c in formats:
                fmt = formats[c]
                try:
                    if isinstance(fmt, str):
                        v = fmt.format(v)
                    else:
                        v = fmt(v)
                except Exception:
                    pass
            text = "" if pd.isna(v) else str(v)
            if newline_cols and c in newline_cols:
                text = text.replace("\n", "<br>")
                td_style_extra = " white-space:pre-line;"
            else:
                td_style_extra = ""
            cell_css = ""
            if per_cell_style_fn is not None:
                try:
                    css_dict = per_cell_style_fn(i, c, row[c])
                    if isinstance(css_dict, dict) and css_dict:
                        cell_css = " " + " ".join(f"{k}:{v};" for k, v in css_dict.items())
                except Exception:
                    pass
            html.append(f'<td style="padding:8px;{td_style_extra}{cell_css}">{text}</td>')
        html.append("</tr>")
    html.append("</tbody></table>")
    return "".join(html)

# ================== DATA GATE ==================
if uploaded is None and "df" not in st.session_state:
    st.info("📄 Upload a CSV to continue. (Use the template headers.)")
    st.stop()

if uploaded is not None:
    b = uploaded.getvalue()
    sig = file_signature(b)
    if st.session_state.get("file_sig") != sig:
        try:
            df_new = read_csv_bytes(b)
        except Exception as e:
            st.error(f"Could not read CSV: {e}")
            st.stop()
        missing = validate_schema(df_new)
        if missing:
            st.error(f"❌ Missing columns: {', '.join(missing)}")
            st.caption("Please fix the headers to match the required schema.")
            st.stop()
        st.session_state["file_sig"] = sig
        st.session_state["df"] = df_new

df = st.session_state.get("df")
if df is None:
    st.info("No dataset in memory yet. Please upload (or re-upload) a CSV to render the 3D pipeline.")
    st.stop()

# ================== 3D PIPELINE FIGURE ==================
def build_pipeline_fig(df: pd.DataFrame, color_col: str = "Corrosion_Risk_Index", style: str = "Tube", tube_radius: float = 0.6, tube_sides: int = 24):
    if df is None or df.empty:
        return go.Figure()

    df = df.sort_values("Cumulative_Pipe_Length_km", ascending=True).copy()
    if "Pipe_Segment_ID" not in df.columns:
        df["Pipe_Segment_ID"] = [f"S{i+1}" for i in range(len(df))]

    x = df["Cumulative_Pipe_Length_km"].values * SEGMENT_STRETCH
    y = np.zeros_like(x)
    z = np.zeros_like(x)

    col = color_col if (color_col and color_col in df.columns) else None
    if col is not None:
        cat_idx = []
        for v in df[col].values:
            cat = categorize_value(col, v)
            cat_idx.append({"Low": 0, "Medium": 1, "High": 2}.get(cat, 0))
    else:
        cat_idx = [0]*len(df)

    hover_lbls = []
    for _, row in df.iterrows():
        seg = row.get("Pipe_Segment_ID", "")
        tt = []
        for k in ["Temperature_degC", "Pressure_MPa", "Flow_Rate_kg_s", "Hydrate_Risk_Index", "Wax_Deposition_Risk_Index", "Corrosion_Risk_Index"]:
            if k in df.columns:
                v = row[k]
                tt.append(f"{COLUMN_LABELS.get(k,k)}: {v}")
        hover_lbls.append(f"<b>Segment {seg}</b><br>" + "<br>".join(tt))

    if style == "Tube":
        mesh = polyline_to_tube_mesh(x, y, z, cat_idx, radius=tube_radius, sides=tube_sides, hover_labels=hover_lbls)
        fig = go.Figure(data=[mesh])
    else:
        fig = go.Figure(data=[go.Scatter3d(x=x, y=y, z=z, mode="lines", line=dict(width=12, color="#16a34a"))])

    fig.update_layout(
        scene=dict(
            xaxis_title="Distance (km)",
            yaxis_title="",
            zaxis_title="",
            xaxis=dict(backgroundcolor="#111", gridcolor="#333", showbackground=True),
            yaxis=dict(backgroundcolor="#111", gridcolor="#333", showbackground=True),
            zaxis=dict(backgroundcolor="#111", gridcolor="#333", showbackground=True),
            aspectmode="data",
        ),
        margin=dict(l=0, r=0, t=0, b=0),
        height=540,
        paper_bgcolor="#0b0f1a",
        plot_bgcolor="#0b0f1a",
    )
    return fig

# ================== MAIN PAGE ==================
def render_main_page(df: pd.DataFrame):
    if df is not None and not df.empty:
        if "Timestamp" in df.columns:
            ts_series = pd.to_datetime(df["Timestamp"], errors="coerce")
            valid_mask = ts_series.notna()
            if valid_mask.any():
                df2 = df.loc[valid_mask].copy()
                unique_times = sorted(ts_series.loc[valid_mask].unique())
                selected_time = synced_time_slider(unique_times, "selected_time_main")
                df_time = df2[pd.to_datetime(df2["Timestamp"]) == selected_time]
            else:
                st.warning("⚠️ All values in 'Timestamp' are invalid. Showing full data.")
                df_time = df
        else:
            st.warning("⚠️ No 'Timestamp' column found in the dataset. Showing full data.")
            df_time = df

        color_choice = st.selectbox(
            "Sensor",
            options=[
                "Corrosion_Risk_Index",
                "Hydrate_Risk_Index",
                "Wax_Deposition_Risk_Index",
                "Temperature_degC",
                "Pressure_MPa",
                "Flow_Rate_kg_s",
            ],
            index=0,
            format_func=lambda x: COLUMN_LABELS.get(x, x),
            help="Choose which column to map to the pipeline color."
        )

        df_to_plot = df_time if "df_time" in locals() else df
        fig = build_pipeline_fig(df_to_plot, color_col=color_choice, style="Tube", tube_radius=1.0, tube_sides=40)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No data loaded yet. Please upload a valid CSV.")

# ================== Top changes (kept but not shown) ==================
def calculate_top_changes(df: pd.DataFrame, selected_time: pd.Timestamp | None, top_n=5, window_minutes=30):
    if df is None or df.empty or "Timestamp" not in df.columns:
        return []

    dfx = df.copy()
    dfx["Timestamp"] = pd.to_datetime(dfx["Timestamp"], errors="coerce")
    dfx = dfx.dropna(subset=["Timestamp"])
    if dfx.empty:
        return []

    if selected_time is None:
        selected_time = dfx["Timestamp"].max()
    else:
        selected_time = pd.to_datetime(selected_time)

    t0 = selected_time - timedelta(minutes=window_minutes)

    metrics = [
        ("Pressure_MPa", "Pressure"),
        ("Temperature_degC", "Temperature"),
        ("Flow_Rate_kg_s", "Flow Rate"),
        ("Hydrate_Risk_Index", "Hydrate Risk Index"),
        ("Wax_Deposition_Risk_Index", "Wax Deposition Risk Index"),
        ("Corrosion_Risk_Index", "Corrosion Risk Index"),
    ]
    metrics = [(c, n) for (c, n) in metrics if c in dfx.columns]

    out = []
    for seg, g in dfx.groupby("Pipe_Segment_ID"):
        g = g.sort_values("Timestamp")
        g_now = g[g["Timestamp"] <= selected_time]
        g_then = g[g["Timestamp"] <= t0]
        if g_now.empty or g_then.empty:
            continue
        row_now = g_now.tail(1).iloc[0]
        row_then = g_then.tail(1).iloc[0]

        for col, nice in metrics:
            v_now = pd.to_numeric(row_now.get(col, np.nan), errors="coerce")
            v_then = pd.to_numeric(row_then.get(col, np.nan), errors="coerce")
            if pd.isna(v_now) or pd.isna(v_then) or v_then == 0:
                continue
            pct = float((v_now - v_then) / v_then * 100.0)
            out.append({"Pipe_Segment_ID": seg, "Metric": nice, "ChangePct": pct})

    if not out:
        return []

    changes = pd.DataFrame(out)
    changes["Abs"] = changes["ChangePct"].abs()
    changes = changes.sort_values("Abs", ascending=False).head(top_n)

    lines = []
    for _, r in changes.iterrows():
        direction = "dropped" if r["ChangePct"] < 0 else "increased"
        lines.append(f"Segment {r['Pipe_Segment_ID']}: {r['Metric']} {direction} by {abs(r['ChangePct']):.1f}% in the last {window_minutes} minutes")
    return lines

# ================== SENSOR LIST / UNITS / THRESHOLDS (FIXED 6 SENSORS) ==================
FIXED_SENSORS = [
    "Pressure_MPa",
    "Temperature_degC",
    "Flow_Rate_kg_s",
    "Hydrate_Risk_Index",
    "Wax_Deposition_Risk_Index",
    "Corrosion_Risk_Index",
]

UNITS = {
    "Pressure_MPa": "MPa",
    "Temperature_degC": "°C",
    "Flow_Rate_kg_s": "kg/s",
    # risk indices unitless
}

# Minimum absolute deltas (more sensitive for risk indices)
MIN_ABS_DELTA = {
    "Pressure_MPa": 0.20,
    "Temperature_degC": 0.80,
    "Flow_Rate_kg_s": 3.00,
    "Hydrate_Risk_Index": 0.03,
    "Wax_Deposition_Risk_Index": 0.03,
    "Corrosion_Risk_Index": 0.03,
}
# Per-sensor std multiplier for adaptive thresholds (gentler for risk indices)
STD_MULTIPLIER_BY_SENSOR = {
    "Pressure_MPa": 1.5,
    "Temperature_degC": 1.5,
    "Flow_Rate_kg_s": 1.5,
    "Hydrate_Risk_Index": 1.0,
    "Wax_Deposition_Risk_Index": 1.0,
    "Corrosion_Risk_Index": 1.0,
}

# ================== ACTIVE WARNINGS — ONLY CHANGES THAT BECAME HIGH (shows actual reading) ==================
def generate_active_warnings(df: pd.DataFrame, selected_time: pd.Timestamp | None, window_minutes=60, max_messages=10):
    """
    Emit warnings ONLY for metrics that TRANSITIONED into High risk within the last `window_minutes`.
    Conditions per metric:
      - |delta| >= adaptive threshold (max(min_abs, multiplier * local std)),
      - category(now) == High,
      - category(then) != High.
    Output shows the actual current reading instead of category text.
    """
    if df is None or df.empty or "Timestamp" not in df.columns:
        return []

    dfx = df.copy()
    dfx["Timestamp"] = pd.to_datetime(dfx["Timestamp"], errors="coerce")
    dfx = dfx.dropna(subset=["Timestamp"])
    if dfx.empty:
        return []

    sensors = [s for s in FIXED_SENSORS if s in dfx.columns]

    if selected_time is None:
        selected_time = dfx["Timestamp"].max()
    else:
        selected_time = pd.to_datetime(selected_time)

    t0 = selected_time - timedelta(minutes=window_minutes)

    def add_roll(g):
        g = g.sort_values("Timestamp").copy()
        for f in sensors:
            s = pd.to_numeric(g[f], errors="coerce")
            g[f] = s
            g[f"{f}_rmean3"] = s.rolling(3, min_periods=1).mean()
            g[f"{f}_rstd3"]  = s.rolling(3, min_periods=1).std()
            g[f"{f}_delta1"] = s.diff()
        return g

    dfx = dfx.groupby("Pipe_Segment_ID", group_keys=False).apply(add_roll)

    messages, severities = [], []

    def delta_and_vals(g, col):
        g = g.sort_values("Timestamp")
        now = g[g["Timestamp"] <= selected_time]
        then = g[g["Timestamp"] <= t0]
        if now.empty or then.empty or col not in g.columns:
            return None, None, None
        v_now = pd.to_numeric(now.tail(1)[col].iloc[0], errors="coerce")
        v_then = pd.to_numeric(then.tail(1)[col].iloc[0], errors="coerce")
        if pd.isna(v_now) or pd.isna(v_then):
            return None, None, None
        return float(v_now - v_then), float(v_now), float(v_then)

    def local_std_threshold(g, col):
        win = g[(g["Timestamp"] >= t0) & (g["Timestamp"] <= selected_time)]
        if win.empty or col not in win.columns:
            return MIN_ABS_DELTA.get(col, 0.0)
        s = pd.to_numeric(win[col], errors="coerce")
        loc_std = float(s.std(skipna=True)) if s.notna().sum() >= 3 else 0.0
        base = MIN_ABS_DELTA.get(col, 0.0)
        mult = STD_MULTIPLIER_BY_SENSOR.get(col, 1.5)
        return max(base, mult * loc_std)

    def fmt_delta(col, d):
        return f"{abs(d):.2f} {UNITS[col]}" if col in UNITS else f"{abs(d):.2f}"

    def fmt_value(col, v):
        return f"{v:.2f} {UNITS[col]}" if col in UNITS else f"{v:.2f}"

    for seg, g in dfx.groupby("Pipe_Segment_ID"):
        g = g[g["Timestamp"] <= selected_time].copy().sort_values("Timestamp")
        if g.empty:
            continue

        seg_msgs, seg_scores = [], []

        for col in sensors:
            d_val, v_now, v_then = delta_and_vals(g, col)
            if d_val is None:
                continue

            # Transition check: then -> now category crossing into High
            now_cat = categorize_value(col, v_now)
            then_cat = categorize_value(col, v_then)
            if not (now_cat == "High" and then_cat != "High"):
                continue

            # Ensure the change is meaningful
            thr = local_std_threshold(g, col)
            if abs(d_val) < thr or thr <= 0:
                continue

            # Direction wording
            if col == "Temperature_degC":
                direction = "cooled" if d_val < 0 else "warmed"
            elif col == "Flow_Rate_kg_s":
                direction = "decreased" if d_val < 0 else "increased"
            else:
                direction = "dropped" if d_val < 0 else "increased"

            seg_msgs.append(
                f"Segment {seg}: {pretty_name(col)} {direction} by {fmt_delta(col, d_val)} in the last {window_minutes} minutes (now {fmt_value(col, v_now)})"
            )
            seg_scores.append(abs(d_val))

        # Keep top 2 transitions per segment
        if seg_msgs:
            order = np.argsort(seg_scores)[::-1]
            for idx in order[:2]:
                messages.append(seg_msgs[idx])
                severities.append(seg_scores[idx])

    if not messages:
        return []
    order = np.argsort(severities)[::-1]
    return [messages[i] for i in order[:max_messages]]

# ================== ML SENSOR TREND INSIGHTS (always 6 lines; blank if insufficient data) ==================
def generate_sensor_trend_insights(df: pd.DataFrame, selected_time: pd.Timestamp | None, trend_minutes=120):
    """
    Always emit one line per sensor (fixed set of 6) describing the trend.
    Uses robust slope vs variability on the median (across segments) series in the selected window.
    If insufficient data (e.g., at the start of the timeline), emit a blank line.
    """
    if df is None or df.empty or "Timestamp" not in df.columns:
        return []

    dfx = df.copy()
    dfx["Timestamp"] = pd.to_datetime(dfx["Timestamp"], errors="coerce")
    dfx = dfx.dropna(subset=["Timestamp"])
    if dfx.empty:
        return []

    sensors = [s for s in FIXED_SENSORS if s in dfx.columns]

    if selected_time is None:
        selected_time = dfx["Timestamp"].max()
    else:
        selected_time = pd.to_datetime(selected_time)

    t0 = selected_time - timedelta(minutes=trend_minutes)
    win = dfx[(dfx["Timestamp"] >= t0) & (dfx["Timestamp"] <= selected_time)].copy()

    def _fit_slope(ts: pd.Series, ys: pd.Series):
        x = (ts.values.astype("datetime64[s]").astype("int64") / 60.0).astype(float)
        x = x - x.min()
        y = pd.to_numeric(ys, errors="coerce").astype(float).values
        mask = np.isfinite(x) & np.isfinite(y)
        x = x[mask]; y = y[mask]
        if len(x) < 3:
            return 0.0, 0.0
        try:
            from sklearn.linear_model import LinearRegression
            lr = LinearRegression()
            lr.fit(x.reshape(-1,1), y)
            slope_per_min = lr.coef_[0]
        except Exception:
            slope_per_min = np.polyfit(x, y, 1)[0]
        y_std = float(np.std(y)) if len(y) > 1 else 0.0
        return slope_per_min * 60.0, y_std  # per hour

    agg = win.groupby("Timestamp")[sensors].median(numeric_only=True).reset_index() if (not win.empty and sensors) else pd.DataFrame()

    insights = []

    def classify(col, slope_h, std):
        if col in {"Hydrate_Risk_Index","Wax_Deposition_Risk_Index","Corrosion_Risk_Index"}:
            low, high = 0.12, 0.45
        else:
            low, high = 0.15, 0.50

        mag = (abs(slope_h) / (std + 1e-9)) if std > 0 else 0.0
        if mag < low:
            return "Stable with minor fluctuations", None
        if slope_h < 0:
            return ("Gradual decline" if mag < high else "Sharp decline"), "down"
        else:
            return ("Gradual increase" if mag < high else "Sharp increase"), "up"

    for col in FIXED_SENSORS:
        if col not in sensors:
            label = {
                "Pressure_MPa":"Pressure Trend",
                "Temperature_degC":"Temperature",
                "Flow_Rate_kg_s":"Flow Rate Trend",
                "Hydrate_Risk_Index":"Hydrate Risk",
                "Wax_Deposition_Risk_Index":"Wax Deposition Risk",
                "Corrosion_Risk_Index":"Corrosion Risk",
            }.get(col, pretty_name(col))
            insights.append(f"{label}: No data in window")
            continue

        s = agg[col] if (not agg.empty and col in agg.columns) else pd.Series(dtype=float)
        if s.notna().sum() < 3:
            # blank line instead of 'insufficient data'
            insights.append("")
        else:
            slope_h, std = _fit_slope(agg["Timestamp"], s)
            desc, _ = classify(col, slope_h, std)
            label = {
                "Pressure_MPa":"Pressure Trend",
                "Temperature_degC":"Temperature",
                "Flow_Rate_kg_s":"Flow Rate Trend",
                "Hydrate_Risk_Index":"Hydrate Risk",
                "Wax_Deposition_Risk_Index":"Wax Deposition Risk",
                "Corrosion_Risk_Index":"Corrosion Risk",
            }.get(col, pretty_name(col))
            insights.append(f"{label}: {desc}")

    return insights

# ================== RECOMMENDATIONS (dynamic, two-line) ==================
def _format_two_actions(suggestions):
    seen, out = set(), []
    for s in suggestions:
        if s and s not in seen:
            out.append(s); seen.add(s)
        if len(out) == 2:
            break
    defaults = ["Increase inspection frequency", "Review inhibition / pigging schedule"]
    for d in defaults:
        if len(out) == 2: break
        if d not in seen:
            out.append(d); seen.add(d)
    return f"1.{out[0]}\n2.{out[1]}"

def _recommend_actions_from_snapshot(row):
    def _get_val(r, col):
        if isinstance(r, pd.Series):
            return r.get(col, np.nan)
        if isinstance(r, dict):
            return r.get(col, np.nan)
        return np.nan

    P  = _get_val(row, "Pressure_MPa")
    T  = _get_val(row, "Temperature_degC")
    F  = _get_val(row, "Flow_Rate_kg_s")
    H  = _get_val(row, "Hydrate_Risk_Index")
    W  = _get_val(row, "Wax_Deposition_Risk_Index")
    C  = _get_val(row, "Corrosion_Risk_Index")

    suggestions = []

    risks = [("Corrosion", C), ("Wax", W), ("Hydrate", H)]
    risks = [(k, (v if pd.notna(v) else -1)) for k, v in risks]
    top_risk = max(risks, key=lambda kv: kv[1])[0] if risks else "Corrosion"

    if pd.notna(P) and P >= 13.5:
        suggestions += ["Verify pressure control/relief systems", "Inspect upstream choke/possible blockage"]
    if pd.notna(F) and F <= 50:
        suggestions += ["Investigate restriction or bypass", "Plan pigging / check valves & filters"]
    if pd.notna(T) and (T <= 10 or T >= 80):
        suggestions += ["Stabilize temperature / check insulation/heating", "Inject hydrate inhibitor (MEG/methanol) if needed"]

    if ((pd.notna(C) and C >= 0.66) or top_risk == "Corrosion"):
        suggestions += ["Prioritize corrosion mitigation", "Schedule ultrasonic thickness survey"]
    elif ((pd.notna(W) and W >= 0.66) or top_risk == "Wax"):
        suggestions += ["Schedule pigging to remove wax", "Optimize temperature/flow to reduce wax deposition"]
    elif ((pd.notna(H) and H >= 0.66) or top_risk == "Hydrate"):
        suggestions += ["Increase temperature / manage insulation", "Adjust hydrate inhibitor dosing"]

    return _format_two_actions(suggestions)

# ---------- quick summary counts reused ----------
def _row_max_risk(r):
    vals = []
    for k in ["Hydrate_Risk_Index", "Wax_Deposition_Risk_Index", "Corrosion_Risk_Index"]:
        try:
            vals.append(float(r[k]))
        except Exception:
            pass
    return max(vals) if vals else np.nan

def sensor_page_counts(df: pd.DataFrame):
    if df is None or df.empty:
        return 0, 0

    dfx = df.copy()
    if "Timestamp" in dfx.columns:
        dfx["Timestamp"] = pd.to_datetime(dfx["Timestamp"], errors="coerce")
        dfx = dfx.sort_values(["Pipe_Segment_ID", "Timestamp"])
        latest = dfx.groupby("Pipe_Segment_ID").tail(1).copy()
    else:
        latest = dfx.groupby("Pipe_Segment_ID").tail(1).copy()

    if latest.empty:
        return 0, 0

    latest["Max_Risk_Index"] = latest.apply(_row_max_risk, axis=1)
    denom = (0.66 - 0.33) + 1e-6
    latest["P(NextHigh)"] = np.clip((latest["Max_Risk_Index"] - 0.33) / denom, 0, 1)

    total_segments = latest["Pipe_Segment_ID"].nunique()
    predicted_high = int((latest["P(NextHigh)"] >= 0.6).sum())
    return total_segments, predicted_high

# ================== SENSOR DATA PAGE ==================
def render_sensor_data_page(df: pd.DataFrame):
    if df is None or (isinstance(df, pd.DataFrame) and df.empty):
        st.info("No dataset loaded. Please upload a CSV first.")
        return

    st.subheader("Sensor Summary Table")

    selected_time = None
    if "Timestamp" in df.columns:
        ts_all = pd.to_datetime(df["Timestamp"], errors="coerce").dropna().unique().tolist()
        ts_all = sorted(ts_all)
        selected_time = synced_time_slider(ts_all, "selected_time_sensor")

    # Build summary table
    num_cols = ["Temperature_degC", "Pressure_MPa", "Flow_Rate_kg_s", "Hydrate_Risk_Index", "Wax_Deposition_Risk_Index", "Corrosion_Risk_Index"]
    exist = [c for c in num_cols if c in df.columns]
    if not exist:
        st.info("No numeric sensor columns found to summarize.")
        return

    # Helper to append units to display name (only when available)
    def sensor_display_name(col):
        base = COLUMN_LABELS.get(col, col)
        unit = UNITS.get(col)
        return f"{base} ({unit})" if unit else base

    rows = []
    for col in exist:
        s = pd.to_numeric(df[col], errors="coerce")
        rows.append({
            "Sensor (raw)": col,
            "Sensor": sensor_display_name(col),  # <-- unit appended here
            "Min Value": float(np.nanmin(s)),
            "Max Value": float(np.nanmax(s)),
            "Average Value": float(np.nanmean(s)),
        })
    summary_df = pd.DataFrame(rows)

    # Round to 2 decimals
    summary_df_display = summary_df.copy()
    for c in ["Min Value", "Max Value", "Average Value"]:
        summary_df_display[c] = summary_df_display[c].round(2)

    visible_df = summary_df_display.drop(columns=["Sensor (raw)"])
    df_clean = visible_df.reset_index(drop=True)

    # HTML table with per-cell risk color (same logic as pipeline)
    sensor_raw_by_idx = summary_df_display["Sensor (raw)"].reset_index(drop=True)
    def _sensor_cell_style(row_idx, col_name, value):
        if col_name not in {"Min Value", "Max Value", "Average Value"}:
            return {}
        raw_col = sensor_raw_by_idx.iloc[row_idx]
        level = categorize_value(raw_col, value)
        color = COLOR_MAP.get(level, None)
        return {"background-color": color, "color": "black"} if color else {}

    sensor_html = html_table(
        df_clean,
        formats={"Min Value": "{:.2f}", "Max Value": "{:.2f}", "Average Value": "{:.2f}"},
        per_cell_style_fn=_sensor_cell_style
    )
    st.markdown(sensor_html, unsafe_allow_html=True)

    # Active Warnings (only transitions into High)
    st.subheader("Active Warnings")
    warnings = generate_active_warnings(df, selected_time=selected_time, window_minutes=60, max_messages=10)
    if not warnings:
        st.info("No new transitions into High risk in the last hour.")
    else:
        for msg in warnings:
            st.write(msg)

    # Sensor Trend Insights (ML) — always 6 lines, blanks if insufficient data
    st.subheader("Sensor Trend Insights")
    insights = generate_sensor_trend_insights(df, selected_time=selected_time, trend_minutes=120)
    if not insights:
        st.info("No clear trend signals in the selected window.")
    else:
        for line in insights:
            st.write(line)

# ================== RISK & TREND FORECAST PAGE ==================
def render_risk_trend_forecast_page(df: pd.DataFrame):
    st.subheader("Pipeline Health Overview & ML Forecast")

    selected_time = None
    if df is not None and not df.empty and "Timestamp" in df.columns:
        ts_all = pd.to_datetime(df["Timestamp"], errors="coerce").dropna().unique().tolist()
        ts_all = sorted(ts_all)
        selected_time = synced_time_slider(ts_all, "selected_time_risk")

        # Metrics placed here under the timestamp slider
        total_segments, predicted_high = sensor_page_counts(df)
        c1, c2, c3 = st.columns([1, 1, 3])
        c1.metric("Total Segments Monitored", f"{total_segments}")
        c2.metric("Predicted High-Risk Segments", f"{predicted_high}")

    if df is None or df.empty:
        st.info("No dataset loaded.")
        return

    if "Timestamp" in df.columns:
        tser = pd.to_datetime(df["Timestamp"], errors="coerce")
        mask = tser.notna()
        dfx = df.loc[mask].copy()
        dfx["Timestamp"] = tser.loc[mask]
    else:
        dfx = df.copy()
        dfx["Timestamp"] = pd.date_range(datetime.utcnow(), periods=len(dfx), freq="H")

    for col in ["Temperature_degC", "Pressure_MPa", "Flow_Rate_kg_s", "Hydrate_Risk_Index", "Wax_Deposition_Risk_Index", "Corrosion_Risk_Index"]:
        if col not in dfx.columns:
            dfx[col] = np.nan

    def row_max_risk(r):
        vals = []
        for k in ["Hydrate_Risk_Index", "Wax_Deposition_Risk_Index", "Corrosion_Risk_Index"]:
            try:
                vals.append(float(r[k]))
            except:
                pass
        return max(vals) if vals else np.nan

    dfx["Max_Risk_Index"] = dfx.apply(row_max_risk, axis=1)

    feats = ["Temperature_degC", "Pressure_MPa", "Flow_Rate_kg_s", "Hydrate_Risk_Index", "Wax_Deposition_Risk_Index", "Corrosion_Risk_Index"]

    def add_roll(g):
        g = g.sort_values("Timestamp").copy()
        for f in feats:
            g[f"{f}_rmean3"] = g[f].rolling(3, min_periods=1).mean()
            g[f"{f}_rstd3"]  = g[f].rolling(3, min_periods=1).std().fillna(0.0)
            g[f"{f}_rmin3"]  = g[f].rolling(3, min_periods=1).min()
            g[f"{f}_rmax3"]  = g[f].rolling(3, min_periods=1).max()
        return g

    dfx = dfx.groupby("Pipe_Segment_ID", group_keys=False).apply(add_roll)

    dfx["Next_High"] = dfx.groupby("Pipe_Segment_ID")["Max_Risk_Index"].shift(-1) >= 0.66
    dfl = dfx.dropna(subset=["Next_High"]).copy()

    split_time = dfl["Timestamp"].quantile(0.80) if not dfl.empty else None
    train = dfl[dfl["Timestamp"] <= split_time].copy() if split_time is not None else dfl.iloc[0:0].copy()
    test  = dfl[dfl["Timestamp"]  > split_time].copy() if split_time is not None else dfl.iloc[0:0].copy()

    feature_cols = []
    for f in feats:
        feature_cols += [f, f"{f}_rmean3", f"{f}_rstd3", f"{f}_rmin3", f"{f}_rmax3"]
    feature_cols = [c for c in feature_cols if c in dfl.columns]

    # Snapshot at selected time
    if selected_time is not None:
        snap_exact = dfx[dfx["Timestamp"] == selected_time]
        if not snap_exact.empty:
            snap = snap_exact.copy()
        else:
            snap = dfx[dfx["Timestamp"] <= selected_time].groupby("Pipe_Segment_ID", as_index=False).tail(1).copy()
    else:
        snap = dfx.groupby("Pipe_Segment_ID", as_index=False).tail(1).copy()

    use_model = False
    try:
        from sklearn.ensemble import RandomForestClassifier

        Xtr = train[feature_cols].astype(float).fillna(0.0)
        ytr = train["Next_High"].astype(int) if "Next_High" in train.columns and len(train) else pd.Series([], dtype=int)

        if len(Xtr) >= 20 and ytr.nunique() > 1:
            clf = RandomForestClassifier(n_estimators=150, random_state=42)
            clf.fit(Xtr, ytr)
            use_model = True

            Xsnap = snap[feature_cols].astype(float).fillna(0.0)
            snap["P(NextHigh)"] = clf.predict_proba(Xsnap)[:,1]
            ranking = snap[["Pipe_Segment_ID","P(NextHigh)","Max_Risk_Index"]].sort_values("P(NextHigh)", ascending=False)
        else:
            use_model = False
    except Exception:
        use_model = False

    if not use_model:
        snap = snap.copy()
        denom = (0.66 - 0.33) + 1e-6
        snap["P(NextHigh)"] = np.clip((snap["Max_Risk_Index"] - 0.33) / denom, 0, 1)
        ranking = snap[["Pipe_Segment_ID","P(NextHigh)","Max_Risk_Index"]].sort_values(["P(NextHigh)","Max_Risk_Index"], ascending=False)

    # Filter to only high-probability segments (>= 0.60)
    p_threshold = 0.60
    ranking_filtered = ranking[ranking["P(NextHigh)"] >= p_threshold].copy()
    ranking_filtered["P(NextHigh)"] = ranking_filtered["P(NextHigh)"].round(3)
    ranking_filtered["Max_Risk_Index"] = ranking_filtered["Max_Risk_Index"].round(3)

    st.markdown("**Predicted High-Risk Probability**")
    # ---- Rename + reorder display columns ----
    prob_display = ranking_filtered.reset_index(drop=True).rename(columns={
        "Pipe_Segment_ID": "Pipe Segment",
        "P(NextHigh)": "Future Risk Forecasted",
        "Max_Risk_Index": "Current Risk Index",
    })
    # Swap order: Pipe Segment, Current Risk Index, Future Risk Forecasted
    prob_display = prob_display[["Pipe Segment", "Current Risk Index", "Future Risk Forecasted"]]
    prob_html = html_table(
        prob_display,
        formats={
            "Current Risk Index": "{:.3f}",
            "Future Risk Forecasted": "{:.3f}",
        }
    )
    st.markdown(prob_html, unsafe_allow_html=True)

    # Recommendations (dynamic, two-line) based on selected time snapshot
    snap_idx = snap.set_index("Pipe_Segment_ID", drop=False) if "Pipe_Segment_ID" in snap.columns else None
    rec_rows = []
    for _, r in ranking_filtered.iterrows():
        seg = r["Pipe_Segment_ID"]
        row_snap = snap_idx.loc[seg] if (snap_idx is not None and seg in snap_idx.index) else {}
        actions_text = _recommend_actions_from_snapshot(row_snap)
        rec_rows.append({
            "Pipe_Segment_ID": seg,
            "Current_Max_Risk": float(r["Max_Risk_Index"]),
            "Recommended Actions": actions_text
        })
    rec_df = pd.DataFrame(rec_rows)

    st.markdown("**Maintenance Recommendations**")
    # Drop P(NextHigh) if present and rename columns for display
    rec_df_display = rec_df.drop(columns=["P(NextHigh)"], errors="ignore").rename(columns={
        "Pipe_Segment_ID": "Pipe Segment",
        "Current_Max_Risk": "Current Risk Index",
    })
    rec_html = html_table(
        rec_df_display,
        formats={"Current Risk Index":"{:.3f}"},
        newline_cols={"Recommended Actions"}
    )
    st.markdown(rec_html, unsafe_allow_html=True)

# ================== TABS ==================
tab1, tab2, tab3 = st.tabs(["Main Page", "Sensor Data Page", "Risk & Trend Forecast Page"])

with tab1:
    render_main_page(df)

with tab2:
    render_sensor_data_page(df)

with tab3:
    render_risk_trend_forecast_page(df)
