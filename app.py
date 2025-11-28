# app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import hashlib, io, json, os, time, random
from datetime import datetime, timedelta
import logging, sys

os.environ.setdefault("OPENAI_API_KEY", "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")
os.environ.setdefault("OPENAI_MODEL", "gpt-5-2025-08-07")
# -------------------------------------------------------------------

logging.basicConfig(stream=sys.stdout, level=logging.INFO, force=True)

# ===== Optional toggles (env-based) =====
USE_OPENAI_ACTIONS            = os.getenv("USE_OPENAI_ACTIONS", "1") == "1"
OPENAI_TIMEOUT_S              = int(os.getenv("OPENAI_TIMEOUT_S", "60"))    # read timeout
OPENAI_COOLDOWN_S             = int(os.getenv("OPENAI_COOLDOWN_S", "180"))  # cooldown after repeated failures
OPENAI_MAX_CALLS_PER_RENDER   = int(os.getenv("OPENAI_MAX_CALLS_PER_RENDER", "3"))
OPENAI_CONNECT_TIMEOUT_S      = int(os.getenv("OPENAI_CONNECT_TIMEOUT_S", "8"))

st.set_page_config(
    page_title="Subsea Pipeline Dashboard (Plotly, Reactive)",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.title("Subsea Pipeline Dashboard")

SEGMENT_STRETCH = 2.0  # spacing on fallback path

# ================== BOOK-ALIGNED RISK BANDS & FIXED PARAMS ==================
RISK_LOW_MAX = 0.20
RISK_MED_MAX = 0.60

# Fixed parameters (book-aligned)
MAOP_MPa   = 13.5   # Pressure banding vs MAOP
T_HYD_EQ_C = 20.0   # Hydrate equilibrium temperature at operating P
WAT_C      = 40.0   # Wax appearance temperature
Q_DESIGN   = 100.0  # Design/nominal flow

# Logistic mapping parameters (book-style margins → PoF)
HYD_M50, HYD_STEEP   = 1.5, 0.8    # PoF_hyd = 0.5 at ΔT_hyd=1.5°C
WAX_M50, WAX_STEEP   = 2.5, 1.0    # PoF_wax = 0.5 at ΔT_wax=2.5°C
UTIL_M50, UTIL_STEEP = 0.90, 0.05  # PoF_cor rises near utilization 0.90

THRESHOLDS_FIXED = {
    "Hydrate_Risk_Index": (RISK_LOW_MAX, RISK_MED_MAX),
    "Wax_Deposition_Risk_Index": (RISK_LOW_MAX, RISK_MED_MAX),
    "Corrosion_Risk_Index": (RISK_LOW_MAX, RISK_MED_MAX),
    "Temperature_degC": (39, 70),
    "Pressure_MPa": (10, 13),
    "Flow_Rate_kg_s": (60, 100),
}

COLOR_MAP = {"Low": "#16a34a", "Medium": "#facc15", "High": "#dc2626"}

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
    "Flow_Assurance_Status",
    "Timestamp",
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

# ================== SIDEBAR (UPLOAD ONLY) ==================
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
            td_style_extra = ""
            if newline_cols and c in newline_cols:
                text = text.replace("\n", "<br>")
                td_style_extra = " white-space:pre-line;"
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

# ================== BOOK-BASED RISK CORE ==================
def _safe_num(s, default=np.nan):
    try:
        v = float(s)
        if np.isfinite(v):
            return v
        return default
    except Exception:
        return default

def _clip01(x):
    return float(np.clip(x, 0.0, 1.0))

def _utilization(p_mpa):
    return _clip01(_safe_num(p_mpa, 0.0) / max(MAOP_MPa, 1e-6))

def _cof_from_pf(p_mpa, flow_kg_s):
    util = _utilization(p_mpa)
    thru = _clip01(_safe_num(flow_kg_s, 0.0) / max(Q_DESIGN, 1e-6)) if Q_DESIGN > 0 else 0.0
    return _clip01(0.5 * util + 0.5 * thru)

def _pof_from_margin(deltaT, m50, steep):
    return float(1.0 / (1.0 + np.exp((deltaT - m50) / max(steep, 1e-6))))

def _pof_from_util(util, m50=UTIL_M50, steep=UTIL_STEEP):
    util = float(util)
    return float(1.0 / (1.0 + np.exp((m50 - util) / max(steep, 1e-6))))

def _pof_hyd(temp_c):
    delta = _safe_num(temp_c, np.nan) - T_HYD_EQ_C
    if np.isnan(delta):
        return 0.0
    return _pof_from_margin(delta, HYD_M50, HYD_STEEP)

def _pof_wax(temp_c):
    delta = _safe_num(temp_c, np.nan) - WAT_C
    if np.isnan(delta):
        return 0.0
    return _pof_from_margin(delta, WAX_M50, WAX_STEEP)

def _pof_cor(pressure_mpa, cor_idx=np.nan):
    util = _utilization(pressure_mpa)
    pof_util = _pof_from_util(util)
    if np.isnan(cor_idx):
        return pof_util
    return _clip01(max(_safe_num(cor_idx, 0.0), pof_util))

def categorize_value(col, val):
    try:
        v = float(val)
    except Exception:
        return "Low"

    if "Risk_Index" in col:
        if v <= RISK_LOW_MAX: return "Low"
        elif v <= RISK_MED_MAX: return "Medium"
        else: return "High"

    if col == "Temperature_degC":
        dT_hyd = v - T_HYD_EQ_C
        dT_wax = v - WAT_C
        if dT_hyd <= 0 or dT_wax <= 0:
            return "High"
        if (0 < dT_hyd <= 3.0) or (0 < dT_wax <= 5.0):
            return "Medium"
        return "Low"

    if col == "Pressure_MPa":
        if v >= 0.95 * MAOP_MPa: return "High"
        elif v >= 0.85 * MAOP_MPa: return "Medium"
        else: return "Low"

    if col == "Flow_Rate_kg_s":
        if Q_DESIGN <= 0:
            return "Low"
        if v <= 0.60 * Q_DESIGN: return "High"
        elif v < 0.90 * Q_DESIGN: return "Medium"
        else: return "Low"

    if col in THRESHOLDS_FIXED:
        low, high = THRESHOLDS_FIXED[col]
        if v <= low: return "Low"
        elif v <= high: return "Medium"
        else: return "High"

    return "Low"

def compute_current_risk_at_time(df_all: pd.DataFrame, t_sel: pd.Timestamp) -> pd.DataFrame:
    dfx = df_all.copy()
    dfx["Timestamp"] = pd.to_datetime(dfx["Timestamp"], errors="coerce")
    target = dfx[dfx["Timestamp"] == t_sel]
    if target.empty:
        prev = dfx[dfx["Timestamp"] <= t_sel]
        target = prev.groupby("Pipe_Segment_ID", as_index=False).tail(1)

    out = []
    for _, r in target.iterrows():
        seg = r.get("Pipe_Segment_ID", "")
        T = _safe_num(r.get("Temperature_degC", np.nan))
        P = _safe_num(r.get("Pressure_MPa", np.nan))
        F = _safe_num(r.get("Flow_Rate_kg_s", np.nan))
        Cidx = _safe_num(r.get("Corrosion_Risk_Index", np.nan))

        pof_h = _pof_hyd(T); pof_w = _pof_wax(T); pof_c = _pof_cor(P, Cidx if not np.isnan(Cidx) else np.nan)
        cof   = _cof_from_pf(P, F)
        risk_overall = max(_clip01(pof_h*cof), _clip01(pof_w*cof), _clip01(pof_c*cof))
        out.append({"Pipe_Segment_ID": seg, "Current_Risk_Index": risk_overall})
    return pd.DataFrame(out)

def _per_segment_slope(df_seg: pd.DataFrame, col: str):
    g = df_seg.sort_values("Timestamp")
    s = pd.to_numeric(g[col], errors="coerce")
    t = pd.to_datetime(g["Timestamp"], errors="coerce")
    mask = s.notna() & t.notna()
    s = s[mask]; t = t[mask]
    if len(s) < 3:
        return 0.0
    x = (t.values.astype("datetime64[s]").astype("int64") / 3600.0).astype(float)  # hours
    x = x - x.min()
    try:
        slope = np.polyfit(x, s.values.astype(float), 1)[0]
    except Exception:
        slope = 0.0
    return float(slope)

def _segment_step_hours(df_seg: pd.DataFrame):
    t = pd.to_datetime(df_seg["Timestamp"], errors="coerce").dropna().sort_values().unique()
    if len(t) < 2:
        return 1.0
    dt = np.diff(t).astype("timedelta64[s]").astype(float) / 3600.0
    med = float(np.median(dt)) if len(dt) else 1.0
    return med if med > 0 else 1.0

def compute_future_risk(df_all: pd.DataFrame, t_sel: pd.Timestamp) -> pd.DataFrame:
    dfx = df_all.copy()
    dfx["Timestamp"] = pd.to_datetime(dfx["Timestamp"], errors="coerce")
    out = []
    for seg, g in dfx.groupby("Pipe_Segment_ID"):
        g = g[g["Timestamp"] <= t_sel]
        if g.empty:
            continue
        g = g.sort_values("Timestamp")

        T_now = _safe_num(g.tail(1).get("Temperature_degC", np.nan).values[0])
        P_now = _safe_num(g.tail(1).get("Pressure_MPa", np.nan).values[0])
        F_now = _safe_num(g.tail(1).get("Flow_Rate_kg_s", np.nan).values[0])
        C_now = _safe_num(g.tail(1).get("Corrosion_Risk_Index", np.nan).values[0])

        dT_dt = _per_segment_slope(g, "Temperature_degC") if "Temperature_degC" in g.columns else 0.0
        dP_dt = _per_segment_slope(g, "Pressure_MPa") if "Pressure_MPa" in g.columns else 0.0
        dF_dt = _per_segment_slope(g, "Flow_Rate_kg_s") if "Flow_Rate_kg_s" in g.columns else 0.0
        dC_dt = _per_segment_slope(g, "Corrosion_Risk_Index") if "Corrosion_Risk_Index" in g.columns else 0.0

        step_h = _segment_step_hours(g)

        T_pred = T_now + dT_dt * step_h
        P_pred = P_now + dP_dt * step_h
        F_pred = F_now + dF_dt * step_h
        C_pred = _clip01(_safe_num(C_now, 0.0) + dC_dt * step_h) if not np.isnan(C_now) else np.nan

        pof_h_f = _pof_hyd(T_pred)
        pof_w_f = _pof_wax(T_pred)
        pof_c_f = _pof_cor(P_pred, C_pred if not np.isnan(C_pred) else np.nan)
        cof_f   = _cof_from_pf(P_pred, F_pred)

        risk_future = max(_clip01(pof_h_f*cof_f), _clip01(pof_w_f*cof_f), _clip01(pof_c_f*cof_f))
        out.append({"Pipe_Segment_ID": seg, "Future_Risk_Forecasted": risk_future})
    return pd.DataFrame(out)

# Snapshot rows per segment at (or up to) selected time
def snapshot_rows_at_time(df_all: pd.DataFrame, t_sel: pd.Timestamp) -> pd.DataFrame:
    dfx = df_all.copy()
    dfx["Timestamp"] = pd.to_datetime(dfx["Timestamp"], errors="coerce")
    dfx = dfx.dropna(subset=["Timestamp"])
    prev = dfx[dfx["Timestamp"] <= t_sel]
    if prev.empty:
        after = dfx[dfx["Timestamp"] >= t_sel].sort_values("Timestamp")
        return after.groupby("Pipe_Segment_ID", as_index=False).head(1)
    return prev.sort_values("Timestamp").groupby("Pipe_Segment_ID", as_index=False).tail(1)

# Vectorized scan (optional, kept)
def compute_max_observed_current_risk(df_all: pd.DataFrame) -> pd.DataFrame:
    dfx = df_all.copy()
    dfx["Timestamp"] = pd.to_datetime(dfx["Timestamp"], errors="coerce")
    dfx = dfx.dropna(subset=["Timestamp"])

    T = pd.to_numeric(dfx.get("Temperature_degC"), errors="coerce")
    P = pd.to_numeric(dfx.get("Pressure_MPa"), errors="coerce")
    F = pd.to_numeric(dfx.get("Flow_Rate_kg_s"), errors="coerce")
    Cidx = pd.to_numeric(dfx.get("Corrosion_Risk_Index"), errors="coerce").fillna(0.0)

    util = (P / max(MAOP_MPa, 1e-6)).clip(0, 1)
    dT_hyd = (T - T_HYD_EQ_C)
    dT_wax = (T - WAT_C)

    pof_h = 1.0 / (1.0 + np.exp((dT_hyd - HYD_M50) / max(HYD_STEEP, 1e-6)))
    pof_w = 1.0 / (1.0 + np.exp((dT_wax - WAX_M50) / max(WAX_STEEP, 1e-6)))
    pof_c_util = 1.0 / (1.0 + np.exp((UTIL_M50 - util) / max(UTIL_STEEP, 1e-6)))
    pof_c = np.maximum(pof_c_util, Cidx).clip(0, 1)

    thru = (F / max(Q_DESIGN, 1e-6)).clip(0, 1)
    cof = (0.5 * util + 0.5 * thru).clip(0, 1)

    risk_h = (pof_h * cof).clip(0, 1)
    risk_w = (pof_w * cof).clip(0, 1)
    risk_c = (pof_c * cof).clip(0, 1)
    risk_overall = np.maximum.reduce([risk_h, risk_w, risk_c])

    dfx["_Risk_Row"] = risk_overall
    return dfx.groupby("Pipe_Segment_ID")["_Risk_Row"].max().reset_index().rename(columns={"_Risk_Row":"Max_Observed_Risk"})

# ================== OpenAI RECOMMENDATIONS + CACHE ==================
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

def _openai_call_budget_ok(t_sel):
    """Limit the number of OpenAI calls per render (per selected timestamp)."""
    try:
        key = f"openai_calls::{pd.to_datetime(t_sel).isoformat() if t_sel is not None else 'none'}"
    except Exception:
        key = "openai_calls::none"
    cnt = st.session_state.get(key, 0)
    if cnt >= OPENAI_MAX_CALLS_PER_RENDER:
        return False
    st.session_state[key] = cnt + 1
    return True

def _openai_generate_actions(row):
    # Respect toggle
    if not USE_OPENAI_ACTIONS:
        return None

    # Circuit breaker
    try:
        if st.session_state.get("openai_circuit_open_until", 0) > time.time():
            return None
    except Exception:
        pass

    try:
        import requests
    except Exception as e:
        print(f"OpenAI diag: requests import failed: {e}", flush=True)
        return None

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("OpenAI diag: OPENAI_API_KEY not set", flush=True)
        return None

    env_model = (os.getenv("OPENAI_MODEL") or "").strip()
    if not env_model:
        print("OpenAI diag: OPENAI_MODEL not set; skipping API", flush=True)
        return None

    # Build snapshot
    def _get_any(r, names, default=None):
        get = getattr(r, "get", None)
        if get is None:
            return default
        for n in names:
            v = get(n, None)
            if v is not None:
                return v
        return default

    snapshot = {
        "Pressure": _get_any(row, ["Pressure_MPa", "Pressure"]),
        "Temperature": _get_any(row, ["Temperature_degC", "Temperature"]),
        "FlowRate": _get_any(row, ["Flow_Rate_kg_s", "FlowRate"]),
        "Hydrate_Risk_Index": _get_any(row, ["Hydrate_Risk_Index"]),
        "Wax_Deposition_Risk_Index": _get_any(row, ["Wax_Deposition_Risk_Index"]),
        "Corrosion_Risk_Index": _get_any(row, ["Corrosion_Risk_Index"]),
    }

    system_msg = (
        "You are a subsea pipeline maintenance assistant. "
        "Return exactly TWO concise, numbered maintenance actions tailored to the snapshot. "
        "No explanations, no extra lines."
    )
    user_prompt = (
        "Snapshot:\n"
        f"- Pressure={snapshot['Pressure']}\n"
        f"- Temperature={snapshot['Temperature']}\n"
        f"- FlowRate={snapshot['FlowRate']}\n"
        f"- Hydrate_Risk_Index={snapshot['Hydrate_Risk_Index']}\n"
        f"- Wax_Deposition_Risk_Index={snapshot['Wax_Deposition_Risk_Index']}\n"
        f"- Corrosion_Risk_Index={snapshot['Corrosion_Risk_Index']}\n\n"
        "Reply in plain text, strictly two lines:\n"
        "1. <first action>\n"
        "2. <second action>"
    )

    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    # Try ONLY the env model (no fallback to other models). If it fails, return None.
    base = {
        "model": env_model,
        "messages": [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_prompt},
        ],
    }

    backoff = 1.5
    for attempt in range(1, 3):  # up to 2 attempts on the specified model
        try:
            r = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=base,
                timeout=(OPENAI_CONNECT_TIMEOUT_S, OPENAI_TIMEOUT_S)
            )
        except Exception as e:
            print(f"OpenAI diag: request error (model={env_model}, attempt={attempt}): {e}", flush=True)
            time.sleep(backoff + random.uniform(0, 0.4))
            backoff *= 2
            continue

        if r.status_code == 200:
            try:
                data = r.json()
                choice = (data.get("choices") or [{}])[0]
                msg = choice.get("message", {})
                # ---- extract text ----
                content = msg.get("content", "")
                if isinstance(content, str) and content.strip():
                    text = content.strip()
                elif isinstance(content, list):
                    parts_out = []
                    for part in content:
                        if isinstance(part, str) and part.strip():
                            parts_out.append(part.strip())
                        elif isinstance(part, dict):
                            t = part.get("text")
                            if isinstance(t, str) and t.strip():
                                parts_out.append(t.strip())
                            elif isinstance(t, dict):
                                val = t.get("value") or t.get("content")
                                if isinstance(val, str) and val.strip():
                                    parts_out.append(val.strip())
                            for k in ("content", "output_text", "input_text"):
                                v = part.get(k)
                                if isinstance(v, str) and v.strip():
                                    parts_out.append(v.strip())
                    text = ("\n".join(parts_out).strip()) if parts_out else ""
                else:
                    text = msg.get("reasoning_content") if isinstance(msg.get("reasoning_content"), str) else ""

                # ---- normalize to exactly two numbered lines ----
                def _normalize_two_lines(t):
                    raw = [ln.strip() for ln in (t or "").splitlines() if ln.strip()]
                    cleaned = []
                    for ln in raw:
                        while ln[:1] in "•-":
                            ln = ln[1:].strip()
                        for p in ("1. ", "1) ", "2. ", "2) "):
                            if ln.startswith(p):
                                ln = ln[len(p):].strip()
                                break
                        if ln:
                            cleaned.append(ln)
                        if len(cleaned) == 2:
                            break
                    if len(cleaned) < 2:
                        return None
                    return f"1. {cleaned[0]}\n2. {cleaned[1]}"

                if text:
                    out = _normalize_two_lines(text)
                    if out:
                        print(f"Recommendations source: OpenAI ({env_model})", flush=True)
                        return out
            except Exception as e:
                print(f"OpenAI diag: JSON parse failed (model={env_model}, attempt={attempt}): {e}", flush=True)
            time.sleep(backoff + random.uniform(0, 0.4))
            backoff *= 2
            continue

        # Transient server/CDN/rate-limit
        if r.status_code in (429, 408) or r.status_code >= 500:
            preview = r.text[:140].replace("\n", " ")
            print(f"OpenAI diag: status {r.status_code} (model={env_model}, attempt={attempt}): {preview}", flush=True)
            time.sleep(backoff + random.uniform(0, 0.6))
            backoff *= 2
            continue

        # Non-retryable (4xx other than 408/429): give up
        preview = r.text[:140].replace("\n", " ")
        print(f"OpenAI diag: non-retryable status {r.status_code} (model={env_model}): {preview}", flush=True)
        break

    # Open circuit briefly to avoid spamming logs
    try:
        st.session_state["openai_circuit_open_until"] = time.time() + OPENAI_COOLDOWN_S
    except Exception:
        pass
    print("OpenAI diag: env model failed; using fallback rules", flush=True)
    return None

def _fallback_actions_from_snapshot(row):
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

    if pd.notna(P) and P >= 0.95 * MAOP_MPa:
        suggestions += ["Verify pressure control/relief systems", "Inspect upstream choke/possible blockage"]
    if pd.notna(F) and Q_DESIGN > 0 and F <= 0.60 * Q_DESIGN:
        suggestions += ["Investigate restriction or bypass", "Plan pigging / check valves & filters"]
    if pd.notna(T) and ((T - T_HYD_EQ_C) <= 0 or (T - WAT_C) <= 0):
        suggestions += ["Stabilize temperature / check insulation/heating", "Inject hydrate inhibitor (MEG/methanol) if needed"]

    if ((pd.notna(C) and C >= RISK_MED_MAX) or top_risk == "Corrosion"):
        suggestions += ["Prioritize corrosion mitigation", "Schedule ultrasonic thickness survey"]
    elif ((pd.notna(W) and W >= RISK_MED_MAX) or top_risk == "Wax"):
        suggestions += ["Schedule pigging to remove wax", "Optimize temperature/flow to reduce wax deposition"]
    elif ((pd.notna(H) and H >= RISK_MED_MAX) or top_risk == "Hydrate"):
        suggestions += ["Increase temperature / manage insulation", "Adjust hydrate inhibitor dosing"]

    return _format_two_actions(suggestions)

def _recommend_actions_from_snapshot(row):
    ai_txt = _openai_generate_actions(row)
    if ai_txt:
        print("Recommendations source: OpenAI", flush=True)
        return ai_txt
    print("Recommendations source: fallback rules", flush=True)
    return _fallback_actions_from_snapshot(row)

# ---- CACHE HELPERS (per segment, per timestamp, per snapshot) ----
def _snapshot_fingerprint(row: dict | pd.Series) -> str:
    keys = [
        "Pressure_MPa", "Temperature_degC", "Flow_Rate_kg_s",
        "Hydrate_Risk_Index", "Wax_Deposition_Risk_Index", "Corrosion_Risk_Index"
    ]
    snap = {}
    get = row.get if hasattr(row, "get") else (lambda k, d=None: getattr(row, k, d))
    for k in keys:
        v = get(k, np.nan)
        try:
            snap[k] = None if pd.isna(v) else float(v)
        except Exception:
            snap[k] = None
    payload = json.dumps(snap, sort_keys=True, default=str)
    return hashlib.md5(payload.encode()).hexdigest()

def get_actions_cached(seg_id, row_snap, t_sel):
    """Cache maintenance actions by (segment, timestamp, snapshot) and cap OpenAI calls per render."""
    if "ai_actions_cache" not in st.session_state:
        st.session_state["ai_actions_cache"] = {}
    fp = _snapshot_fingerprint(row_snap if isinstance(row_snap, dict) else row_snap.to_dict())
    key = (str(seg_id), pd.to_datetime(t_sel).isoformat() if t_sel is not None else "none", fp)
    cache = st.session_state["ai_actions_cache"]
    if key in cache:
        return cache[key]

    txt = None
    if _openai_call_budget_ok(t_sel):
        txt = _openai_generate_actions(row_snap)
    if not txt:
        txt = _fallback_actions_from_snapshot(row_snap)
    cache[key] = txt
    return txt

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

# ================== SENSOR TREND INSIGHTS, WARNINGS ==================
MIN_ABS_DELTA = {
    "Pressure_MPa": 0.20,
    "Temperature_degC": 0.80,
    "Flow_Rate_kg_s": 3.00,
    "Hydrate_Risk_Index": 0.03,
    "Wax_Deposition_Risk_Index": 0.03,
    "Corrosion_Risk_Index": 0.03,
}
STD_MULTIPLIER_BY_SENSOR = {
    "Pressure_MPa": 1.5,
    "Temperature_degC": 1.5,
    "Flow_Rate_kg_s": 1.5,
    "Hydrate_Risk_Index": 1.0,
    "Wax_Deposition_Risk_Index": 1.0,
    "Corrosion_Risk_Index": 1.0,
}
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
}

def generate_active_warnings(df: pd.DataFrame, selected_time: pd.Timestamp | None, window_minutes=60, max_messages=10):
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

            now_cat = categorize_value(col, v_now)
            then_cat = categorize_value(col, v_then)
            if not (now_cat == "High" and then_cat != "High"):
                continue

            thr = local_std_threshold(g, col)
            if abs(d_val) < thr or thr <= 0:
                continue

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

        if seg_msgs:
            order = np.argsort(seg_scores)[::-1]
            for idx in order[:2]:
                messages.append(seg_msgs[idx])
                severities.append(seg_scores[idx])

    if not messages:
        return []
    order = np.argsort(severities)[::-1]
    return [messages[i] for i in order[:max_messages]]

def generate_sensor_trend_insights(df: pd.DataFrame, selected_time: pd.Timestamp | None, trend_minutes=120):
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
        label = {
            "Pressure_MPa":"Pressure Trend",
            "Temperature_degC":"Temperature",
            "Flow_Rate_kg_s":"Flow Rate Trend",
            "Hydrate_Risk_Index":"Hydrate Risk",
            "Wax_Deposition_Risk_Index":"Wax Deposition Risk",
            "Corrosion_Risk_Index":"Corrosion Risk",
        }.get(col, pretty_name(col))
        if col not in sensors:
            insights.append(f"{label}: No data in window")
            continue

        s = agg[col] if (not agg.empty and col in agg.columns) else pd.Series(dtype=float)
        if s.notna().sum() < 3:
            insights.append(f"{label}: Insufficient data in window")
            continue
        slope_h, std = _fit_slope(agg["Timestamp"], s)
        desc, _ = classify(col, slope_h, std)
        insights.append(f"{label}: {desc}")

    return insights

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

    num_cols = ["Temperature_degC", "Pressure_MPa", "Flow_Rate_kg_s", "Hydrate_Risk_Index", "Wax_Deposition_Risk_Index", "Corrosion_Risk_Index"]
    exist = [c for c in num_cols if c in df.columns]
    if not exist:
        st.info("No numeric sensor columns found to summarize.")
        return

    def sensor_display_name(col):
        base = COLUMN_LABELS.get(col, col)
        unit = UNITS.get(col)
        return f"{base} ({unit})" if unit else base

    rows = []
    for col in exist:
        s = pd.to_numeric(df[col], errors="coerce")
        rows.append({
            "Sensor (raw)": col,
            "Sensor": sensor_display_name(col),
            "Min Value": float(np.nanmin(s)),
            "Max Value": float(np.nanmax(s)),
            "Average Value": float(np.nanmean(s)),
        })
    summary_df = pd.DataFrame(rows)

    summary_df_display = summary_df.copy()
    for c in ["Min Value", "Max Value", "Average Value"]:
        summary_df_display[c] = summary_df_display[c].round(2)

    visible_df = summary_df_display.drop(columns=["Sensor (raw)"]).reset_index(drop=True)

    def _sensor_cell_style(row_idx, col_name, value):
        if col_name not in {"Min Value", "Max Value", "Average Value"}:
            return {}
        raw_col = summary_df["Sensor (raw)"].iloc[row_idx]
        level = categorize_value(raw_col, value)
        color = COLOR_MAP.get(level, None)
        return {"background-color": color, "color": "black"} if color else {}

    sensor_html = html_table(
        visible_df,
        formats={"Min Value": "{:.2f}", "Max Value": "{:.2f}", "Average Value": "{:.2f}"},
        per_cell_style_fn=_sensor_cell_style
    )
    st.markdown(sensor_html, unsafe_allow_html=True)

    st.subheader("Active Warnings")
    warnings = generate_active_warnings(df, selected_time=selected_time, window_minutes=60, max_messages=10)
    if not warnings:
        st.info("No new transitions into High risk in the last hour.")
    else:
        for msg in warnings:
            st.write(msg)

    st.subheader("Sensor Trend Insights")
    trend_window = 120
    insights = generate_sensor_trend_insights(df, selected_time=selected_time, trend_minutes=trend_window)
    if not insights or not any(str(x).strip() for x in insights):
        st.info(f"No discernible sensor trends in the last {trend_window} minutes.")
    else:
        for line in insights:
            st.write(line)

# ================== RISK & TREND FORECAST PAGE ==================
def render_risk_trend_forecast_page(df: pd.DataFrame):
    st.subheader("Pipeline Health Overview & Forecast (Book-Based PoF×CoF)")

    selected_time = None
    if df is not None and not df.empty and "Timestamp" in df.columns:
        ts_all = pd.to_datetime(df["Timestamp"], errors="coerce").dropna().unique().tolist()
        ts_all = sorted(ts_all)
        selected_time = synced_time_slider(ts_all, "selected_time_risk")

    if df is None or df.empty:
        st.info("No dataset loaded.")
        return

    # === Compute at selected time ===
    if selected_time is None:
        selected_time = pd.to_datetime(df["Timestamp"], errors="coerce").dropna().max()

    cur = compute_current_risk_at_time(df, selected_time)
    fut = compute_future_risk(df, selected_time)
    overview = cur.merge(fut, on="Pipe_Segment_ID", how="left")

    total_segments = overview["Pipe_Segment_ID"].nunique() if not overview.empty else 0
    predicted_high = int((overview["Future_Risk_Forecasted"] >= 0.60).sum()) if "Future_Risk_Forecasted" in overview.columns else 0

    c1, c2, _ = st.columns([1, 1, 3])
    c1.metric("Total Segments Monitored", f"{total_segments}")
    c2.metric("Predicted High-Risk Segments", f"{predicted_high}")

    ranking = overview.fillna(0.0).sort_values(
        ["Future_Risk_Forecasted", "Current_Risk_Index"], ascending=False
    ) if not overview.empty else overview.copy()

    # ---- Table A: Predicted High-Risk Probability (future risk ≥ 0.60) ----
    st.markdown("**Predicted High-Risk Probability (PoF×CoF forecast)**")
    if ranking is None or ranking.empty:
        st.markdown(html_table(pd.DataFrame(columns=["Pipe Segment","Current Risk Index","Future Risk Forecasted"])), unsafe_allow_html=True)
    else:
        ranking_future = ranking[ranking["Future_Risk_Forecasted"] >= 0.60].copy()
        prob_display = ranking_future.reset_index(drop=True).rename(columns={
            "Pipe_Segment_ID": "Pipe Segment",
            "Current_Risk_Index": "Current Risk Index",
            "Future_Risk_Forecasted": "Future Risk Forecasted",
        })[["Pipe Segment", "Current Risk Index", "Future Risk Forecasted"]]
        prob_html = html_table(
            prob_display,
            formats={"Current Risk Index": "{:.3f}", "Future Risk Forecasted": "{:.3f}"}
        )
        st.markdown(prob_html, unsafe_allow_html=True)

    # ---- Table B: Maintenance Recommendations (SYNCED) filtered by CURRENT risk ≥ 0.60 ----
    st.markdown("**Maintenance Recommendations**")
    rec_rows = []
    if ranking is not None and not ranking.empty:
        ranking_current = ranking[ranking["Current_Risk_Index"] >= 0.60].copy()

        snap_rows = snapshot_rows_at_time(df, selected_time)
        snap_idx = snap_rows.set_index("Pipe_Segment_ID", drop=False) if "Pipe_Segment_ID" in snap_rows.columns else None

        for _, r in ranking_current.iterrows():
            seg = r["Pipe_Segment_ID"]
            row_snap = snap_idx.loc[seg].to_dict() if (snap_idx is not None and seg in snap_idx.index) else {}
            actions_text = get_actions_cached(seg, row_snap, selected_time)  # cached OpenAI/fallback
            rec_rows.append({
                "Pipe Segment": seg,
                "Current Risk Index": float(r["Current_Risk_Index"]),
                "Recommended Actions": actions_text
            })

    # Safe construction + sorting (prevents KeyError if empty)
    rec_df = pd.DataFrame(rec_rows, columns=["Pipe Segment","Current Risk Index","Recommended Actions"])
    if not rec_df.empty and "Current Risk Index" in rec_df.columns:
        rec_df = rec_df.sort_values("Current Risk Index", ascending=False).reset_index(drop=True)

    if rec_df.empty:
        st.info("No segments with Current Risk Index ≥ 0.60 at the selected time.")
    else:
        rec_html = html_table(
            rec_df,
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

        # Fall back exactly as you requested
        return rule_based_actions(payload) if "rule_based_actions" in globals() else {"summary": f"OpenAI call failed ({e})", "actions": []}


