# app.py
import streamlit as st
import pandas as pd
from datetime import date

st.set_page_config(page_title="MTD vs Cohort Analyzer", layout="wide")

# =========================
# Utilities
# =========================
REQUIRED_COLS = [
    "Pipeline",
    "JetLearn Deal Source",
    "Country",
    "Student/Academic Counsellor",
    "Deal Stage",
    "Create Date",
]

def robust_read_csv(file_or_path):
    """Try multiple encodings."""
    for enc in ["utf-8", "utf-8-sig", "cp1252", "latin1"]:
        try:
            return pd.read_csv(file_or_path, encoding=enc)
        except Exception:
            continue
    raise RuntimeError("Could not read the CSV with tried encodings.")

def detect_measure_date_columns(df: pd.DataFrame):
    """Find date-like columns (other than Create Date)."""
    date_like = []
    for col in df.columns:
        if col == "Create Date":
            continue
        if any(k in col.lower() for k in ["date", "time", "timestamp"]):
            parsed = pd.to_datetime(df[col], errors="coerce", dayfirst=True, infer_datetime_format=True)
            if parsed.notna().sum() > 0:
                df[col] = parsed
                date_like.append(col)
    # Put Payment Received Date first if present
    if "Payment Received Date" in date_like:
        date_like = ["Payment Received Date"] + [c for c in date_like if c != "Payment Received Date"]
    return date_like

def in_filter(series: pd.Series, all_checked: bool, selected_values):
    """Return a boolean mask for global categorical filters."""
    if series is None:
        return pd.Series(True, index=selected_values.index if hasattr(selected_values, "index") else [])
    if all_checked:
        return pd.Series(True, index=series.index)
    uniq = series.dropna().astype(str).nunique()
    if selected_values and len(selected_values) == uniq:
        return pd.Series(True, index=series.index)
    if not selected_values:
        return pd.Series(False, index=series.index)
    return series.astype(str).isin(selected_values)

def month_period(s: pd.Series):
    return s.dt.to_period("M") if pd.api.types.is_datetime64_any_dtype(s) else pd.to_datetime(s, errors="coerce").dt.to_period("M")

def safe_minmax_date(s: pd.Series, fallback=(date(2024,1,1), date.today())):
    if s.isna().all():
        return fallback
    return (pd.to_datetime(s.min()).date(), pd.to_datetime(s.max()).date())

# =========================
# Data loader
# =========================
@st.cache_data(show_spinner=False)
def load_and_prepare(data_bytes: bytes | None, path_text: str | None):
    if data_bytes:
        from io import BytesIO
        df = robust_read_csv(BytesIO(data_bytes))
    elif path_text:
        df = robust_read_csv(path_text)
    else:
        raise ValueError("Please upload a CSV or provide a file path.")

    # clean headers
    df.columns = [c.strip() for c in df.columns]

    # check required
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}\nAvailable: {list(df.columns)}")

    # exclude invalid deals
    df = df[~df["Deal Stage"].astype(str).str.strip().eq("1.2 Invalid Deal")].copy()

    # parse Create Date
    df["Create Date"] = pd.to_datetime(df["Create Date"], errors="coerce", dayfirst=True, infer_datetime_format=True)
    df["Create_Month"] = df["Create Date"].dt.to_period("M")

    # find other date-like columns
    date_like_cols = detect_measure_date_columns(df)
    return df, date_like_cols

# =========================
# Sidebar: Data input
# =========================
st.sidebar.header("Data")
uploaded = st.sidebar.file_uploader("Upload your CSV", type=["csv"])
default_path = st.sidebar.text_input("...or enter CSV path", value="Master_sheet_DB_10percent.csv")
try:
    df, date_like_cols = load_and_prepare(uploaded.getvalue() if uploaded else None, default_path if not uploaded else None)
except Exception as e:
    st.sidebar.error(str(e))
    st.stop()

# =========================
# Sidebar: Global Filters (with All)
# =========================
st.sidebar.header("Global Filters")

def filter_block(label, colname, key_prefix):
    options = sorted([v for v in df[colname].dropna().astype(str).unique()])
    c1, c2 = st.sidebar.columns([1, 3])
    all_flag = c1.checkbox("All", value=True, key=f"{key_prefix}_all")
    selected = c2.multiselect(label, options, default=options, disabled=all_flag, key=f"{key_prefix}_sel")
    return all_flag, selected

pipe_all,  pipe_sel  = filter_block("Pipeline", "Pipeline", "pipe")
src_all,   src_sel   = filter_block("Deal Source", "JetLearn Deal Source", "src")
ctry_all,  ctry_sel  = filter_block("Country", "Country", "ctry")
cslr_all,  cslr_sel  = filter_block("Counsellor", "Student/Academic Counsellor", "cslr")

mask_cat = (
    in_filter(df["Pipeline"], pipe_all,  pipe_sel) &
    in_filter(df["JetLearn Deal Source"], src_all,   src_sel) &
    in_filter(df["Country"], ctry_all,  ctry_sel) &
    in_filter(df["Student/Academic Counsellor"], cslr_all, cslr_sel)
)

base = df[mask_cat].copy()

# =========================
# Main Controls
# =========================
st.title("MTD vs Cohort Analyzer")

if not date_like_cols:
    st.error("No usable date-like columns (other than Create Date) found. Add a column like 'Payment Received Date'.")
    st.stop()

measure_col = st.selectbox("What do you want to count on? (Measure date column)", date_like_cols, index=0)
if f"{measure_col}_Month" not in base.columns:
    base[f"{measure_col}_Month"] = base[measure_col].dt.to_period("M")

cols = st.columns(2)
mtd    = cols[0].checkbox("MTD — Create-date window; count where Measure-month == Create-month", value=True)
cohort = cols[1].checkbox("Cohort — Measure-date window; Create ignored for the measure", value=True)

# --- MTD date range: Create Date window
mtd_from, mtd_to = None, None
if mtd:
    st.subheader("MTD (Create-Date window)")
    cmin, cmax = safe_minmax_date(base["Create Date"])
    dr = st.date_input("Create Date window (From → To)", value=(cmin, cmax), key="mtd_window")
    if isinstance(dr, tuple) or isinstance(dr, list):
        mtd_from, mtd_to = dr
    else:
        # Streamlit may return a single date if user picks one
        mtd_from, mtd_to = cmin, cmax
    if mtd_from > mtd_to:
        st.error("MTD: 'From' is after 'To'. Adjust the Create-Date range.")
        mtd = False

# --- Cohort date range: Measure Date window
coh_from, coh_to = None, None
if cohort:
    st.subheader("Cohort (Measure-Date window)")
    mmin, mmax = safe_minmax_date(base[measure_col])
    dr2 = st.date_input("Measure Date window (From → To)", value=(mmin, mmax), key="coh_window")
    if isinstance(dr2, tuple) or isinstance(dr2, list):
        coh_from, coh_to = dr2
    else:
        coh_from, coh_to = mmin, mmax
    if coh_from > coh_to:
        st.error("Cohort: 'From' is after 'To'. Adjust the Measure-Date range.")
        cohort = False

# --- Split & Top sections (after date selections)
st.markdown("---")
st.subheader("Breakdowns & Leaderboards")

split_dims = st.multiselect(
    "Split by (optional)",
    options=["JetLearn Deal Source", "Country"],
    default=[],
    help="Add per-group tables for MTD/Cohort."
)

col_top1, col_top2, col_top3 = st.columns(3)
show_top_countries = col_top1.checkbox("Also show Top 5 Countries", value=True)
show_top_sources   = col_top2.checkbox("Also show Top 3 Deal Sources", value=True)
show_combo_pairs   = col_top3.checkbox("Also show Country × Deal Source together", value=False, help="Shows top 10 Country × Deal Source pairs")

st.markdown("---")

# =========================
# Metrics & Tables
# =========================
metrics_rows = []
split_tables = []
top_tables   = []

# ---------- MTD ----------
if mtd:
    in_create_window = base["Create Date"].between(pd.to_datetime(mtd_from), pd.to_datetime(mtd_to), inclusive="both")
    sub_mtd = base[in_create_window].copy()
    mtd_flag = (sub_mtd[measure_col].notna()) & (sub_mtd[f"{measure_col}_Month"] == sub_mtd["Create_Month"])

    metrics_rows.append({"Metric": f"MTD Count on '{measure_col}' (Create window {mtd_from} → {mtd_to})",
                         "Count": int(mtd_flag.sum())})
    metrics_rows.append({"Metric": f"Create Count in MTD window ({mtd_from} → {mtd_to})",
                         "Count": int(len(sub_mtd))})

    # Split tables
    if split_dims:
        g = sub_mtd.copy()
        g["_MTD Count"] = mtd_flag.astype(int)
        g["_Create Count in window"] = 1
        grp = g.groupby(split_dims, dropna=False).agg({
            "_Create Count in window": "sum",
            "_MTD Count": "sum"
        }).reset_index().rename(columns={
            "_Create Count in window": "Create Count in window",
            "_MTD Count": f"MTD Count on '{measure_col}'"
        }).sort_values(by=f"MTD Count on '{measure_col}'", ascending=False)
        split_tables.append(("MTD split by " + ", ".join(split_dims) + f" (Create window {mtd_from} → {mtd_to})", grp))

    # Top 5 Countries (MTD)
    if show_top_countries and "Country" in sub_mtd.columns:
        g2 = sub_mtd.copy()
        g2["_MTD Count"] = mtd_flag.astype(int)
        g2["_Create Count in window"] = 1
        top_ctry = g2.groupby("Country", dropna=False).agg({
            "_Create Count in window": "sum",
            "_MTD Count": "sum"
        }).reset_index().rename(columns={
            "_Create Count in window": "Create Count in window",
            "_MTD Count": f"MTD Count on '{measure_col}'"
        }).sort_values(by=f"MTD Count on '{measure_col}'", ascending=False).head(5)
        top_tables.append((f"Top 5 Countries — MTD (Create window {mtd_from} → {mtd_to})", top_ctry))

    # Top 3 Deal Sources (MTD)
    if show_top_sources and "JetLearn Deal Source" in sub_mtd.columns:
        g3 = sub_mtd.copy()
        g3["_MTD Count"] = mtd_flag.astype(int)
        g3["_Create Count in window"] = 1
        top_src = g3.groupby("JetLearn Deal Source", dropna=False).agg({
            "_Create Count in window": "sum",
            "_MTD Count": "sum"
        }).reset_index().rename(columns={
            "_Create Count in window": "Create Count in window",
            "_MTD Count": f"MTD Count on '{measure_col}'"
        }).sort_values(by=f"MTD Count on '{measure_col}'", ascending=False).head(3)
        top_tables.append((f"Top 3 Deal Sources — MTD (Create window {mtd_from} → {mtd_to})", top_src))

    # Country × Deal Source (MTD)
    if show_combo_pairs and {"Country", "JetLearn Deal Source"}.issubset(sub_mtd.columns):
        g4 = sub_mtd.copy()
        g4["_MTD Count"] = mtd_flag.astype(int)
        g4["_Create Count in window"] = 1
        both = g4.groupby(["Country", "JetLearn Deal Source"], dropna=False).agg({
            "_Create Count in window": "sum",
            "_MTD Count": "sum"
        }).reset_index().rename(columns={
            "_Create Count in window": "Create Count in window",
            "_MTD Count": f"MTD Count on '{measure_col}'"
        }).sort_values(by=f"MTD Count on '{measure_col}'", ascending=False).head(10)
        top_tables.append((f"Top Country × Deal Source — MTD (Create window {mtd_from} → {mtd_to})", both))

# ---------- Cohort ----------
if cohort:
    in_measure_window = base[measure_col].between(pd.to_datetime(coh_from), pd.to_datetime(coh_to), inclusive="both")
    in_create_cohort  = base["Create Date"].between(pd.to_datetime(coh_from), pd.to_datetime(coh_to), inclusive="both")

    metrics_rows.append({"Metric": f"Cohort Count on '{measure_col}' (Measure window {coh_from} → {coh_to})",
                         "Count": int(in_measure_window.sum())})
    metrics_rows.append({"Metric": f"Create Count in Cohort window (Create Date within {coh_from} → {coh_to})",
                         "Count": int(in_create_cohort.sum())})

    # Split tables
    if split_dims:
        g = base.copy()
        g["_Cohort Count"] = in_measure_window.astype(int)
        g["_Create Count in Cohort window"] = in_create_cohort.astype(int)
        grp2 = g.groupby(split_dims, dropna=False).agg({
            "_Cohort Count": "sum",
            "_Create Count in Cohort window": "sum"
        }).reset_index().rename(columns={
            "_Cohort Count": f"Cohort Count on '{measure_col}'",
            "_Create Count in Cohort window": "Create Count in Cohort window"
        }).sort_values(by=f"Cohort Count on '{measure_col}'", ascending=False)
        split_tables.append((f"Cohort split by " + ", ".join(split_dims) + f" (window {coh_from} → {coh_to})", grp2))

    # Top 5 Countries (Cohort)
    if show_top_countries and "Country" in base.columns:
        g2 = base.copy()
        g2["_Cohort Count"] = in_measure_window.astype(int)
        g2["_Create Count in Cohort window"] = in_create_cohort.astype(int)
        top_ctry2 = g2.groupby("Country", dropna=False).agg({
            "_Cohort Count": "sum",
            "_Create Count in Cohort window": "sum"
        }).reset_index().rename(columns={
            "_Cohort Count": f"Cohort Count on '{measure_col}'",
            "_Create Count in Cohort window": "Create Count in Cohort window"
        }).sort_values(by=f"Cohort Count on '{measure_col}'", ascending=False).head(5)
        top_tables.append((f"Top 5 Countries — Cohort (Measure window {coh_from} → {coh_to})", top_ctry2))

    # Top 3 Deal Sources (Cohort)
    if show_top_sources and "JetLearn Deal Source" in base.columns:
        g3 = base.copy()
        g3["_Cohort Count"] = in_measure_window.astype(int)
        g3["_Create Count in Cohort window"] = in_create_cohort.astype(int)
        top_src2 = g3.groupby("JetLearn Deal Source", dropna=False).agg({
            "_Cohort Count": "sum",
            "_Create Count in Cohort window": "sum"
        }).reset_index().rename(columns={
            "_Cohort Count": f"Cohort Count on '{measure_col}'",
            "_Create Count in Cohort window": "Create Count in Cohort window"
        }).sort_values(by=f"Cohort Count on '{measure_col}'", ascending=False).head(3)
        top_tables.append((f"Top 3 Deal Sources — Cohort (Measure window {coh_from} → {coh_to})", top_src2))

    # Country × Deal Source (Cohort)
    if show_combo_pairs and {"Country", "JetLearn Deal Source"}.issubset(base.columns):
        g4 = base.copy()
        g4["_Cohort Count"] = in_measure_window.astype(int)
        g4["_Create Count in Cohort window"] = in_create_cohort.astype(int)
        both2 = g4.groupby(["Country", "JetLearn Deal Source"], dropna=False).agg({
            "_Cohort Count": "sum",
            "_Create Count in Cohort window": "sum"
        }).reset_index().rename(columns={
            "_Cohort Count": f"Cohort Count on '{measure_col}'",
            "_Create Count in Cohort window": "Create Count in Cohort window"
        }).sort_values(by=f"Cohort Count on '{measure_col}'", ascending=False).head(10)
        top_tables.append((f"Top Country × Deal Source — Cohort (Measure window {coh_from} → {coh_to})", both2))

# =========================
# Output
# =========================
st.write("✅ Filters applied (excluding `1.2 Invalid Deal`)")
st.caption(f"Measure date: **{measure_col}** · Pipeline: {'All' if pipe_all else ', '.join(pipe_sel) or 'None'} · Deal Source: {'All' if src_all else ', '.join(src_sel) or 'None'} · Country: {'All' if ctry_all else ', '.join(ctry_sel) or 'None'} · Counsellor: {'All' if cslr_all else ', '.join(cslr_sel) or 'None'}")

if metrics_rows:
    st.subheader("Metrics")
    st.dataframe(pd.DataFrame(metrics_rows), use_container_width=True)
else:
    st.info("Select MTD and/or Cohort to see counts.")

for title, table in split_tables:
    st.subheader(title)
    st.dataframe(table, use_container_width=True)

for title, table in top_tables:
    st.subheader(title)
    st.dataframe(table, use_container_width=True)
