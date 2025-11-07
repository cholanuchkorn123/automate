import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import re
from pathlib import Path
import glob
import os
from datetime import datetime
from typing import Optional, Tuple  # <-- ใช้แทน union type แบบ 3.10+

# ==========================
# Settings
# ==========================
CSV_PATH = "report/rf_report.csv"   # fallback ถ้าไม่มี history
HISTORY_DIR = "report/history"
COLORS = {"PASS": "#28a745", "FAIL": "#dc3545"}  # เขียว/แดง

st.set_page_config(page_title="QA Test Dashboard [Tablet]", layout="wide")

# ==========================
# Helpers
# ==========================
def find_latest_history_file(history_dir: str) -> Optional[str]:
    files = glob.glob(os.path.join(history_dir, "*.csv"))
    if not files:
        return None
    return max(files, key=os.path.getmtime)  # mtime ล่าสุด

def read_latest_report() -> Tuple[pd.DataFrame, str]:
    latest_hist = find_latest_history_file(HISTORY_DIR)
    source = latest_hist if latest_hist else CSV_PATH
    df = pd.read_csv(source)
    return df, source

def nice_name(path: str) -> str:
    try:
        ts = os.path.getmtime(path)
        return f"{Path(path).name} • {datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')}"
    except Exception:
        return Path(path).name

# ==========================
# Load & Clean (LATEST REPORT for dashboard)
# ==========================
df, source_path = read_latest_report()
df["Status"] = df["Status"].astype(str).str.upper().str.strip()
df["Feature"] = df["Feature"].astype(str).str.strip()
df["Duration(s)"] = pd.to_numeric(df["Duration(s)"], errors="coerce")

# index ลำดับการรัน (กรณีไม่มี timestamp)
df = df.reset_index(drop=False).rename(columns={"index": "Order"})
df["Order"] = df["Order"] + 1

# สำหรับเรียง Test ID เป็นตัวเลขจริง
df["TestID_num"] = df["Test ID"].astype(str).apply(
    lambda s: int(re.search(r"(\d+)", s).group(1)) if re.search(r"(\d+)", s) else 0
)

# ==========================
# Header & KPIs (ล่าสุด)
# ==========================
st.title("📊 QA Automation Dashboard [Tablet]")
st.caption(f"Using latest report: **{nice_name(source_path)}**")

total_tests = len(df)
pass_count = int((df["Status"] == "PASS").sum())
fail_count = int((df["Status"] == "FAIL").sum())
pass_rate = (pass_count / total_tests * 100) if total_tests else 0.0
fail_rate = 100 - pass_rate
avg_duration = df["Duration(s)"].mean()

n_features = df["Feature"].nunique()
slow_row = df.sort_values("Duration(s)", ascending=False).head(1)
slow_name = slow_row["Test Case Name"].iloc[0] if not slow_row.empty else "-"
slow_dur = slow_row["Duration(s)"].iloc[0] if not slow_row.empty else np.nan

# ==========================
# TOP: Stability — Run-level cumulative stability (รวมหลายรัน)
# ==========================
st.subheader("🩺 Stability (per run)")

def load_runs_for_stability(base_df: pd.DataFrame):
    """
    1) ถ้า base_df มี 'Run' -> ใช้เลย (รวมหลายรันในไฟล์เดียว)
    2) ถ้ามีไฟล์ report/history/*.csv -> รวมทุกไฟล์ (1 ไฟล์ = 1 รัน)
    3) ไม่งั้นถือว่าไฟล์ปัจจุบันคือรันเดียว (Run=1)
    """
    if "Run" in base_df.columns:
        tmp = base_df.copy()
        tmp["Run"] = tmp["Run"]
        return {"type": "column", "data": tmp}

    files = sorted(glob.glob(os.path.join(HISTORY_DIR, "*.csv")))
    if files:
        frames = []
        for i, f in enumerate(files, start=1):
            t = pd.read_csv(f)
            t["Status"] = t["Status"].astype(str).str.upper().str.strip()
            t["Run"] = i
            frames.append(t)
        all_df = pd.concat(frames, ignore_index=True)
        return {"type": "files", "data": all_df, "files": files}

    one = base_df.copy()
    one["Run"] = 1
    return {"type": "single", "data": one}

src = load_runs_for_stability(df)
runs_df = src["data"]

run_summary = (
    runs_df.assign(is_pass=(runs_df["Status"] == "PASS").astype(int))
           .groupby("Run", as_index=False)
           .agg(total_tests=("Status", "size"),
                pass_count=("is_pass", "sum"))
)
run_summary["pass_rate"] = (run_summary["pass_count"] / run_summary["total_tests"]) * 100
run_summary["fail_rate"] = 100 - run_summary["pass_rate"]

run_summary = run_summary.sort_values("Run")
run_summary["cum_pass_stability"] = run_summary["pass_rate"].expanding().mean()
run_summary["cum_fail_stability"] = 100 - run_summary["cum_pass_stability"]

# Plot
x = run_summary["Run"]
y_pass = run_summary["cum_pass_stability"]
y_fail = run_summary["cum_fail_stability"]

fig_stab_run = go.Figure()
fig_stab_run.add_trace(go.Scatter(
    x=x, y=y_pass, name="Cumulative PASS stability",
    mode="lines+markers",
    line=dict(color=COLORS["PASS"], width=3),
    fill="tozeroy",
    hovertemplate="Run %{x}<br>Stability %{y:.1f}%<extra></extra>"
))
fig_stab_run.add_trace(go.Scatter(
    x=x, y=y_fail, name="Cumulative FAIL stability",
    mode="lines+markers",
    line=dict(color=COLORS["FAIL"], width=2, dash="dot"),
    hovertemplate="Run %{x}<br>Stability %{y:.1f}%<extra></extra>"
))
fig_stab_run.add_trace(go.Bar(
    x=x, y=run_summary["pass_rate"],
    name="Pass rate (this run)",
    marker_color="rgba(40,167,69,0.35)",
    opacity=0.25,
    hovertemplate="Run %{x}<br>Pass rate %{y:.1f}%<extra></extra>",
))
target = 95
fig_stab_run.add_hline(
    y=target, line_dash="dash", line_color="#9aa0a6",
    annotation_text=f"Target {target}%", annotation_position="top left"
)
title_suffix = {
    "column": "from 'Run' column (single file)",
    "files": f"from {len(src.get('files', []))} files in report/history/",
    "single": "from current file (1 run)"
}[src["type"]]
fig_stab_run.update_layout(
    title=f"Stability per Run — Cumulative Pass/Fail ({title_suffix})",
    xaxis_title="Run #",
    yaxis_title="Rate (%)",
    barmode="overlay",
    legend_title="",
    margin=dict(t=60, r=10, l=10, b=10),
)
fig_stab_run.update_yaxes(range=[0, 100], ticks="outside")
st.plotly_chart(fig_stab_run, use_container_width=True)

# ---- Stability Summary Cards
st.markdown("### 📈 Stability Summary")
latest_pass_stab = float(run_summary["cum_pass_stability"].iloc[-1])
latest_fail_stab = float(run_summary["cum_fail_stability"].iloc[-1])
avg_pass_rate = float(run_summary["pass_rate"].mean())
n_runs = int(len(run_summary))
best_run = int(run_summary.loc[run_summary["pass_rate"].idxmax(), "Run"])
worst_run = int(run_summary.loc[run_summary["pass_rate"].idxmin(), "Run"])

col_a, col_b, col_c, col_d = st.columns(4)
with col_a:
    st.metric("Cumulative PASS Stability", f"{latest_pass_stab:.1f}%")
    st.metric("Avg Pass Rate per Run", f"{avg_pass_rate:.1f}%")
with col_b:
    st.metric("Cumulative FAIL Stability", f"{latest_fail_stab:.1f}%")
    st.metric("# Runs", n_runs)
with col_c:
    st.metric("Best Run #", best_run)
    st.metric("Worst Run #", worst_run)
with col_d:
    st.metric("Target", "95%")
    if latest_pass_stab >= 95:
        st.success("✅ Above Target")
    else:
        st.error("⚠️ Below Target")

with st.expander("Show run summary table"):
    st.dataframe(
        run_summary.rename(columns={
            "pass_rate": "Pass rate (this run) %",
            "cum_pass_stability": "Cumulative PASS stability %",
            "cum_fail_stability": "Cumulative FAIL stability %"
        }),
        use_container_width=True
    )

st.markdown("---")

# ==========================
# BOTTOM ROW (ล่าสุด)
# ==========================
col_left, col_mid, col_right = st.columns([1.1, 1.1, 0.8])

with col_left:
    st.subheader("Avg Duration by Feature (s)")
    avg_by_feat = (
        df.groupby("Feature", as_index=False)["Duration(s)"].mean()
          .sort_values("Duration(s)", ascending=False)
    )
    fig_avg = px.bar(
        avg_by_feat, x="Feature", y="Duration(s)", title="", text="Duration(s)"
    )
    fig_avg.update_traces(texttemplate="%{text:.1f}")
    fig_avg.update_layout(xaxis_title="", yaxis_title="Seconds")
    st.plotly_chart(fig_avg, use_container_width=True)

with col_mid:
    st.subheader("Pass / Fail by Feature")
    agg_pf = (df.groupby(["Feature", "Status"]).size().reset_index(name="Count"))
    fig_pf = px.bar(
        agg_pf, x="Feature", y="Count", color="Status",
        barmode="stack", color_discrete_map=COLORS, title=""
    )
    fig_pf.update_layout(xaxis_title="", yaxis_title="Count", legend_title="")
    st.plotly_chart(fig_pf, use_container_width=True)

with col_right:
    st.subheader("Summary")
    c1, c2 = st.columns(2)
    with c1:
        st.metric("Pass rate", f"{pass_rate:.1f}%", f"{pass_count} / {total_tests}")
        st.metric("Avg Duration", f"{avg_duration:.1f}s")
        st.metric("#Features", f"{n_features}")
    with c2:
        st.metric("Fail rate", f"{fail_rate:.1f}%", f"{fail_count} fails")
        st.metric("Slowest", f"{slow_dur:.1f}s" if pd.notna(slow_dur) else "-")
        st.metric("Total Tests", total_tests)
    st.caption(f"⏱ Slowest: {slow_name}")

st.markdown("---")

# ==========================
# TABLE + Filters (ล่าสุด)
# ==========================
st.subheader("All Tests (Filterable)")

if "Error Message" not in df.columns:
    df["Error Message"] = ""

feat_sel = st.multiselect("Filter by Feature", sorted(df["Feature"].unique().tolist()))
stat_sel = st.multiselect("Filter by Status", ["PASS", "FAIL"])
only_fail = st.checkbox("Show only FAILED rows", value=False)

table_df = df.copy()
if feat_sel:
    table_df = table_df[table_df["Feature"].isin(feat_sel)]
if stat_sel:
    table_df = table_df[table_df["Status"].isin(stat_sel)]
if only_fail:
    table_df = table_df[table_df["Status"] == "FAIL"]

table_df = table_df.sort_values(by=["TestID_num", "Feature"], ascending=[True, True])

cols_to_show = [
    "Test ID", "Feature", "Test Case Name", "Status",
    "Duration(s)", "Label", "Error Message"
]

st.dataframe(
    table_df[cols_to_show],
    use_container_width=True,
    height=380
)

# Error Preview (FULL)
fails = table_df[table_df["Status"] == "FAIL"].copy()
if not fails.empty:
    st.markdown("### 🔎 Error Preview (FULL)")
    picked = st.selectbox(
        "Select a failed test to preview its full error message",
        options=fails["Test ID"] + " — " + fails["Test Case Name"],
        index=0 if len(fails) > 0 else None,
        key="error_pick"
    )
    if picked:
        _picked_id = picked.split(" — ")[0]
        row = fails[fails["Test ID"] == _picked_id].iloc[0]
        st.write(f"**Test ID:** {row['Test ID']}  |  **Feature:** {row['Feature']}  |  **Status:** {row['Status']}")
        st.code(str(row["Error Message"]) if pd.notna(row["Error Message"]) else "—", language="text")
        st.download_button(
            "⬇️ Download this error as .txt",
            data=(str(row["Error Message"]) or "").encode("utf-8"),
            file_name=f"{row['Test ID']}_error.txt",
            mime="text/plain"
        )
else:
    st.info("No FAILED rows to preview.")
