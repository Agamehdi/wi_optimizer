"""Rumaila Pattern Waterflood | upload-driven Streamlit dashboard.

Run locally: streamlit run streamlit_app.py
GitHub/Streamlit Cloud: deploy with streamlit_app.py + requirements.txt.
No field data is bundled in this source code.
"""

from __future__ import annotations

import io
import re
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


# -----------------------------------------------------------------------------
# Input definitions — these are matched without regard to case or punctuation.
# -----------------------------------------------------------------------------
FIELDS = {
    "Pattern": ["Pattern Name", "Pattern", "Pattern ID"],
    "Reservoir": ["Reservoir", "Reservoir Name", "Resevoir", "Resevoir Name", "Reservior", "Reservior Name", "Reservoir Type"],
    "Manifold": ["Manifold", "Manifold Name", "Injection Manifold", "Production Manifold", "Manifold ID", "Manifold Code"],
    "Oil": ["Produced Oil (stb)", "Produced Oil", "Oil Production (stb)", "Oil Volume"],
    "Water": ["Produced Water (stb)", "Produced Water", "Water Production (stb)"],
    "Injection": ["Injected Water (stb)", "Injected Water", "Water Injection (stb)", "Water Injected"],
    "VRR": ["VRR", "Voidage Replacement Ratio"],
    "Res Pressure": ["Average of Res P", "Average Reservoir Pressure", "Reservoir Pressure"],
    "Psat": ["Average of P Sat", "Average Saturation Pressure", "Bubble Point Pressure"],
    "Delta P": ["Delta P from P Sat", "Delta P from Psat", "Pressure Margin"],
    "FBHP Trend": ["Avg. FBHP Trend (psi/day)", "FBHP Trend (psi/day)", "Average FBHP Trend"],
    "Watercut Input": ["Watercut %", "Water Cut", "Watercut", "Water Cut %"],
    "Agreed Target": ["Agreed Target (stb)", "Agreed Target"],
    "RMT Target": ["RMT Target (stb)", "RMT Target"],
    "Actual Supported Oil": ["Oil Supported (stb)", "Supported Oil (stb)", "Actual Supported Oil (stb)", "Incremental Oil (stb)"],
    "Support Fraction": ["Oil Support Fraction", "Supported Fraction", "Support Factor", "Support Fraction"],
}
REQUIRED = ["Pattern", "Oil", "Water", "Injection"]
NUMERIC_FIELDS = [k for k in FIELDS if k not in ("Pattern", "Reservoir", "Manifold")]
VOL_STB_TO_M3 = 0.158987294928
UNIT_OPTIONS = {
    "stb": (1.0, "stb"),
    "Mstb": (1e-3, "Mstb"),
    "MMstb": (1e-6, "MMstb"),
    "m³": (VOL_STB_TO_M3, "m³"),
    "10³ m³": (VOL_STB_TO_M3 / 1_000, "10³ m³"),
}
DEFAULT_CPS_COLORS = {
    "CPS1": "#1677B9", "CPS2": "#DA6D24", "CPS3": "#159579",
    "CPS4": "#8E62A6", "CPS5": "#C43B5F", "CPS6": "#807026",
    "CPS7": "#18A9B4", "Unassigned": "#8A929C",
}


@dataclass
class Design:
    font: str = "Arial"
    text: str = "#182536"
    paper: str = "#FFFFFF"
    plot: str = "#FFFFFF"
    grid: str = "#E7EDF5"
    title_size: int = 19
    axis_size: int = 14
    tick_size: int = 11
    legend_size: int = 11
    line_width: float = 2.5
    marker_size: int = 13
    opacity: float = 0.83
    height: int = 570
    legend_pos: str = "Bottom"
    legend_x: float = 0.5
    legend_y: float = -0.27
    legend_anchor: str = "center"
    grid_on: bool = True


def keyify(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(s).lower())


def col_lookup(frame: pd.DataFrame, aliases: list[str]) -> Optional[str]:
    dic = {keyify(x): x for x in frame.columns}
    for term in aliases:
        if keyify(term) in dic:
            return dic[keyify(term)]
    return None


# Canonical labels used only for recognising a misnamed reservoir column.
# Other reservoir names are preserved when the workbook contains a valid header.
RESERVOIR_LABELS = {
    "mishrif": "Mishrif",
    "uppershale": "Upper Shale",
    "mainpay": "Main Pay",
}


def canonical_reservoir(value: object) -> str:
    if pd.isna(value) or not str(value).strip():
        return "Unspecified"
    original = str(value).strip()
    return RESERVOIR_LABELS.get(keyify(original), original)


def detect_reservoir_column(frame: pd.DataFrame) -> Optional[str]:
    """Find a mislabeled reservoir column by values, not just the header.

    Some Rumaila files have B1='Mishrif' instead of B1='Reservoir', although
    the cells below contain Mishrif, Upper Shale and Main Pay. Requiring a
    strong concentration of *known reservoir labels* avoids mistaking Pattern
    names or other text columns for a reservoir.
    """
    candidates: list[tuple[float, int, str]] = []
    for col in frame.columns:
        if keyify(col) in {"pattern", "patternname", "patternid"}:
            continue
        values = frame[col].dropna()
        if values.empty:
            continue
        strings = values.astype(str).str.strip()
        strings = strings.loc[strings.ne("")]
        if len(strings) < 3:
            continue
        recognized = strings.map(lambda v: keyify(v) in RESERVOIR_LABELS)
        confidence = float(recognized.mean())
        known_kinds = strings.loc[recognized].map(keyify).nunique()
        is_reservoir_header = keyify(col) in RESERVOIR_LABELS
        if confidence >= 0.80 and (is_reservoir_header or known_kinds >= 2):
            candidates.append((confidence, int(recognized.sum()), str(col)))
    if not candidates:
        return None
    return max(candidates, key=lambda item: (item[0], item[1]))[2]


def to_numeric(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")
    cleaned = (series.astype("string").str.replace(",", "", regex=False)
               .str.replace("%", "", regex=False)
               .str.replace("−", "-", regex=False).str.strip())
    return pd.to_numeric(cleaned, errors="coerce")


def get_cps(pattern: str) -> tuple[str, str]:
    text = str(pattern).upper()
    match = re.search(r"(?<![A-Z0-9])CPS[\s_-]*(\d+)(?!\d)", text)
    if match:
        return f"CPS{int(match.group(1))}", ""
    # One known typo in the October source: B-CSP4-i9-R-856.
    typo = re.search(r"(?<![A-Z0-9])CSP[\s_-]*(\d+)(?!\d)", text)
    if typo:
        return f"CPS{int(typo.group(1))}", "CSP/CPS spelling normalised; check source"
    return "Unassigned", "CPS not found in pattern name"


@st.cache_data(show_spinner=False)
def sheet_names(file_bytes: bytes, extension: str) -> list[str]:
    if extension == "csv":
        return ["CSV"]
    excel = pd.ExcelFile(io.BytesIO(file_bytes))
    return excel.sheet_names


@st.cache_data(show_spinner=False)
def read_sheet(file_bytes: bytes, extension: str, name: str) -> pd.DataFrame:
    if extension == "csv":
        return pd.read_csv(io.BytesIO(file_bytes))
    return pd.read_excel(io.BytesIO(file_bytes), sheet_name=name)


def clean_manifold(value: object) -> str:
    """Normalise explicit Manifold identifiers without inferring them from Pattern names."""
    if pd.isna(value):
        return "Unmapped"
    raw = str(value).strip()
    if not raw or raw.lower() in ("nan", "none", "unmapped", "unknown", "n/a", "-", "tbc", "not assigned"):
        return "Unmapped"
    # Operational comments are not real manifolds; keep them visible in source QC.
    if any(phrase in raw.lower() for phrase in ("not converted", "not an injector", "to be converted", "no manifold")):
        return "Unmapped"
    raw = re.sub(r"\s*[/\\]\s*", "/", raw.strip()).upper()
    raw = re.sub(r"\s+", " ", raw)
    # e.g., CPS-2 / FM09, CPS2/FM09, cps-2/SM01 => CPS-2/FM09
    match = re.fullmatch(r"CPS[\s_-]*(\d+)\s*/\s*([A-Z0-9-]+)", raw)
    if match:
        return f"CPS-{int(match.group(1))}/{match.group(2)}"
    return raw


def manifold_cps(name: str) -> Optional[str]:
    """CPS declared by the Manifold ID; no Pattern-name guess when present."""
    m = re.match(r"^CPS[\s_-]*(\d+)\s*/", str(name), flags=re.I)
    return f"CPS{int(m.group(1))}" if m else None


def manifold_rollup(frame: pd.DataFrame, keys: Optional[list[str]] = None) -> pd.DataFrame:
    """Group by physical Manifold ID, which already includes CPS e.g. CPS-2/FM09.

    Pattern CPS may disagree with the supplied Manifold CPS. Do not split the
    *same* Manifold into multiple points on account of that mismatch.
    """
    keys = ["Manifold"] + ([k for k in (keys or []) if k != "Manifold"])
    result = rollup(frame, keys)
    if result.empty:
        return result
    pattern_cps_sets = frame.groupby("Manifold")["CPS"].agg(lambda ss: sorted(set(ss.dropna())))
    result["CPS"] = result["Manifold"].map(
        lambda mn: manifold_cps(mn) or (
            pattern_cps_sets[mn][0] if len(pattern_cps_sets[mn]) == 1 else "Mixed"
        )
    )
    return result


def normalise_vow(frame: pd.DataFrame, reference: float) -> pd.DataFrame:
    """Display-only 0–1 score. Never overwrite the observed volume ratio."""
    out = frame.copy()
    if out.empty:
        out["VoW Normalized"] = pd.Series(dtype=float)
        return out
    raw = out["VoW Technical"]
    out["VoW Normalized"] = (raw / reference).clip(lower=0, upper=1)
    return out


def vow_normalization_reference(frame: pd.DataFrame) -> float:
    """Current selected group reference, NEVER pooled over different levels.

    Example: when CPS is selected, max CPS ratio becomes 1 even if a
    Pattern elsewhere has a much higher raw VoW. Calculated on the
    *currently filtered* rows at the level actually displayed.
    """
    if frame.empty or "VoW Technical" not in frame.columns:
        return 1.0
    vals = pd.to_numeric(frame["VoW Technical"], errors="coerce")
    vals = vals.loc[np.isfinite(vals) & vals.gt(0)]
    return float(vals.max()) if len(vals) else 1.0


def map_manifolds(frame: pd.DataFrame, mapping: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Safely apply one Manifold per Pattern+Reservoir; never guess from a name.

    Supports a Pattern-only map only when that Pattern is unambiguous across
    reservoirs. Conflicting assignments stay unmapped with a QC warning.
    """
    df = frame.copy()
    notes: list[str] = []
    pc = col_lookup(mapping, ["Pattern", "Pattern Name", "Pattern ID"])
    mc = col_lookup(mapping, FIELDS["Manifold"])
    rc = col_lookup(mapping, FIELDS["Reservoir"])
    if not pc or not mc:
        return df, ["Mapping requires Pattern (or Pattern Name) and Manifold columns."]
    cols = [pc, mc] + ([rc] if rc else [])
    ref = mapping[cols].copy()
    ref.columns = ["Pattern", "Manifold"] + (["Reservoir"] if rc else [])
    ref["Pattern"] = ref["Pattern"].fillna("").astype(str).str.strip()
    ref["Manifold"] = ref["Manifold"].map(clean_manifold)
    ref = ref.loc[ref["Pattern"].ne("") & ref["Manifold"].ne("Unmapped")].copy()
    if ref.empty:
        return df, ["No assigned Manifold entries found in mapping; cross-plot requires a mapping."]
    df["_map_key"] = df["Pattern"].str.casefold()
    ref["_map_key"] = ref["Pattern"].str.casefold()
    join = ["_map_key"]
    if rc:
        df["_res_key"] = df["Reservoir"].str.casefold()
        ref["_res_key"] = ref["Reservoir"].fillna("").astype(str).str.strip().map(canonical_reservoir).str.casefold()
        join.append("_res_key")
    else:
        amb = df.groupby("_map_key")["Reservoir"].nunique()
        ambiguous = set(amb.loc[amb.gt(1)].index)
        if ambiguous:
            ref = ref.loc[~ref["_map_key"].isin(ambiguous)].copy()
            notes.append(f"{len(ambiguous)} Pattern names span multiple reservoirs; include a Reservoir column in the mapping to disambiguate them.")
    conflicts = ref.groupby(join, dropna=False)["Manifold"].nunique()
    conflict_keys = conflicts.loc[conflicts.gt(1)]
    if len(conflict_keys):
        notes.append(f"{len(conflict_keys)} conflicting Pattern→Manifold assignments ignored. Each Pattern needs one assignment; split multi-Manifold allocation upstream.")
        bad = ref.set_index(join).index.isin(conflict_keys.index)
        ref = ref.loc[~bad]
    ref = ref.drop_duplicates(join, keep="first")
    if ref.empty:
        return frame.copy(), notes
    assigned = df.merge(ref[join + ["Manifold"]].rename(columns={"Manifold": "_assigned"}), on=join, how="left", sort=False)
    changed = assigned["_assigned"].notna()
    assigned.loc[changed, "Manifold"] = assigned.loc[changed, "_assigned"]
    assigned = assigned.drop(columns=["_map_key", "_res_key", "_assigned"], errors="ignore")
    notes.append(f"{int(changed.sum())} source rows matched to a Manifold.")
    return assigned, notes


def display_volume(values: pd.Series, factor: float, months: float, mode: str, days: int) -> pd.Series:
    divisor = days if mode == "Average per day" else (months if mode == "Average per month" else 1)
    return values * factor / divisor


def attach_display(frame: pd.DataFrame, factor: float, months: float, mode: str, days: int) -> pd.DataFrame:
    frame = frame.copy()
    for original, name in (("Oil", "Oil Display"), ("Water", "Water Display"),
                           ("Injection", "Injection Display"), ("Supported", "Supported Display")):
        frame[name] = display_volume(frame[original], factor, months, mode, days)
    frame["Supported mbd"] = frame["Supported"] / (days * 1000.0)
    frame["Injection mbd"] = frame["Injection"] / (days * 1000.0)
    return frame


def standardise(raw: pd.DataFrame) -> tuple[pd.DataFrame, list[str], dict[str, str]]:
    # Detect before discarding unnamed columns: the reservoir header is
    # sometimes overwritten (e.g., B1='Mishrif') or left blank in Excel.
    matched = {k: col_lookup(raw, aliases) for k, aliases in FIELDS.items()}
    if matched["Reservoir"] is None:
        matched["Reservoir"] = detect_reservoir_column(raw)
    missing = [k for k in REQUIRED if matched[k] is None]
    if missing:
        return pd.DataFrame(), missing, {k: v for k, v in matched.items() if v}
    df = pd.DataFrame(index=raw.index)
    for key in FIELDS:
        if matched[key]:
            df[key] = raw[matched[key]]
        else:
            # Missing Reservoir must never silently become Mishrif.
            df[key] = "Unspecified" if key == "Reservoir" else ("Unmapped" if key == "Manifold" else np.nan)
    df["Pattern"] = df["Pattern"].astype("string").str.strip()
    df = df[df["Pattern"].notna() & df["Pattern"].ne("")].copy()
    df["Reservoir"] = df["Reservoir"].map(canonical_reservoir)
    df["Manifold"] = df["Manifold"].map(clean_manifold)
    for key in NUMERIC_FIELDS:
        df[key] = to_numeric(df[key])
    df["Manifold CPS"] = df["Manifold"].map(manifold_cps)
    # User files may store fractions [0,1] or percentage points [0,100].
    for pct_col in ["Watercut Input", "Support Fraction"]:
        non_null = df[pct_col].dropna()
        if len(non_null) and float(non_null.quantile(0.95)) > 1.0:
            df[pct_col] = df[pct_col] / 100.0
    cps = df["Pattern"].map(get_cps)
    df["CPS"] = cps.map(lambda pair: pair[0])
    df["CPS QC"] = cps.map(lambda pair: pair[1])
    df["Manifold CPS mismatch"] = (
        df["Manifold CPS"].notna() & df["CPS"].ne("Unassigned") & df["Manifold CPS"].ne(df["CPS"])
    )
    df["Source Row"] = df.index.astype(int) + 2
    df["Liquid"] = df["Oil"].fillna(0) + df["Water"].fillna(0)
    df["Watercut Calculated"] = np.where(df["Liquid"] > 0, df["Water"].fillna(0) / df["Liquid"], np.nan)
    df["Oil vs Agreed"] = df["Oil"] - df["Agreed Target"]
    df["Oil vs RMT"] = df["Oil"] - df["RMT Target"]
    return df.reset_index(drop=True), [], {k: v for k, v in matched.items() if v}


BP_ANALYTICAL_METHOD = "BP Analytical (fluid balance)"
DEFAULT_BO = {"Mishrif": 1.25, "Main Pay": 1.43, "Upper Shale": 1.36}


def analytical_waterflood(frame: pd.DataFrame, bo_by_reservoir: dict[str, float],
                          bwi: float = 1.0, bwp: float = 1.0,
                          delta_gor: float = 0.0, bg: float = 0.005) -> pd.DataFrame:
    """BP material-balance, ideal-displacement/constant-pressure limit (slide eq. 10/11).

    VoW_BP = Bwi*Qo / (Qo*Bo + Qw*Bwp + Qg*Bg)
           = Bwi / (Bo + WC/(1-WC)*Bwp + delta_gor*Bg).
    Qo,Qw are period stock-tank stb; Qg=delta_gor*Qo scf, Bg rb/scf.
    Returned 'BP Supported' is an *ideal-model equivalent*, NOT observed incremental oil.
    An injection-weighted rollup uses SUM(BP Supported)/SUM(Injection).
    """
    out = frame.copy()
    out["Bo"] = out["Reservoir"].map(bo_by_reservoir)
    oi = pd.to_numeric(out["Oil"], errors="coerce")
    wa = pd.to_numeric(out["Water"], errors="coerce")
    inj = pd.to_numeric(out["Injection"], errors="coerce")
    bo = pd.to_numeric(out["Bo"], errors="coerce")
    gas_reservoir_volume = oi * float(delta_gor) * float(bg)
    total_reservoir_offtake = oi * bo + wa * float(bwp) + gas_reservoir_volume
    valid = (oi.notna() & wa.notna() & bo.notna() &
             oi.ge(0) & wa.ge(0) & bo.gt(0) & total_reservoir_offtake.gt(0))
    analytical = pd.Series(np.nan, index=out.index, dtype=float)
    analytical.loc[valid] = (float(bwi) * oi.loc[valid] /
                              total_reservoir_offtake.loc[valid])
    # For oil-free water production and positive total off-take the equation returns 0.
    out["BP VoW (raw)"] = analytical
    out["BP Offtake (rb)"] = total_reservoir_offtake.where(valid)
    out["BP Oil Supported (stb)"] = (analytical * inj).where(inj.gt(0))
    out["BP Oil Supported (stb)"] = out["BP Oil Supported (stb)"].where(~inj.eq(0), 0.0)
    out["BP Gas Term (rb)"] = gas_reservoir_volume
    return out


def support_scenario(frame: pd.DataFrame, method: str, share: float) -> pd.DataFrame:
    df = frame.copy()
    if method == "Excel: Oil Supported":
        df["Supported"] = df["Actual Supported Oil"]
    elif method == "Excel: Support Fraction":
        df["Supported"] = df["Oil"] * df["Support Fraction"]
    else:
        df["Supported"] = df["Oil"] * share / 100.0
    return df


def safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return numerator.div(denominator.where(denominator > 0)).replace([np.inf, -np.inf], np.nan)


def rollup(frame: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """CRITICAL: aggregate numerator/denominator FIRST, then calculate VoW."""
    if frame.empty:
        return pd.DataFrame()
    sum_cols = ["Oil", "Water", "Injection", "Supported", "Agreed Target", "RMT Target"]
    # Keep analytical prediction as a separate non-averaged numerator at all levels.
    if "BP Oil Supported (stb)" in frame.columns:
        sum_cols += ["BP Oil Supported (stb)"]
    if "Existing Supported" in frame.columns:
        sum_cols += ["Existing Supported"]
    aggs: dict[str, object] = {c: (lambda x: x.sum(min_count=1)) for c in sum_cols}
    for col in ("VRR", "Res Pressure", "Psat", "Delta P", "FBHP Trend"):
        aggs[col] = "mean"  # Informational only; NOT a recomputed reservoir-condition VRR.
    if "Pattern" not in keys:
        aggs["Pattern"] = "nunique"
    result = frame.groupby(keys, dropna=False, as_index=False).agg(aggs)
    if "Pattern" not in keys:
        result = result.rename(columns={"Pattern": "Patterns"})
    else:
        result["Patterns"] = 1
    result["Liquid"] = result["Oil"].fillna(0) + result["Water"].fillna(0)
    result["Watercut"] = safe_ratio(result["Water"], result["Liquid"])
    result["VoW Technical"] = safe_ratio(result["Supported"], result["Injection"])
    if "BP Oil Supported (stb)" in result:
        result["BP VoW (raw)"] = safe_ratio(result["BP Oil Supported (stb)"], result["Injection"])
    if "Existing Supported" in result:
        result["Existing VoW (raw)"] = safe_ratio(result["Existing Supported"], result["Injection"])
    # Raw ratio is kept for engineering QC; 0–1 score added later for charts.
    result["Production vs Agreed"] = result["Oil"] - result["Agreed Target"]
    result["Production vs RMT"] = result["Oil"] - result["RMT Target"]
    result["VoW Valid"] = result["Injection"].gt(0) & result["Supported"].notna()
    return result


def cps_sort(cps: str) -> tuple[int, str]:
    m = re.search(r"\d+", str(cps))
    return (int(m.group()) if m else 1000, str(cps))


def pretty(value: float, digits: int = 2) -> str:
    return "—" if pd.isna(value) else f"{value:,.{digits}f}"


def apply_style(fig: go.Figure, design: Design, title: str, x_label: str = "", y_label: str = "", *, show_legend: bool = True, height: Optional[int] = None) -> go.Figure:
    if design.legend_pos == "Hidden" or not show_legend:
        legend = dict()
        top_margin, bottom_margin, right_margin = 95, 100, 75
    elif design.legend_pos == "Top":
        legend = dict(orientation="h", x=0.5, y=1.14, xanchor="center", yanchor="bottom")
        top_margin, bottom_margin, right_margin = 135, 95, 70
    elif design.legend_pos == "Right":
        legend = dict(orientation="v", x=1.02, y=1, xanchor="left", yanchor="top")
        top_margin, bottom_margin, right_margin = 95, 100, 175
    elif design.legend_pos == "Custom":
        legend = dict(orientation="h", x=design.legend_x, y=design.legend_y, xanchor=design.legend_anchor, yanchor="middle")
        top_margin, bottom_margin, right_margin = 130, 150, 95
    else:
        legend = dict(orientation="h", x=0.5, y=-0.28, xanchor="center", yanchor="top")
        top_margin, bottom_margin, right_margin = 95, 165, 75
    legend.update(font=dict(size=design.legend_size, color=design.text), bgcolor="rgba(0,0,0,0)")
    fig.update_layout(
        title=dict(text=f"<b>{title}</b>", x=0.01, y=0.98, xanchor="left", yanchor="top", font=dict(size=design.title_size, color=design.text)),
        font=dict(family=design.font, color=design.text, size=design.tick_size),
        plot_bgcolor=design.plot, paper_bgcolor=design.paper,
        height=height or design.height, hovermode="closest",
        margin=dict(l=90, r=right_margin, b=bottom_margin, t=top_margin),
        legend=legend, showlegend=show_legend and design.legend_pos != "Hidden",
        hoverlabel=dict(font_size=max(11, design.tick_size), font_family=design.font),
    )
    for axis, label in (("xaxes", x_label), ("yaxes", y_label)):
        kwargs = {
            "title_text": label,
            "title_font": dict(size=design.axis_size, color=design.text),
            "title_standoff": 18,
            "tickfont": dict(size=design.tick_size, color=design.text),
            "showgrid": design.grid_on,
            "gridcolor": design.grid,
            "gridwidth": 1,
            "zeroline": False,
            "linecolor": design.grid,
            "showline": True,
            "automargin": True,
        }
        getattr(fig, f"update_{axis}")(**kwargs)
    fig.update_traces(selector=dict(type="scatter"), line=dict(width=design.line_width))
    return fig


def plot(fig: go.Figure, key: str):
    st.plotly_chart(fig, use_container_width=True, key=key,
                    config={"displaylogo": False, "scrollZoom": False, "toImageButtonOptions": {"format": "png", "filename": key, "scale": 2}})


def add_secondary_vow_reference_scale(fig: go.Figure, design: Design,
                                      metric: str, raw_max: float) -> go.Figure:
    """Show an optional right-side VoW scale, WITHOUT plotting another series.

    It is an independent visual reference scale; the Oil Supported dots stay
    on the left Y-axis. It does not imply dots lie at their VoW Y-position.
    The empty, invisible Plotly trace is only an axis registration trick so
    plotly.js renders the right-hand axis; it cannot add plotted information.
    """
    is_normalised = metric == "VoW — normalised (0–1)"
    title = "VoW (normalised 0–1)" if is_normalised else "VoW (raw stb/stb)"
    upper = 1.0 if is_normalised else max(float(raw_max) * 1.05, 0.01)

    # Empty invisible placeholder makes the secondary scale render reliably.
    # There are no VoW coordinates, markers, lines, annotations or tooltips.
    fig.add_trace(go.Scatter(
        x=[None], y=[None], yaxis="y2", mode="markers",
        marker=dict(opacity=0), showlegend=False, hoverinfo="skip",
        name="", visible=True,
    ))
    fig.update_layout(yaxis2=dict(
        title=dict(text=title, font=dict(size=design.axis_size, color=design.text), standoff=14),
        overlaying="y", side="right", anchor="x", range=[0, upper],
        showgrid=False, zeroline=False, showline=True, linecolor=design.grid,
        tickfont=dict(size=design.tick_size, color=design.text),
        tickmode="array" if is_normalised else "auto",
        tickvals=[0, 0.2, 0.4, 0.6, 0.8, 1] if is_normalised else None,
        tickformat=".1f" if is_normalised else ".3g",
        ticks="outside", automargin=True,
    ))
    # Leave room to the right of this independent axis for the VoW colour bar.
    fig.update_layout(margin_r=295)
    return fig


def scale_volume(x: pd.Series, factor: float, months: float, monthly: bool) -> pd.Series:
    return x * factor / (months if monthly else 1)


def file_download_excel(patterns: pd.DataFrame, cps: pd.DataFrame, manifold: pd.DataFrame, detailed: pd.DataFrame, qc: pd.DataFrame, method: str, months: int, days: int, share: float, exclude_zero: bool, norm_references: dict[str, float], bo_by_reservoir: dict[str, float], bwi: float, bwp: float, delta_gor: float, bg: float) -> bytes:
    how = ("BP Analytical — FVF-based material balance" if method == BP_ANALYTICAL_METHOD else
           ("Excel supported oil" if method == "Excel: Oil Supported" else
            ("Excel support fraction × produced oil" if method == "Excel: Support Fraction" else f"Produced oil × {share:g}%")))
    guide = pd.DataFrame([
        ("Scope", f"Uploaded period: {months} months. Raw data are total volumes in stb."),
        ("Oil Supported", how),
        ("Selected method", method),
        ("BP slide Eq. 10", "VoW_BP = Bwi*Qo / (Bo*Qo + Bwp*Qw + Bg*Qg), volumes matched over reporting period."),
        ("BP slide Eq. 11", "VoW_BP = Bwi/(Bo + Bwp*WC/(1-WC) + Bg*(Rp-Rs))."),
        ("BP analytical support", "Model-equivalent Oil Supported = VoW_BP * actual injected water. Not measured incremental oil."),
        ("BP assumptions", "Ideal displacement; constant pressure; no aquifer/gas injection/expansion in simplified Eq. 10/11."),
        ("Bwi injected water FVF (rb/stb)", f"{bwi:.6f}"),
        ("Bwp produced water FVF (rb/stb)", f"{bwp:.6f}"),
        ("Gas ratio Rp-Rs (scf/stb)", f"{delta_gor:.6f}"),
        ("Bg gas FVF (rb/scf)", f"{bg:.8f}"),
        *[(f"Bo {k} (rb/stb)", f"{v:.6f}") for k, v in bo_by_reservoir.items()],
        ("Exclude zero-injection supported proxy", f"{exclude_zero}; applies to Produced Oil proxy methods only"),
        ("Raw Value of Water", "Sum(Oil Supported, stb) / Sum(Injected Water, stb), can exceed 1."),
        ("Normalized VoW (0–1)", "Raw group VoW / maximum raw VoW WITHIN THAT GROUPING LEVEL. Pattern, CPS and Manifold have separate references. Recomputed when filters change."),
        ("Pattern normalization reference (stb/stb)", f"{norm_references['Pattern']:.8f}"),
        ("CPS normalization reference (stb/stb)", f"{norm_references['CPS']:.8f}"),
        ("Manifold normalization reference (stb/stb)", f"{norm_references['Manifold']:.8f}"),
        ("Cross-plot breakdowns", "X-axis can show uploaded injected-water period volume, actual injected-water average mbd, or normalised VoW. Colour / bubble sizing may show VoW. If a color option changes point granularity (e.g. CPS x Reservoir), normalisation uses the maximum among those displayed points."),
        ("Normalization warning", "0–1 VoW is a relative score, NOT stb/stb and not the physically observed injection-to-oil ratio."),
        ("Aggregate VoW", "Ratio of sums, never an average of pattern VoW ratios."),
        ("Zero injection", "VoW is blank / undefined, not zero; retained in QC."),
        ("Interpretation", "Produced oil proxy is NOT demonstrated incremental oil caused by water injection."),
        ("CPS", "CPS number extracted from Pattern name, e.g. A-CPS2..., CPS-2-N."),
        ("CPS typo", "CSP4 variant mapped to CPS4; check original record."),
        ("Manifold data", "Uses the uploaded Manifold column directly; optional mapping only if missing/incomplete."),
        ("Manifold aggregation", "Group by unique Manifold ID (e.g. CPS-2/FM09); physical Manifold CPS takes priority over mismatched CPS in Pattern name. Mismatches flagged in QC."),
        ("Daily rate (mbd)", f"Full-period stb / {days} days / 1000; mbd means thousand stb/day."),
        ("VoW > 2", "Possible when numerator exceeds twice denominator; not a recovery factor. Confirm volumes, allocations and reporting periods."),
        ("VRR", "Reported source VRR is informational. Group averages are not field-corrected VRR."),
    ], columns=["Topic", "Rule / assumption"])
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        patterns.to_excel(writer, sheet_name="Pattern_Ranking", index=False)
        cps.to_excel(writer, sheet_name="CPS_Ranking", index=False)
        manifold.to_excel(writer, sheet_name="Manifold_Ranking", index=False)
        detailed.to_excel(writer, sheet_name="Calculated_Data", index=False)
        qc.to_excel(writer, sheet_name="QC", index=False)
        guide.to_excel(writer, sheet_name="Guideline", index=False)
        for sheet in writer.sheets.values():
            sheet.freeze_panes = "A2"
            sheet.auto_filter.ref = sheet.dimensions
            sheet.row_dimensions[1].height = 25
            for cell in sheet[1]:
                from copy import copy
                font_style = copy(cell.font)
                font_style.bold = True
                font_style.color = "FFFFFF"
                cell.font = font_style
                from openpyxl.styles import PatternFill
                cell.fill = PatternFill(fill_type="solid", fgColor="173252")
            for col in sheet.columns:
                col_cells = list(col)
                max_len = max(len(str(c.value or "")) for c in col_cells[:250])
                from openpyxl.utils import get_column_letter
                sheet.column_dimensions[get_column_letter(col_cells[0].column)].width = min(48, max(13, max_len + 2))
    return buffer.getvalue()
