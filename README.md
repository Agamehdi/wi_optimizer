# Rumaila — Integrated Water Injection Optimization

**One GitHub / Streamlit Cloud application with three pages:**

1. **Integrated Water Injection Optimizer** (`streamlit_app.py`) — combines the source data, models constraints and produces candidate operational changes.
2. **WI System Gap Analysis** (`pages/1_WI_Gap_Analysis.py`) — original Gap Analysis dashboard, driven by the supplied well and facility CSVs.
3. **Pattern Waterflood Analytics** (`pages/2_Pattern_Waterflood.py`) — October 22, 2026 analytical dashboard, using the original upload/VoW calculations and charts.

The original calculators are retained in `wi_gap.py` and in the original Pattern page. The optimizer reuses the well constraint calculations and Pattern analytical definitions in `pattern_core.py` (extracted unchanged from the supplied Pattern page apart from its standalone display initialization).

## Deploy via GitHub / Streamlit Community Cloud

1. Create a **private GitHub repository** if field data or any sensitive source needs storing. The repository can contain the **code only**; do not upload confidential field production/injection CSVs to GitHub.
2. Unzip this package and commit **all** files and folders at the **repository root** (especially `pages/`, `pattern_core.py`, `wi_gap.py`, `optimizer_core.py`, `requirements.txt`).
3. In Streamlit Cloud choose **New app**, select the repository/branch and specify the main file path **`streamlit_app.py`**.
4. Once it deploys, upload the three source files **in the main optimizer page**. The original analytical tools remain selectable via Streamlit's Pages navigation.
5. Review **Input QC**, run **⚡ Run connected-network optimization**, then inspect results and export the Excel action report.

Locally: `pip install -r requirements.txt` then `streamlit run streamlit_app.py`.

## Inputs

### 1. Well target / capacity CSV — original Gap input (required)

Required columns:

- `Well`, `CPS`, `Record Date`, `SW Manifold`
- `RMT Rate [bbl/day]`, `RMT Rate+OE [bbl/day]`
- `BMT Constrained Rate [bbl/day]`, `Agreed Target [bbl/day]`

Optional: `Reservoir`, `PWRI Manifold`, `Well Efficiency`, and a measured `Current Injection [bbl/day]` (or `Actual Injection Rate [bbl/day]`). One or more dated rows per well may be supplied; the latest eligible record at the selected date is used. Units for well rates are **bbl/day**, not mbd. The BMT data must have no missing values for wells to participate in optimization.

If measured actual injection is missing for even one assessed well, the optimizer uses the original Gap Analysis **modeled constrained injection** for the *entire* baseline and clearly labels it as modeled. It does **not** fabricate measured actual injection.

### 2. Facility capacities CSV — original Gap input (required)

Use the existing long form `FAC_NAME,Manifold,DATE,CAP`, with a blank Manifold on CPS/PWRI capacity-total rows. Example:

```csv
FAC_NAME,Manifold,DATE,CAP
CPS1,,2026-10-01,100
CPS1_FM01,CPS1_FM01,2026-10-01,55
CPS1_FM02,CPS1_FM02,2026-10-01,45
CPS7,,2026-10-01,70
CPS7_FM07,CPS7_FM07,2026-10-01,70
```

These are **fictional example capacities**, not field values. Default CAP units are **mbd**; you can change units to bbl/day in the sidebar. Effective-dated constraints are used for the selected date. PWRI remains separated per the original Gap routing option.

### 3. Pattern analytical Excel / CSV — original Pattern input (required)

Required: `Pattern Name` (or `Pattern`), `Produced Oil (stb)`, `Produced Water (stb)`, `Injected Water (stb)`.

Recommended: `Reservoir`, `Manifold`, and supported-oil fields if using a non-BP calculation. The source workbook worksheet can be selected. The application uses the original Pattern tool's **raw VoW** definition:

- BP analytical (default): `Bwi × Qo / (Bo × Qo + Bwp × Qw + Bg × Qg)`, with Bo defaults Mishrif 1.25, Main Pay 1.43, Upper Shale 1.36.
- Excel Oil Supported: `sum(Oil Supported) / sum(Injection)`.
- Excel Support Fraction or percentage proxy: from the original Pattern analytical approach.

Raw VoW is a screening *score*, not demonstrated marginal response. The original Pattern tool's 0–1 normalized rankings are **not** used in the optimizer objective.

### 4. Approved optional Well → Manifold connections CSV

**Required only to recommend physical manifold rerouting.** Download the CSV header template in the app:

```csv
Well,CPS,Allowed Manifold,Approved
INJ-001,CPS1,CPS1_FM02,Yes
INJ-002,CPS7,CPS7_FM07,No
```

These are fictional example well IDs. Only `Approved=Yes` same-CPS connections are valid. If you provide no mapping, the app will **change injection rates only within the existing well/manifold assignment**. It never assumes a well can move to another CPS, or that all manifolds within a CPS are automatically physically connected. PWRI routing is never altered by the optimizer.

### 5. Optional Well → Pattern mapping CSV

Use this for best-quality well-level scoring:

```csv
Well,Pattern,Reservoir,Weight
INJ-001,A-CPS1-P01,Mishrif,1
```

If one well supports multiple patterns, provide the individual fractional weights adding to exactly 1. In its absence, the model falls back to physical *manifold + reservoir*, physical manifold, and optionally CPS group coefficients. This is a **group-level proxy** (confidence 0.90, 0.75, 0.55, 0.35), not evidence for a given well's marginal oil response. Set the **minimum score confidence** accordingly. No scores are inferred from zero-injection patterns.

## What is optimized

The model solves a **mixed-integer linear program** using `scipy.optimize.milp`:

- Maximizes `Σ(optimized well injection × original well/Pattern raw VoW)` as a **screening index** (minus configurable change/reroute penalties).
- Enforces BMT well injectivity; optional RMT+OE or Agreed Target ceilings; configurable per-well increase/decrease limits; maximum manifold capacity; maximum CPS/PWRI capacity.
- Constrains water **independently within each CPS or PWRI**. `Conserve water` keeps each facility's original supply unchanged. `Additional supply` makes explicitly allotted extra supply available **only in the chosen CPS**. A CPS cannot silently borrow from another CPS.
- Prohibits unapproved connections, cross-CPS tie-ins, and split allocations (each injector uses just one manifold).
- If a well's VoW confidence is below the selected threshold, the model keeps its baseline rate unless the well's own capacity requires a reduction. Some infeasible cases require engineering review rather than a forced solution.
- Tracks any baseline injection that exceeds stated capacities (negative ullage) as a QC/operability issue.

The optimizer **does not** calculate network pressure drops, pump head, pressure fracture limits, injectivity decline on changed rates, reservoir pressure changes, sweep, interference, economic costs, or a validated increase in actual oil production. These require additional engineering input and validation. Treat the recommended values as **screening cases for well-by-well engineering approval**, not remote-control actions or immediately deployable operations.

## Outputs

The main page shows an optimization KPI scorecard; CPS/bar comparison; manifold rate redistribution; well-level ranked action statements (`Increase`, `Decrease`, or `Approved reroute`); a physically constrained max-rate estimate; data quality flags; and a multi-sheet Excel report (well actions, manifold actions, CPS actions, original gap results, Pattern VoW records, settings and QC). It provides source-context labels and flags when a rate basis is modeled versus actual.

## QA

`pytest -q` runs offline tests for source parsing, scoring, CPS isolation, conservation, unscored protection and approved manifold rerouting. **6 tests pass** against the packaged calculation engine in this build environment. The Streamlit UI is statically syntax-checked; a deployed browser session has **not** been run in this environment. Always verify the app against the actual Rumaila worksheets, especially mapping and facility labels.

## Package files

```text
streamlit_app.py             # Integrated optimizer UI
optimizer_core.py            # MILP; routing + well scoring
pattern_core.py              # Reused original Pattern calculation functions
wi_gap.py                    # Original Gap Analysis plus bare-manifold CSV handling
pages/
  1_WI_Gap_Analysis.py       # Original Gap dashboard, calls wi_gap.main()
  2_Pattern_Waterflood.py    # Original Pattern dashboard, unchanged
requirements.txt
README.md
tests/
  test_optimizer.py
  test_pipeline.py
```
