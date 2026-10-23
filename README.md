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

`pytest -q` runs offline tests for source parsing, scoring, CPS isolation, conservation, unscored protection and approved manifold rerouting. **22 tests pass** against the packaged calculation engine in this build environment. The Streamlit UI is statically syntax-checked; a deployed browser session has **not** been run in this environment. Always verify the app against the actual Rumaila worksheets, especially mapping and facility labels.

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

## V2: Original vs Optimized — automatic charting

After running **⚡ Run connected-network optimization**, open the **📊 Original vs Optimized** tab.
The tab calculates every displayed number directly from the optimizer's well-level
Original (`baseline`) and Optimized (`Optimized injection`) result. It does **not**
load demo/synthetic scenarios, overwrite the real data, or assume water can move
from one CPS to another. Main views:

1. CPS and Manifold **Target Delivery Gap** before and after. Select RMT,
   RMT+OE or Agreed Target, and filter CPS/PWRI.
2. **Structural** facility shortfall separately, based on unchanged facility
   capacity and target. Rate reallocation alone cannot fix capacity shortfall.
3. **Injection vs Oil-Supported proxy** scatter/connecting lines at CPS and
   CPS×Manifold, plus weighted **raw VoW vs Oil-Supported proxy**.
4. Pattern-level before/after **only** with approved Well→Pattern mapping;
   otherwise historic pattern input shown as a reference without fabricated After.
5. CPS and Manifold **Ullage** before and after, including negative baseline
   ullage if source injection is inconsistent with dated facility capacities.
6. Recomputed **Capacity Gap Table** by CPS and manifold; after-reroute targets
   and BMT well capacities move with the approved rerouted well to its new MF.
7. Download a scenario comparison workbook, or a standalone interactive
   HTML report with all currently selected Plotly charts.

**Technical interpretation:** Supported oil is fixed raw VoW coefficient ×
well allocation and has the units of an *oil-supported proxy*, not incremental
future oil forecast. It does not assume higher rates improve VoW automatically.
Calculated supported oil excludes unmatched/unscored injection volumes. Score
coverage is displayed, and unmet targets can be genuine operational decisions.
The baseline may be a **modelled** constrained target if a complete Current
Injection column is not provided; in this case the comparisons must not be
represented as historical measured-versus-optimized performance.

A full `Well,Pattern,Reservoir,Weight` mapping can be uploaded to generate
Pattern-level Original/Optimized; mapping weights per well must sum to 1. Without
this file, the tool does not claim pattern redistribution accuracy.

## V2.1 bug fix — 23 October 2026

Resolved **`Cannot compute before/after comparisons: Grouper for 'CPS' not 1-dimensional`**.
The parser intentionally retains the input's original `CPS` column, and the
comparison module previously renamed `facility` to `CPS` on the whole input
frame, silently creating duplicate `CPS` column labels. The comparison now
constructs an explicit, unique CPS / manifold routing frame from the solved
well rows, preserving the original reported values while avoiding duplicate
column names.

Also fixed a premature reference to the `mf` dataframe in the Comparison UI
(that would previously have caused an exception immediately after the groupby
fix). Optional source `Reservoir` values are now handled more consistently.

**To update:** replace all Python files with this package (or at minimum
`comparison_views.py`, `comparison_ui.py`, and `streamlit_app.py`), retaining
`pages/`, `templates/`, `pattern_core.py`, `wi_gap.py`, `optimizer_core.py`,
and `requirements.txt`. Redeploy Streamlit Cloud and click Run optimization
again; if a previous session caches stale state, refresh/restart the app.

**Verification (V2.1):** 17 offline tests passed, including a representative full
source parser → MILP optimization → comparison flow and a non-browser mocked
Streamlit render of all plots. Real private Rumaila inputs and the live
Streamlit deployment have not yet been exercised in this environment.


## V2.2 — Reference-style Gap Summary (Original / Optimized)

In the **📊 Original vs Optimized** tab, open **Capacity Gap & Tables**.
At the TOP of the tab, three new views have been added, without changing the
existing charts, filters, optimization model or other pages:

- **Original — Before**: grouped, screenshot-style Gap Analysis summary using
  the *selected dated well baseline*.
- **Optimized — After**: exactly the same layout and units using the solved
  optimized injection and any explicitly approved within-CPS reroutes.
- **Changes by Manifold**: original / optimized injection, injection delta,
  target-delivery gap reduction, remaining manifold ullage and physical MF
  target shortfall delta. A KPI strip shows overall improvements.

Both grouped tables include Reservoir (RMT, RMT+OE), Well (BMT capacity and
Agreed), manifold limits/allocated rates, CPS maximum/allocated rates,
"High Gap", signed well deliverability constraint, manifold and CPS capacity
shortfalls, and additional Decision Impact columns (delivery gap and ullage).
CPS-only rates/capacities appear once per CPS in the HTML report, and are
**counted only once** in the TOTAL rows. Rates are displayed in mbd.

**Technical interpretation:** The source Gap report's original RMT/OE
injection columns represent independently calculated target scenarios.
The IIC and INJ_RATE @ OE fields in these **optimization comparison tables**
are repurposed to show the *same original/optimized selected allocation*;
this is intentional, disclosed immediately below the table, and must **not**
be described as a fresh OE-only prediction. Physical CPS gap stays unchanged
when only rates are reallocated. A manifold physical gap can change if an
approved reroute moves a well and its full target/BMT capacity. Missing BMT
assessment makes that manifold's physical well-constraint gap **n.a.**

The **Export Original vs Optimized Excel** button now also includes:
`Gap_Summary_Original`, `Gap_Summary_Optimized`, and
`Gap_Manifold_Changes`. Existing report worksheets remain unchanged.

**Required new module:** `gap_summary.py`. Add this file to the repository
root alongside `comparison_ui.py` and `comparison_views.py`.

**Verification (V2.2):** 22 offline tests passed, including the full existing
17-test suite and five new summary tests for totals, signed gaps, missing BMT
assessment, screenshot-style headers, and target attribution during rerouting.
Real Rumaila inputs and Streamlit Cloud deployment still require field QC.

## V2.3 — Exact Original / After-Optimized Gap Analysis table format

The original **Original vs Optimized → Capacity Gap & Tables** tab now reproduces
**the original 18-column Gap Analysis screenshot layout**, once for Original and
once for After Optimized:

- 2-row white grouped headers (RESERVOIR, WELL, MANIFOLD, CPS, FINAL INJ,
  INJ_RATE @ OE, Capacity Gap_TABLE)
- all six capacity-gap columns, with legacy pastel green / yellow / orange
  threshold coloring
- vertically merged CPS and CPS-level rate/capacity/shortfall cells
- original manifold naming, one-decimal mbd format and bold TOTAL row
- selected analysis snapshot date included in the title
- horizontal scrolling so the full reference layout works on smaller screens

No optimizer maths, pattern calculations, scatter plots, ullage plots, input
handling, or Excel export layout was changed. The separate **Changes by
Manifold** tab remains available; the extra decision columns are no longer
embedded in the 18-column reference table.

To verify: `python -m pytest -q` — **24 tests** pass in this build. A browser
rendering of the exact HTML layout has also been inspected against the reference
screenshot with synthetic regression data; actual field input/output should still
be QCed in the deployed Streamlit app.

## V2.4 — TypeError handling and comparison UI regression checks (2026-10-23)

The user-provided Streamlit Cloud traceback ended at the call to
`comparison_ui.render_comparison(...)` and omitted the final exception message
and inner traceback line. That evidence alone does **not** identify the exact
failing statement or prove a particular root cause.

This update keeps the optimization engine, its constraints, both legacy
calculators, and the Original/Optimized presentation unchanged when data is
valid. It adds:

- An error boundary specifically around the 18-column screenshot-format tables.
  If rendering one of these raises a `TypeError` (or malformed-data error), the
  **other comparison plots still render** and the error type/message are shown.
- An error boundary around the three screenshot-specific Excel worksheets:
  other Excel worksheets are still downloadable, and the file contains
  `Gap_Summary_QC` with the message if the new summary could not be created.
- A top-level guard in the app's `Original vs Optimized` tab so that a failure in
  any remaining comparison widget does not take down the other tabs. The actual
  exception type and message appear in the tab and its traceback is preserved
  in Streamlit Cloud server logs / a diagnostics expander.
- Three UI integration/regression tests. They exercise the real optimizer result,
  both exact-format HTML tables, the plots, the Excel export, a real `date`
  instance, and injected TypeErrors. The tested Python package passes **27 tests**.

**Verification limits:** The build was tested in Python with a minimal Streamlit
UI shim because the Streamlit library is not installed in this build
container. The original Cloud exception's final line and the user's raw input
files were not provided; live Cloud reproduction is not yet confirmed. If the
new diagnostics reports an error on your data, copy the new *exact exception
message* and traceback from the expanded diagnostics for a targeted fix.
