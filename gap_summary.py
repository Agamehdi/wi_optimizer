"""Screenshot-style Rumaila Capacity Gap summary, original vs actual optimizer allocation.

Input: the canonical results of comparison_views.prepare_comparisons().
All source rates are bbl/day; presentation tables use mbd (thousand bbl/day).
This module never modifies or re-runs the optimization solution.
"""
from __future__ import annotations

import html
import numpy as np
import pandas as pd


RATE_FIELDS = [
    'RMT_TARGET', 'RMT_OE', 'WELL_CAPACITY', 'AGREED_TARGET',
    'Maximum Manifold Capacity', 'Manifold Injection Rate',
    'Maximum CPS Capacity', 'CPS Injection Rate', 'IIC', 'INJ_RATE @ OE',
    'Well Constraint to Selected Target',
    'MF Capacity Shortfall to Agreed Target',
    'MF Capacity Shortfall to RMT+OE Target',
    'CPS Capacity Shortfall to Agreed Target',
    'CPS Capacity Shortfall to RMT+OE Target',
    'Delivery Gap to Selected Target', 'MF Ullage', 'CPS Ullage',
]
FACILITY_FIELDS = {
    'Maximum CPS Capacity', 'CPS Injection Rate', 'IIC', 'INJ_RATE @ OE',
    'CPS Capacity Shortfall to Agreed Target',
    'CPS Capacity Shortfall to RMT+OE Target', 'CPS Ullage',
}
PHYSICAL_GAP_FIELDS = [
    'Well Constraint to Selected Target',
    'MF Capacity Shortfall to Agreed Target',
    'MF Capacity Shortfall to RMT+OE Target',
    'CPS Capacity Shortfall to Agreed Target',
    'CPS Capacity Shortfall to RMT+OE Target',
]
ALL_FIELDS = ['CPS', 'Manifold', *RATE_FIELDS[:10], 'High Gap', *RATE_FIELDS[10:]]


def build_gap_summary(comparisons: dict, target: str = 'RMT+OE Target',
                      high_threshold_mbd: float = 20.0) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return (Original table, Optimized table, per-manifold change) in mbd.

    Signed well gap matches legacy screenshot: min(well capacity-target, 0).
    Demand/structural gap may change at a manifold only for an approved reroute;
    CPS physical shortfall cannot change due to allocation alone.
    """
    if target not in ('RMT Target', 'RMT+OE Target', 'Agreed Target'):
        raise ValueError('Unsupported selected gap basis')
    mf = comparisons['capacity_manifold'].copy()
    fac = comparisons['capacity_cps'].copy()
    if mf.columns.duplicated().any() or fac.columns.duplicated().any():
        raise ValueError('Duplicate column names in capacity comparison')
    if mf.empty or fac.empty:
        raise ValueError('No capacity data to summarize')
    if mf[['CPS', 'Manifold']].duplicated().any() or fac['CPS'].duplicated().any():
        raise ValueError('Non-unique physical manifold/CPS keys')
    indexed_fac = fac.set_index('CPS')
    tables = {}
    for label, suffix in [('Original', 'Before'), ('Optimized', 'After')]:
        t = pd.DataFrame(index=mf.index)
        t['CPS'] = mf['CPS'].astype(str)
        t['Manifold'] = mf['Manifold'].astype(str)
        t['RMT_TARGET'] = mf[f'RMT Target {suffix}']
        t['RMT_OE'] = mf[f'RMT+OE Target {suffix}']
        t['WELL_CAPACITY'] = mf[f'BMT Well Capacity {suffix}']
        t['AGREED_TARGET'] = mf[f'Agreed Target {suffix}']
        t['Maximum Manifold Capacity'] = mf['MF Capacity']
        t['Manifold Injection Rate'] = mf[f'Injection {suffix}']
        t['Maximum CPS Capacity'] = t['CPS'].map(indexed_fac['CPS Capacity'])
        t['CPS Injection Rate'] = t['CPS'].map(indexed_fac[f'Injection {suffix}'])
        # IIC and INJ_RATE @ OE from screenshot are used here as the *selected*
        # original / optimized scenario's allocated injection. Not a separate
        # post-optimization RMT+OE scenario prediction.
        t['IIC'] = t['CPS Injection Rate']
        t['INJ_RATE @ OE'] = t['CPS Injection Rate']
        t['Well Constraint to Selected Target'] = -mf[f'Well Physical Shortfall {suffix}']
        t['MF Capacity Shortfall to Agreed Target'] = (t['AGREED_TARGET'] - t['Maximum Manifold Capacity']).clip(lower=0)
        t['MF Capacity Shortfall to RMT+OE Target'] = (t['RMT_OE'] - t['Maximum Manifold Capacity']).clip(lower=0)
        agreed_fac = indexed_fac[f'Agreed Target {suffix}']
        oe_fac = indexed_fac[f'RMT+OE Target {suffix}']
        facility_capacity = indexed_fac['CPS Capacity']
        t['CPS Capacity Shortfall to Agreed Target'] = t['CPS'].map((agreed_fac - facility_capacity).clip(lower=0))
        t['CPS Capacity Shortfall to RMT+OE Target'] = t['CPS'].map((oe_fac - facility_capacity).clip(lower=0))
        t['Delivery Gap to Selected Target'] = mf[f'Delivery Gap {suffix}']
        t['MF Ullage'] = mf[f'MF Ullage {suffix}']
        t['CPS Ullage'] = t['CPS'].map(indexed_fac[f'CPS Ullage {suffix}'])
        if 'Missing BMT wells' in mf:
            t['Missing BMT wells'] = pd.to_numeric(mf['Missing BMT wells'], errors='coerce').fillna(0).astype(int)
            # Neither original nor optimized well-based gap can certify an
            # unassessed BMT capacity. Keep uncertainty visible as N/A.
            t.loc[t['Missing BMT wells'].gt(0), 'Well Constraint to Selected Target'] = np.nan
        else:
            t['Missing BMT wells'] = 0
        # Convert bbl/day to mbd once, at the boundary of display/export.
        for c in RATE_FIELDS:
            t[c] = pd.to_numeric(t[c], errors='coerce') / 1000.0
        severity = pd.concat([
            (-t['Well Constraint to Selected Target']).clip(lower=0).rename('well'),
            *[t[c].clip(lower=0).rename(c) for c in PHYSICAL_GAP_FIELDS[1:]],
        ], axis=1).max(axis=1)
        t['High Gap'] = np.where(severity.ge(float(high_threshold_mbd)), 'HIGH', '')
        t = t.loc[:, ['CPS', 'Manifold', *RATE_FIELDS[:10], 'High Gap', *RATE_FIELDS[10:], 'Missing BMT wells']]
        # Aggregate facility-only fields only once per CPS, not per manifold.
        total = {'CPS': 'TOTAL', 'Manifold': ''}
        t['CPS'] = pd.Categorical(t['CPS'], categories=sorted(t['CPS'].unique()), ordered=True)
        t = t.sort_values(['CPS', 'Manifold']).reset_index(drop=True)
        t['CPS'] = t['CPS'].astype(str)
        for c in RATE_FIELDS:
            vals = t.drop_duplicates('CPS')[c] if c in FACILITY_FIELDS else t[c]
            total[c] = float(vals.sum()) if vals.notna().all() else float('nan')
        total['High Gap'] = ''
        total['Missing BMT wells'] = int(t['Missing BMT wells'].sum())
        t = pd.concat([t, pd.DataFrame([total])], ignore_index=True)
        tables[label] = t

    b = tables['Original'].query("CPS != 'TOTAL'").set_index(['CPS', 'Manifold'])
    a = tables['Optimized'].query("CPS != 'TOTAL'").set_index(['CPS', 'Manifold'])
    before = b.reindex(b.index.union(a.index)).fillna({'Manifold Injection Rate': 0})
    after = a.reindex(before.index).fillna({'Manifold Injection Rate': 0})
    diff = pd.DataFrame(index=before.index)
    diff['Original Injection (mbd)'] = before['Manifold Injection Rate']
    diff['Optimized Injection (mbd)'] = after['Manifold Injection Rate']
    diff['Injection Δ (mbd)'] = diff['Optimized Injection (mbd)'] - diff['Original Injection (mbd)']
    diff['Original Delivery Gap (mbd)'] = before['Delivery Gap to Selected Target']
    diff['Optimized Delivery Gap (mbd)'] = after['Delivery Gap to Selected Target']
    diff['Delivery Gap Reduction (mbd)'] = diff['Original Delivery Gap (mbd)'] - diff['Optimized Delivery Gap (mbd)']
    diff['Original MF Ullage (mbd)'] = before['MF Ullage']
    diff['Optimized MF Ullage (mbd)'] = after['MF Ullage']
    diff['MF Ullage Δ (mbd)'] = diff['Optimized MF Ullage (mbd)'] - diff['Original MF Ullage (mbd)']
    diff['MF Physical Shortfall Δ (mbd)'] = (after['MF Capacity Shortfall to RMT+OE Target']
                                             - before['MF Capacity Shortfall to RMT+OE Target'])
    diff = diff.reset_index().sort_values(['CPS', 'Manifold']).reset_index(drop=True)
    return tables['Original'], tables['Optimized'], diff


# Exact column structure and visual styling of the legacy ROO Gap Analysis
# screen: no extra Decision Impact columns inside the 18-column screenshot.
# The separate Changes by Manifold tab remains unchanged.
SCREENSHOT_COLUMNS = [
    'CPS', 'Manifold', 'RMT_TARGET', 'RMT_OE', 'WELL_CAPACITY',
    'AGREED_TARGET', 'Maximum Manifold Capacity', 'Manifold Injection Rate',
    'Maximum CPS Capacity', 'CPS Injection Rate', 'IIC', 'INJ_RATE @ OE',
    'High Gap', 'Well Constraint to Selected Target',
    'MF Capacity Shortfall to Agreed Target',
    'MF Capacity Shortfall to RMT+OE Target',
    'CPS Capacity Shortfall to Agreed Target',
    'CPS Capacity Shortfall to RMT+OE Target',
]
SCREENSHOT_FACILITY_COLUMNS = FACILITY_FIELDS.intersection(SCREENSHOT_COLUMNS)

_SCREENSHOT_GAP_CSS = r'''<style>
/* Scoped to the reference table, and compatible with Streamlit dark mode. */
.roo-gap-scroll{width:100%;max-width:100%;overflow-x:auto;overflow-y:visible;
 padding:0 0 8px;margin:5px 0 18px;background:transparent}
.roo-gap-reference{min-width:1710px;width:100%;border-collapse:collapse;border-spacing:0;
 font-family:Arial,Helvetica,sans-serif;font-size:10.5px;color:#0b1017;background:#fff;
 table-layout:fixed;line-height:1.12}
.roo-gap-reference th,.roo-gap-reference td{border:1px solid #929292;
 padding:4px 5px;box-sizing:border-box;vertical-align:middle}
.roo-gap-reference thead th{background:#fff;color:#0a111a;font-weight:700;
 text-align:center;white-space:normal;word-break:normal}
.roo-gap-reference thead tr.top-groups th{height:32px}
.roo-gap-reference thead tr.column-headings th{height:34px;line-height:1.10}
.roo-gap-reference td{height:22px;text-align:right;white-space:nowrap;
 font-variant-numeric:tabular-nums}
.roo-gap-reference td.fac{text-align:center;font-weight:500;background:#fff}
.roo-gap-reference td.manifold{text-align:center;font-weight:500}
.roo-gap-reference td.flag{text-align:center;font-weight:700}
.roo-gap-reference td.gap-zero{background:#e0f1d5}
.roo-gap-reference td.gap-low{background:#e0f1d5}
.roo-gap-reference td.gap-med{background:#ffcb3f}
.roo-gap-reference td.gap-high{background:#db671e;color:#101010}
.roo-gap-reference td.gap-neg{background:#df6721;color:#101010}
.roo-gap-reference tbody tr.total-row td{background:#fff;font-weight:700;
 border-top:2px solid #6c6c6c;height:25px}
.roo-gap-reference tbody tr.total-row td:first-child{text-align:left}
.roo-gap-heading{font:700 17px Arial,Helvetica,sans-serif;color:inherit;margin:4px 0 6px}
.roo-gap-reference .long-cap{line-height:1.07}
@media(max-width:800px){.roo-gap-heading{font-size:14px}}
</style>'''


def gap_table_html(frame: pd.DataFrame, scenario: str, target: str,
                   analysis_date=None) -> str:
    '''Render exactly the same 18 column layout as the original Gap Analysis screenshot.

    The underlying frame comes from the unchanged optimizer comparison engine.
    The screen merges group cells vertically, keeps the bold TOTAL row,
    and does not add any unsupported physical capacity improvements.
    '''
    missing = [c for c in SCREENSHOT_COLUMNS if c not in frame.columns]
    if missing:
        raise ValueError(f'Columns missing for legacy gap view: {missing}')
    groups = [('', 2), ('RESERVOIR', 2), ('WELL', 2), ('MANIFOLD', 2),
              ('CPS', 2), ('FINAL INJ', 1), ('INJ_RATE @ OE', 1),
              ('Capacity Gap_TABLE', 6)]
    headers = [
        'CPS', 'Manifold', 'RMT_TARGET', 'RMT_OE', 'WELL_CAPACITY',
        'AGREED_TARGET', 'Maximum Manifold Capacity', 'Manifold Injection Rate',
        'Maximum CPS Capacity', 'CPS Injection Rate', 'IIC', '', 'High Gap',
        # The selected target is shown in the column to avoid claiming the
        # signed well gap is against RMT when RMT+OE is selected.
        f'Well Constraint to {target}',
        'MF Capacity Shortfall to Agreed Target',
        'MF Capacity Shortfall to RMT+OE Target',
        'CPS Capacity Shortfall to Agreed Target',
        'CPS Capacity Shortfall to RMT+OE Target',
    ]
    assert len(SCREENSHOT_COLUMNS) == len(headers) == sum(span for _,span in groups) == 18
    date_text = '' if analysis_date is None else f' {analysis_date}'
    title = f'Gap analysis{date_text} — {scenario.title()} — mbd'
    # Widths sum to approximately the reference screenshot.  The last 6
    # columns preserve the widest gap headers for exact screenshot alignment.
    col_widths = [42, 58, 64, 64, 94, 96, 104, 105, 95, 95, 52, 78,
                  49, 119, 145, 145, 154, 170]
    out = [_SCREENSHOT_GAP_CSS, '<div class="roo-gap-heading">',
           html.escape(title), '</div>', '<div class="roo-gap-scroll">',
           '<table class="roo-gap-reference"><colgroup>']
    out.extend(f'<col style="width:{size}px">' for size in col_widths)
    out.extend(['</colgroup><thead><tr class="top-groups">'])
    out.extend(f'<th colspan="{span}">{html.escape(group)}</th>' for group,span in groups)
    out.append('</tr><tr class="column-headings">')
    out.extend(f'<th>{html.escape(h)}</th>' for h in headers)
    out.append('</tr></thead><tbody>')
    data=frame.loc[frame['CPS'].astype(str).ne('TOTAL')]
    counts=data.groupby('CPS',sort=False,observed=True).size().to_dict()
    previous=None
    for _, r in data.iterrows():
        fac=str(r['CPS'])
        first=fac!=previous
        previous=fac
        out.append('<tr>')
        for field in SCREENSHOT_COLUMNS:
            if field=='CPS' or field in SCREENSHOT_FACILITY_COLUMNS:
                if not first:
                    continue
                span=f' rowspan="{counts[fac]}"'
            else:
                span=''
            val=r[field]
            cell_class=''
            if field=='CPS':
                content=fac
                cell_class='fac'
            elif field=='Manifold':
                value=str(val)
                content=value[len(fac)+1:] if value.startswith(fac+'_') else value
                cell_class='manifold'
            elif field=='High Gap':
                content='' if pd.isna(val) else str(val)
                cell_class='flag'
            elif pd.isna(val):
                content='n.a.'
            else:
                value=float(val)
                content=f'{value:,.1f}'
                if field in PHYSICAL_GAP_FIELDS:
                    # Screenshot palette: green = no gap, yellow = moderate,
                    # orange = severe. Signed well constraints are negative.
                    shortfall = (-value if field=='Well Constraint to Selected Target' else value)
                    if shortfall <= 0.001:
                        cell_class='gap-zero'
                    elif shortfall >= 20:
                        cell_class='gap-high'
                    elif shortfall >= 5:
                        cell_class='gap-med'
                    else:
                        cell_class='gap-low'
            out.append(f'<td class="{cell_class}"{span}>{html.escape(content)}</td>')
        out.append('</tr>')
    total=frame.loc[frame['CPS'].astype(str).eq('TOTAL')]
    if not total.empty:
        row=total.iloc[0]
        out.append('<tr class="total-row">')
        for field in SCREENSHOT_COLUMNS:
            value=row[field]
            if field in RATE_FIELDS:
                content='n.a.' if pd.isna(value) else f'{float(value):,.1f}'
            elif field=='High Gap':
                content=''
            else:
                content='' if pd.isna(value) else str(value)
            out.append(f'<td>{html.escape(content)}</td>')
        out.append('</tr>')
    out.append('</tbody></table></div>')
    return ''.join(out)
