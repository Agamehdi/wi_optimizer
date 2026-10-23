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


# Screenshot-inspired layout: grouped headers, CPS-only values merged once,
# total row with CPS measures de-duplicated. Horizontal scrolling on desktop.
_SUMMARY_CSS = '''<style>
.wi-gap-wrap{width:100%;overflow-x:auto;margin:8px 0 23px;padding:0 0 6px}
.wi-gap-title{font:700 17px Arial,sans-serif;color:#173553;margin:8px 0 10px}
.wi-gap-table{border-collapse:separate;border-spacing:0;font:12px Arial,sans-serif;
 min-width:2200px;background:#fff;color:#24384b;table-layout:auto;width:100%}
.wi-gap-table th,.wi-gap-table td{border-right:1px solid #d4dde6;border-bottom:1px solid #d4dde6;
 padding:8px 10px;text-align:right;white-space:nowrap;font-variant-numeric:tabular-nums}
.wi-gap-table th:first-child,.wi-gap-table td:first-child{border-left:1px solid #d4dde6}
.wi-gap-table thead tr:first-child th{background:#173553;color:#fff;border-color:#49627c;
 font-weight:700;text-align:center;padding:10px 7px}
.wi-gap-table thead tr:nth-child(2) th{background:#e7eef5;text-align:center;
 font-weight:700;color:#183554;white-space:normal;min-width:96px;line-height:1.3}
.wi-gap-table thead tr:nth-child(2) th:nth-last-child(-n+3){background:#d7e9e3}
.wi-gap-table td.fac{font-weight:700;background:#f0f5f9;text-align:center;vertical-align:middle}
.wi-gap-table td.manifold{text-align:left;font-weight:600}
.wi-gap-table tbody tr.fac-start>td{border-top:2px solid #a6b9cb}
.wi-gap-table td.high{font-weight:700;color:#b45309;background:#fff0d5;text-align:center}
.wi-gap-table td.lowgap{background:#e1f3e7}
.wi-gap-table td.mediumgap{background:#ffebac}
.wi-gap-table td.criticalgap{background:#f7b181;color:#78300f}
.wi-gap-table td.negative{color:#b91c1c;font-weight:700;background:#fff0f0}
.wi-gap-table .total td{border-top:3px double #173553;background:#dce8f3;font-weight:700}
.wi-gap-table tbody tr:hover td:not(.fac){filter:brightness(.965)}
</style>'''


def gap_table_html(frame: pd.DataFrame, scenario: str, target: str) -> str:
    """Self-contained, escaped HTML preserving the attached grouped-header table."""
    cols = ALL_FIELDS
    # 2 id columns + 2 reservoir + 2 well + 2 manifold + 2 cps + 1 final
    # + 1 rate @ OE + 6 structural gap fields + 3 decision fields.
    groups = [('', 2), ('RESERVOIR', 2), ('WELL', 2), ('MANIFOLD', 2),
              ('CPS', 2), ('FINAL INJ', 1), ('INJ RATE', 1),
              ('Capacity Gap Table', 6), ('Decision Impact', 3)]
    groups_html = ''.join(f'<th colspan="{span}">{html.escape(group)}</th>' for group,span in groups)
    headers = [
        'CPS','Manifold','RMT_TARGET','RMT_OE','WELL_CAPACITY','AGREED_TARGET',
        'Maximum Manifold Capacity','Manifold Injection Rate','Maximum CPS Capacity',
        'CPS Injection Rate','IIC','INJ_RATE @ OE','High Gap',
        f'Well Constraint to {target}', 'MF Capacity Shortfall to Agreed Target',
        'MF Capacity Shortfall to RMT+OE Target',
        'CPS Capacity Shortfall to Agreed Target',
        'CPS Capacity Shortfall to RMT+OE Target',
        f'MF Delivery Gap to {target}', 'MF Ullage', 'CPS Ullage',
    ]
    assert len(cols) == len(headers) == sum(span for _,span in groups)
    output=[_SUMMARY_CSS, '<div class="wi-gap-wrap">',
            f'<div class="wi-gap-title">{html.escape(scenario)} — Gap Analysis Summary (mbd)</div>',
            '<table class="wi-gap-table"><thead><tr>',groups_html,'</tr><tr>']
    output.extend(f'<th>{html.escape(h)}</th>' for h in headers)
    output.extend(['</tr></thead><tbody>'])
    data = frame[frame['CPS'].ne('TOTAL')]
    cps_only = FACILITY_FIELDS
    counts = data.groupby('CPS', sort=False).size().to_dict()
    last_fac = None
    for _, r in data.iterrows():
        first = r['CPS'] != last_fac
        fac = str(r['CPS']); last_fac = fac
        output.append(f'<tr class="{ "fac-start" if first else ""}">')
        for field in cols:
            if field == 'CPS' or field in cps_only:
                if not first:
                    continue
                span=f' rowspan="{counts[fac]}"'
            else: span=''
            value=r[field]
            if field == 'Manifold':
                txt = str(value)
                shown = txt[len(fac)+1:] if txt.startswith(fac+'_') else txt
                klass='manifold'
            elif field == 'CPS':
                shown = fac;klass='fac'
            elif field == 'High Gap':
                shown = str(value);klass='high' if shown=='HIGH' else ''
            elif pd.isna(value):
                shown = 'n.a.';klass=''
            else:
                val=float(value)
                shown=f'{val:,.1f}'
                if field in PHYSICAL_GAP_FIELDS:
                    sev=-val if field.startswith('Well Constraint') else val
                    klass='criticalgap' if sev>20 else 'mediumgap' if sev>5 else 'lowgap'
                elif field in ('MF Ullage','CPS Ullage') and val < -0.0001:
                    klass='negative'
                else: klass=''
            output.append(f'<td class="{klass}"{span}>{html.escape(shown)}</td>')
        output.append('</tr>')
    total = frame[frame.CPS.eq('TOTAL')]
    if len(total):
        r=total.iloc[0]
        output.append('<tr class="total">')
        for field in cols:
            val=r[field]
            txt=('n.a.' if pd.isna(val) else f'{float(val):,.1f}') if field in RATE_FIELDS else str(val)
            output.append(f'<td>{html.escape(txt)}</td>')
        output.append('</tr>')
    output.append('</tbody></table></div>')
    return ''.join(output)
