"""Run: streamlit run WI_System_Gap_Analysis.py
All rates are bbl/day internally and mbd in the report. No external services.
"""
from __future__ import annotations
import io
import re
import html
from datetime import timedelta
import numpy as np
import pandas as pd

RATE_COLUMNS = {'rmt': 'RMT Rate [bbl/day]', 'oe': 'RMT Rate+OE [bbl/day]',
                'capacity': 'BMT Constrained Rate [bbl/day]', 'agreed': 'Agreed Target [bbl/day]'}
METRICS = ['RMT_OE', 'WELL_CAPACITY', 'AGREED_TARGET', 'Maximum Manifold Capacity',
           'Manifold Injection Rate', 'Maximum CPS Capacity', 'CPS Injection Rate',
           'IIC', 'INJ_RATE @ OE', 'Well Constraint to Selected RMT Target',
           'MF Capacity Shortfall to Agreed Target', 'MF Capacity Shortfall to RMT+OE Target',
           'CPS Capacity Shortfall to Agreed Target', 'CPS Capacity Shortfall to RMT+OE Target']
GAPS = METRICS[9:]
QUALITY = ['Missing BMT wells', 'Unassessed RMT Target', 'Unassessed RMT+OE', 'Unassessed Agreed Target']
# Facility-level fields below are repeated across manifold rows for Excel/filtering.
# Never sum them across manifolds; count each CPS/PWRI facility exactly once.
ULLAGE = ['Injection Potential (Well+MF+CPS)', 'Unutilizable CPS Ullage',
          'Target/Operational Ullage', 'Efficiency Ullage', 'Total CPS Ullage']
AUX_FIELDS = ['RMT_TARGET', *ULLAGE]
FORECAST_FIELDS = [*METRICS, *AUX_FIELDS]


def read_csv(source):
    if hasattr(source, 'getvalue'):
        source = io.BytesIO(source.getvalue())
    return pd.read_csv(source, sep=None, engine='python', encoding='utf-8-sig')

def parse_dates(series, dayfirst=False):
    # ISO facility dates always use year-month-day; well export defaults to US dates.
    def one(value):
        if pd.isna(value):
            return pd.NaT
        text = str(value).strip()
        return pd.to_datetime(text, errors='coerce', dayfirst=False if re.match(r'^\d{4}-', text) else dayfirst)
    return series.map(one).dt.normalize()

def normal_cps(value):
    text = re.sub(r'[^A-Z0-9]', '', str(value).upper())
    match = re.search(r'CPS0*(\d+)', text)
    if match:
        return 'CPS' + str(int(match.group(1)))
    return text if text.startswith('PWRI') else 'UNASSIGNED'

def normal_facility(value):
    text = re.sub(r'[^A-Z0-9]', '', str(value).upper())
    if text.startswith('PWRI'):
        match = re.match(r'(PWRI\d+)', text)
        return match.group(1) if match else text
    return normal_cps(value)

def normal_manifold(value):
    text = re.sub(r'[^A-Z0-9]', '', str(value).upper())
    if text in ('', 'NAN', 'NONE'):
        return ''
    cps = normal_cps(value)
    match = re.search(r'(FM|SM|PM)0*(\d*)', text)
    if not match:
        raise ValueError(f'Unrecognized manifold: {value}')
    kind, number = match.groups()
    suffix = 'SM' if kind in ('SM', 'PM') else 'FM' + number.zfill(2)
    return (cps + '_' if cps != 'UNASSIGNED' else '') + suffix

def prepare_wells(raw, dayfirst=False, route='PWRI priority'):
    d = raw.copy()
    d.columns = d.columns.str.strip()
    required = ['Well', 'CPS', 'Record Date', 'SW Manifold', *RATE_COLUMNS.values()]
    absent = [c for c in required if c not in d]
    if absent:
        raise ValueError('Well CSV missing columns: ' + ', '.join(absent))
    if d['Well'].isna().any():
        raise ValueError('Well CSV contains missing well IDs.')
    d['Well'] = d['Well'].astype(str).str.strip()
    d['date'] = parse_dates(d['Record Date'], dayfirst)
    if d['date'].isna().any():
        raise ValueError('Invalid Record Date values. Check date format selection.')
    for short, column in RATE_COLUMNS.items():
        d[short] = pd.to_numeric(d[column].astype(str).str.replace(',', '', regex=False), errors='coerce')
        if (d[short].dropna() < 0).any():
            raise ValueError(f'Negative rates found in {column}.')
    if d[['rmt', 'oe', 'agreed']].isna().any().any():
        raise ValueError('Missing or invalid RMT, RMT+OE or Agreed Target values.')
    d['facility'] = d['CPS'].map(normal_cps)
    d['manifold'] = d['SW Manifold'].fillna('').map(normal_manifold)
    bare_sw = d['manifold'].ne('') & d['SW Manifold'].fillna('').map(normal_cps).eq('UNASSIGNED')
    d.loc[bare_sw, 'manifold'] = [normal_manifold(fac + '_' + str(m))
        for fac,m in zip(d.loc[bare_sw,'facility'], d.loc[bare_sw,'SW Manifold'])]
    d.loc[d['manifold'].eq(''), 'manifold'] = 'UNASSIGNED'
    if route == 'PWRI priority':
        for i, row in d.iterrows():
            value = row.get('PWRI Manifold', '')
            if pd.notna(value) and str(value).strip():
                cps = normal_cps(value)
                explicit = re.search(r'PWRI\s*[-_]?\s*(\d+)', str(value), flags=re.I)
                if explicit:
                    prefix = 'PWRI' + explicit.group(1)
                elif cps in ('CPS1', 'CPS4', 'CPS5'):
                    prefix = 'PWRI1'
                elif cps == 'CPS7':
                    prefix = 'PWRI2'
                else:
                    raise ValueError(f'PWRI routing is not defined for {value}. Use an explicit PWRI1/PWRI2 prefix.')
                d.at[i, 'facility'] = prefix
                d.at[i, 'manifold'] = prefix + '_' + normal_manifold(value)
    efficiency = pd.to_numeric(d.get('Well Efficiency', pd.Series(100., index=d.index)), errors='coerce') / 100
    if efficiency.isna().any() or ((efficiency < 0) | (efficiency > 1)).any():
        raise ValueError('Well Efficiency must be a percentage from 0 to 100.')
    d['eff'] = efficiency
    duplicate = d.duplicated(['Well', 'date'], keep=False)
    if duplicate.any():
        columns = ['Well', 'date', 'facility', 'manifold', 'rmt', 'oe', 'capacity', 'agreed', 'eff']
        d = d.drop_duplicates(columns)
        if d.duplicated(['Well', 'date']).any():
            raise ValueError('Conflicting records exist for the same well and date; resolve them first.')
    return d.sort_values(['date', 'Well'])

def prepare_facilities(raw, units='mbd', dayfirst=False):
    d = raw.copy()
    d.columns = d.columns.str.strip()
    if {'FAC_NAME', 'Manifold', 'DATE', 'CAP'}.issubset(d.columns):
        records = []
        for _, r in d.iterrows():
            if not re.search(r'CPS|PWRI', str(r['FAC_NAME']), flags=re.I):
                # Field/QA summary rows are not individual routing constraints.
                continue
            fac = normal_facility(r['FAC_NAME'])
            m = str(r['Manifold']).strip() if pd.notna(r['Manifold']) else ''
            fac_text = re.sub(r'[^A-Z0-9]', '', str(r['FAC_NAME']).upper())
            is_cps = bool(re.fullmatch(r'CPS\d+', fac_text))
            is_pwri = bool(re.fullmatch(r'PWRI\d+', fac_text))
            if is_cps or is_pwri:
                records.append([fac, '', r['DATE'], r['CAP']])
            else:
                manifold_source = m or r['FAC_NAME']
                key = normal_manifold(manifold_source)
                # Accept bare FM05/SM labels in FAC_NAME long-form inputs, but
                # never silently misassign an explicitly different CPS.
                if fac.startswith('CPS'):
                    key_cps = normal_cps(manifold_source)
                    if key_cps == 'UNASSIGNED':
                        key = normal_manifold(fac + '_' + manifold_source)
                    elif key_cps != fac:
                        raise ValueError(f'Facility/Manifold CPS mismatch: {r["FAC_NAME"]} / {manifold_source}')
                if fac.startswith('PWRI'):
                    key = fac + '_' + key
                records.append([fac, key, r['DATE'], r['CAP']])
        f = pd.DataFrame(records, columns=['facility', 'manifold', 'date', 'cap'])
    elif {'CPS', 'Manifold', 'DATE', 'Manifold Capacity', 'CPS Capacity'}.issubset(d.columns):
        records = []
        for _, r in d.iterrows():
            fac = normal_facility(r['CPS'])
            if pd.notna(r['CPS Capacity']):
                records.append([fac, '', r['DATE'], r['CPS Capacity']])
            if pd.notna(r['Manifold Capacity']):
                records.append([fac, normal_manifold(str(r['CPS']) + '_' + str(r['Manifold'])), r['DATE'], r['Manifold Capacity']])
        f = pd.DataFrame(records, columns=['facility', 'manifold', 'date', 'cap'])
    else:
        raise ValueError('facility.csv needs FAC_NAME, Manifold, DATE, CAP (blank Manifold for CPS totals).')
    f['date'] = parse_dates(f['date'], dayfirst)
    f['cap'] = pd.to_numeric(f['cap'].astype(str).str.replace(',', '', regex=False), errors='coerce')
    if f['date'].isna().any() or f['cap'].isna().any() or (f['cap'] < 0).any():
        raise ValueError('Facility dates/capacities are invalid or missing. Zero is a valid shutdown capacity.')
    if units == 'mbd':
        f['cap'] *= 1000
    if f.empty:
        raise ValueError('No CPS or PWRI constraints found in facility.csv.')
    f = f.drop_duplicates()
    if f.duplicated(['facility', 'manifold', 'date']).any():
        raise ValueError('Conflicting facility capacity rows for the same facility/manifold/date.')
    return f.sort_values('date')

def latest_wells(wells, date, missing='Exclude and flag'):
    date = pd.Timestamp(date)
    d = wells[wells['date'] <= date].sort_values('date').drop_duplicates('Well', keep='last').copy()
    missing_rows = d[d['capacity'].isna()].copy()
    if missing == 'Exclude and flag':
        d = d[d['capacity'].notna()].copy()
    elif missing == 'Treat as zero':
        d['capacity'] = d['capacity'].fillna(0)
    else:
        if not missing_rows.empty:
            raise ValueError(f'{len(missing_rows)} wells have missing BMT constrained capacity.')
    return d, missing_rows

def capacity_map(facilities, date):
    f = facilities[facilities['date'] <= pd.Timestamp(date)].sort_values('date').drop_duplicates(['facility', 'manifold'], keep='last')
    return {(r.facility, r.manifold): float(r.cap) for r in f.itertuples()}

def calculate(wells, facilities, date, missing='Exclude and flag', scenario='Agreed Target',
              efficiency=True, decline=0., decline_start=None, decline_period=365.25,
              baseline=None, gap_basis='RMT+OE'):
    # Forecast freezes the well snapshot at baseline; capacity schedules remain time-dependent.
    d, unknown = latest_wells(wells, baseline if baseline is not None else date, missing)
    snapshot_date = baseline if baseline is not None else date
    all_wells = wells[wells['date'] <= pd.Timestamp(snapshot_date)].sort_values('date').drop_duplicates('Well', keep='last').copy()
    if d.empty:
        raise ValueError('No usable well records on or before the selected date.')
    days = max(0., (pd.Timestamp(date) - pd.Timestamp(decline_start or date)).days)
    factor = (1 - decline / 100.) ** (days / decline_period)
    d['capacity'] *= factor
    caps = capacity_map(facilities, date)
    for fac, manifold in d[['facility', 'manifold']].drop_duplicates().itertuples(index=False, name=None):
        for key in [(fac, manifold), (fac, '')]:
            if key not in caps:
                raise ValueError(f'Missing facility capacity effective on {pd.Timestamp(date).date()}: {key}. Add it to facility.csv.')
    d['Reservoir'] = d.get('Reservoir', pd.Series('Unspecified', index=d.index)).fillna('Unspecified')
    audit = d[['Well', 'Reservoir', 'date', 'facility', 'manifold', 'rmt', 'oe', 'capacity', 'agreed', 'eff']].copy()
    # The chosen well gap is independent of the injection scenario.
    # Both well-gap alternatives remain in the detailed audit (bbl/day).
    audit['rmt_well_gap'] = np.minimum(d['rmt'], d['capacity']) - d['rmt']
    audit['oe_well_gap'] = np.minimum(d['oe'], d['capacity']) - d['oe']
    audit['selected_well_gap'] = audit['rmt_well_gap'] if gap_basis == 'RMT Target' else audit['oe_well_gap']

    # Target-independent physical deliverability: each well's BMT capacity is
    # reduced for shared manifold and facility constraints, but NOT for RMT or
    # Agreed Target. This defines physical/structural stranded CPS ullage.
    manifold_total = d.groupby(['facility', 'manifold'])['capacity'].transform('sum')
    mf_limit = pd.Series([caps[(f, m)] for f, m in zip(d.facility, d.manifold)], index=d.index)
    manifold_factor = np.minimum(1., mf_limit / manifold_total.replace(0, np.nan)).fillna(1.)
    network_mf = d['capacity'] * manifold_factor
    facility_total = network_mf.groupby(d['facility']).transform('sum')
    cps_limit = d['facility'].map(lambda f: caps[(f, '')])
    facility_factor = np.minimum(1., cps_limit / facility_total.replace(0, np.nan)).fillna(1.)
    audit['potential_after_network'] = network_mf * facility_factor

    scenario_results = {}
    for target in ['oe', 'agreed']:
        work = d.copy()
        work['after_well'] = np.minimum(work[target], work['capacity'])
        work['well_gap'] = work['after_well'] - work[target]
        totals = work.groupby(['facility', 'manifold'])['after_well'].transform('sum')
        mf_limits = pd.Series([caps[(f, m)] for f, m in zip(work.facility, work.manifold)], index=work.index)
        mf_scale = np.minimum(1., mf_limits / totals.replace(0, np.nan)).fillna(1.)
        work['after_mf'] = work['after_well'] * mf_scale
        work['mf_gap'] = work['after_mf'] - work['after_well']
        totals = work.groupby('facility')['after_mf'].transform('sum')
        cps_limits = work.facility.map(lambda f: caps[(f, '')])
        cps_scale = np.minimum(1., cps_limits / totals.replace(0, np.nan)).fillna(1.)
        work['after_cps'] = work['after_mf'] * cps_scale
        work['cps_gap'] = work['after_cps'] - work['after_mf']
        work['injection'] = work['after_cps'] * (work['eff'] if efficiency else 1.)
        for column in ['after_well', 'well_gap', 'after_mf', 'mf_gap', 'after_cps', 'cps_gap', 'injection']:
            audit[target + '_' + column] = work[column]
        scenario_results[target] = work
    selected_name = 'agreed' if scenario == 'Agreed Target' else 'oe'
    audit['selected_injection'] = audit[selected_name+'_injection']
    main = scenario_results[selected_name]
    grouped = main.groupby(['facility', 'manifold'])
    rows = []
    for (fac, manifold), group in grouped:
        ix = group.index
        oe = scenario_results['oe'].loc[ix]
        agreed = scenario_results['agreed'].loc[ix]
        fac_main = main[main.facility.eq(fac)]
        fac_oe = scenario_results['oe'][scenario_results['oe'].facility.eq(fac)]
        fac_agreed = scenario_results['agreed'][scenario_results['agreed'].facility.eq(fac)]
        values = [group.oe.sum(), group.capacity.sum(), group.agreed.sum(), caps[(fac, manifold)],
                  group.after_mf.sum(), caps[(fac, '')], fac_main.after_cps.sum(),
                  fac_main.injection.sum(), fac_main.injection.sum(), audit.loc[ix, 'selected_well_gap'].sum(),
                  agreed.mf_gap.sum(), oe.mf_gap.sum(), fac_agreed.cps_gap.sum(), fac_oe.cps_gap.sum()]
        row = {'CPS': fac, 'Manifold': manifold, 'Wells': len(group)}
        row.update(dict(zip(METRICS, np.array(values) / 1000.)))
        rows.append(row)
    table = pd.DataFrame(rows).sort_values(['CPS', 'Manifold']).reset_index(drop=True)
    # Demand always covers the full dated snapshot, independently of BMT availability.
    demand = all_wells.groupby(['facility', 'manifold']).agg(
        RMT_OE=('oe', 'sum'), RMT_TARGET=('rmt', 'sum'), AGREED_TARGET=('agreed', 'sum'), Wells=('Well', 'size')).reset_index()
    demand = demand.rename(columns={'facility': 'CPS', 'manifold': 'Manifold'})
    demand[['RMT_OE', 'RMT_TARGET', 'AGREED_TARGET']] /= 1000.
    table = demand.merge(table.drop(columns=['RMT_OE', 'AGREED_TARGET', 'Wells']),
                         on=['CPS', 'Manifold'], how='left')
    unresolved = unknown if missing == 'Exclude and flag' else unknown.iloc[:0]
    quality = unresolved.groupby(['facility', 'manifold']).agg(
        **{'Missing BMT wells': ('Well', 'size'), 'Unassessed RMT Target': ('rmt', 'sum'),
           'Unassessed RMT+OE': ('oe', 'sum'),
           'Unassessed Agreed Target': ('agreed', 'sum')}).reset_index()
    quality = quality.rename(columns={'facility':'CPS', 'manifold':'Manifold'})
    quality[['Unassessed RMT Target', 'Unassessed RMT+OE', 'Unassessed Agreed Target']] /= 1000.
    table = table.merge(quality, on=['CPS', 'Manifold'], how='left')
    table[QUALITY] = table[QUALITY].fillna(0)
    for fac, group in table.groupby('CPS'):
        for column in ['Maximum CPS Capacity', 'CPS Injection Rate', 'IIC', 'INJ_RATE @ OE', *GAPS[-2:]]:
            valid = group[column].dropna()
            if len(valid): table.loc[group.index, column] = valid.iloc[0]
    # Direct facility-versus-demand shortfalls are independent of BMT well capacity.
    for idx, row in table.iterrows():
        table.at[idx, 'Maximum Manifold Capacity'] = caps.get((row.CPS, row.Manifold), np.nan) / 1000.
        table.at[idx, 'Maximum CPS Capacity'] = caps.get((row.CPS, ''), np.nan) / 1000.
    table[GAPS[1]] = (table.AGREED_TARGET - table['Maximum Manifold Capacity']).clip(lower=0)
    table[GAPS[2]] = (table.RMT_OE - table['Maximum Manifold Capacity']).clip(lower=0)
    # A missing BMT assessment prevents the selected gap from being certified,
    # even if its target happens to be zero.
    table.loc[table['Missing BMT wells'] > 0, GAPS[0]] = np.nan
    for fac, group in table.groupby('CPS'):
        limit = caps.get((fac, ''), np.nan) / 1000.
        table.loc[group.index, GAPS[3]] = max(group.AGREED_TARGET.sum() - limit, 0.) if pd.notna(limit) else (0. if group.AGREED_TARGET.sum() == 0 else np.nan)
        table.loc[group.index, GAPS[4]] = max(group.RMT_OE.sum() - limit, 0.) if pd.notna(limit) else (0. if group.RMT_OE.sum() == 0 else np.nan)
    for gap, demand_col in [(GAPS[1], 'AGREED_TARGET'), (GAPS[2], 'RMT_OE')]:
        table.loc[table[demand_col].eq(0), gap] = 0.
    # Resolve gross unused CPS capacity into two distinct sources, plus the
    # optional post-constraint well-efficiency effect. Missing BMT records cause
    # ullage to be n.a., rather than wrongly labelling it unusable.
    selected_name = 'agreed' if scenario == 'Agreed Target' else 'oe'
    for fac, group in table.groupby('CPS'):
        limit = caps.get((fac, ''), np.nan) / 1000.
        all_assessed = not (group['Missing BMT wells'] > 0).any()
        potential = audit.loc[audit.facility.eq(fac), 'potential_after_network'].sum() / 1000.
        pre_eff = audit.loc[audit.facility.eq(fac), selected_name+'_after_cps'].sum() / 1000.
        final = audit.loc[audit.facility.eq(fac), selected_name+'_injection'].sum() / 1000.
        vals = ([potential, max(0., limit-potential), max(0., potential-pre_eff),
                 max(0., pre_eff-final), max(0., limit-final)]
                if all_assessed and np.isfinite(limit) else [np.nan]*len(ULLAGE))
        table.loc[group.index, ULLAGE] = vals

    # CPS values repeat in the flat export for filtering; totals count each facility once.
    facility_cols = ['Maximum CPS Capacity', 'CPS Injection Rate', 'IIC', 'INJ_RATE @ OE', *GAPS[-2:], *ULLAGE]
    total = {'CPS': 'TOTAL', 'Manifold': '', 'Wells': len(all_wells)}
    for col in FORECAST_FIELDS:
        source = table.drop_duplicates('CPS')[col] if col in facility_cols else table[col]
        # Never turn unavailable well/ullage assessments into numeric totals.
        total[col] = float(source.sum()) if source.notna().all() else np.nan
    for col in QUALITY:
        total[col] = table[col].sum()
    if total['Missing BMT wells'] > 0:
        total[GAPS[0]] = np.nan
    for col in GAPS[1:]:
        if table[col].isna().any(): total[col] = np.nan
    audit['Capacity assessment'] = 'Assessed' if missing != 'Treat as zero' else 'Missing BMT explicitly treated as zero where applicable'
    if not unresolved.empty:
        unresolved = unresolved.copy()
        unresolved['Reservoir'] = unresolved.get('Reservoir', pd.Series('Unspecified', index=unresolved.index)).fillna('Unspecified')
        unaudited = unresolved[['Well', 'Reservoir', 'date', 'facility', 'manifold', 'rmt', 'oe', 'capacity', 'agreed', 'eff']].copy()
        unaudited['Capacity assessment'] = 'Unknown BMT — demand included; constraint result unavailable'
        audit = pd.concat([audit, unaudited], ignore_index=True)
    audit['Selected scenario'] = scenario
    audit['Well gap basis'] = gap_basis
    audit['Forecast factor'] = factor
    audit['Analysis date'] = pd.Timestamp(date)
    return table, pd.Series(total), audit, unknown

def forecast(wells, facilities, baseline, start, end, decline, period, **kwargs):
    dates = pd.date_range(start, end, freq='D')
    if len(dates) > 3660:
        raise ValueError('Select a forecast period of 10 years or less.')
    tables, totals, reservoirs, last_audit = [], [], [], None
    for day in dates:
        table, total, last_audit, unknown = calculate(wells, facilities, day, decline=decline,
            decline_start=start, decline_period=period, baseline=baseline, **kwargs)
        table['DATE'] = day
        tables.append(table)
        total['DATE'] = day
        totals.append(total)
        reservoirs.append(reservoir_summary(last_audit))
    daily = pd.concat(tables, ignore_index=True)
    average = daily.groupby(['CPS', 'Manifold'], as_index=False)[['Wells', *FORECAST_FIELDS, *QUALITY]].mean()
    total_frame = pd.DataFrame(totals)
    avg_total = total_frame[['Wells', *FORECAST_FIELDS, *QUALITY]].mean()
    avg_total['CPS'], avg_total['Manifold'] = 'TOTAL', ''
    reservoir_average = pd.concat(reservoirs).groupby('Reservoir', as_index=False)[['RMT_TARGET', 'RMT_OE', 'WELL_CAPACITY', 'AGREED_TARGET', 'INJECTION', 'NETWORK_POTENTIAL', 'WELL_GAP_LOSS']].mean()
    return average, avg_total, daily, total_frame, last_audit, unknown, reservoir_average

def reservoir_summary(audit):
    # Audited well rates in bbl/day; reservoir summary in mbd.
    injection_col = 'selected_injection'
    if injection_col not in audit:
        # Historical audit uses the current injection scenario; this column is
        # set in calculate() to avoid mixing scenarios in a reservoir plot.
        raise ValueError('Reservoir injection scenario is absent from well audit.')
    spec = {'rmt':'RMT_TARGET', 'oe':'RMT_OE', 'capacity':'WELL_CAPACITY',
            'agreed':'AGREED_TARGET', 'selected_injection':'INJECTION',
            'potential_after_network':'NETWORK_POTENTIAL', 'selected_well_gap':'WELL_GAP_LOSS'}
    source = audit.groupby('Reservoir', dropna=False)[list(spec)].sum(min_count=1).div(1000)
    source = source.rename(columns=spec)
    source['WELL_GAP_LOSS'] = -source['WELL_GAP_LOSS']
    # Partial BMT coverage cannot establish physical deliverability or gap.
    invalid = audit.groupby('Reservoir', dropna=False)['capacity'].apply(lambda v: v.isna().any())
    source.loc[invalid[invalid].index, ['WELL_GAP_LOSS', 'NETWORK_POTENTIAL']] = np.nan
    return source.reset_index()

def chart_data(table):
    additive = ['RMT_TARGET', 'RMT_OE', 'WELL_CAPACITY', 'AGREED_TARGET',
                'Manifold Injection Rate', GAPS[0], *GAPS[1:3]]
    cps = table.groupby('CPS', sort=False)[additive].sum(min_count=1).reset_index()
    # CPS/PWRI facility fields occur once per manifold: take once, never sum.
    once = table.groupby('CPS', sort=False)[['Maximum CPS Capacity', 'IIC', *GAPS[-2:], *ULLAGE]].first().reset_index()
    return cps.merge(once, on='CPS')

def chart_layout(title, horizontal=False):
    return {'title':{'text':title, 'font':{'size':18, 'color':'#17365D'}},
            'paper_bgcolor':'white', 'plot_bgcolor':'white', 'font':{'family':'Arial', 'size':12, 'color':'#334155'},
            'height':370, 'margin':{'l':65, 'r':25, 't':65, 'b':75},
            'xaxis':{'title':{'text':'mbd' if horizontal else ''}, 'gridcolor':'#E8EDF3', 'zerolinecolor':'#CBD5E1'},
            'yaxis':{'title':{'text':'' if horizontal else 'mbd'}, 'gridcolor':'#E8EDF3', 'zerolinecolor':'#CBD5E1'},
            'legend':{'orientation':'h', 'y':1.13, 'x':0}, 'hovermode':'closest'}

def waterfall_spec(labels, values, title):
    valid = [(str(label), float(value)) for label, value in zip(labels, values) if pd.notna(value)]
    names, rates = zip(*valid) if valid else ([], [])
    total = sum(rates)
    return {'data':[{'type':'waterfall', 'orientation':'v', 'name':'Contribution',
        'x':['Start', *names, 'Total'], 'y':[0., *rates, 0.],
        'measure':['absolute', *['relative']*len(rates), 'total'],
        'text':['0', *[f'{v:,.1f}' for v in rates], f'{total:,.1f}'], 'textposition':'outside',
        'connector':{'line':{'color':'#B9C5D3', 'width':1}},
        'increasing':{'marker':{'color':'#65DBA0'}}, 'decreasing':{'marker':{'color':'#F97565'}},
        'totals':{'marker':{'color':'#65C9CD'}},
        'hovertemplate':'%{x}<br>%{text} mbd<extra></extra>'}], 'layout':chart_layout(title)}

def comparison_spec(data, category, capacity, injection, title):
    colors = ['#2463EB', '#59CB9A', '#F4B83C', '#8593A8', '#64C7CD']
    columns = ['RMT_OE', 'WELL_CAPACITY', 'AGREED_TARGET', capacity, injection]
    labels = ['RMT+OE demand', 'Well capacity', 'Agreed target', 'Facility capacity', 'Assessed injection']
    traces = []
    for col, label, color in zip(columns, labels, colors):
        traces.append({'type':'bar', 'name':label, 'x':data[category].astype(str).tolist(),
                       'y':[None if pd.isna(v) else float(v) for v in data[col]], 'marker':{'color':color},
                       'hovertemplate':'%{x}<br>'+label+': %{y:,.1f} mbd<extra></extra>'})
    layout = chart_layout(title)
    layout.update(barmode='group', height=450)
    return {'data':traces, 'layout':layout}

def shortfall_spec(data, category, agreed_col, oe_col, title):
    traces = []
    for col, label, color in [(agreed_col, 'To Agreed Target', '#F5B841'), (oe_col, 'To RMT+OE', '#F27565')]:
        traces.append({'type':'bar', 'orientation':'h', 'name':label, 'y':data[category].astype(str).tolist(),
            'x':[None if pd.isna(v) else float(v) for v in data[col]], 'marker':{'color':color},
            'hovertemplate':'%{y}<br>'+label+': %{x:,.1f} mbd<extra></extra>'})
    layout = chart_layout(title, True)
    layout.update(barmode='group', height=max(360, 40*len(data)+130))
    return {'data':traces, 'layout':layout}

def ullage_spec(data):
    colors = ['#E87C55', '#F5BC49', '#86B5E0']
    labels = ['Unutilizable (well/manifold)', 'Target / operational headroom', 'Efficiency effect']
    cols = ULLAGE[1:4]
    traces = []
    for col, label, color in zip(cols, labels, colors):
        traces.append({'type':'bar', 'name':label, 'x':data.CPS.astype(str).tolist(),
            'y':[None if pd.isna(x) else float(x) for x in data[col]],
            'marker':{'color':color},
            'hovertemplate':'%{x}<br>'+label+': %{y:,.1f} mbd<extra></extra>'})
    # Show actual ullage totals on the bars, without requiring hover.
    totals = data['Total CPS Ullage']
    traces.append({'type':'scatter', 'mode':'text', 'name':'Total ullage (mbd)',
        'x':data.CPS.astype(str).tolist(),
        'y':[None if pd.isna(v) else float(v) for v in totals],
        'text':[None if pd.isna(v) else f'{v:,.1f}' for v in totals],
        'textposition':'top center', 'textfont':{'color':'#17365D', 'size':11},
        'showlegend':False, 'hoverinfo':'skip'})
    layout = chart_layout('Unused CPS capacity (ullage) by facility')
    layout.update(barmode='stack', height=460, legend={'orientation':'h', 'y':1.20, 'x':0},
                  yaxis={'title':'mbd', 'gridcolor':'#E8EDF3'})
    return {'data':traces, 'layout':layout}

def capacity_headroom_spec(data):
    traces = []
    for col, label, color in [('Maximum CPS Capacity', 'Facility capacity', '#B1BAC9'),
                               (ULLAGE[0], 'Potential from wells + manifolds', '#2563EB'),
                               ('IIC', 'Selected scenario injection', '#55BF9A')]:
        traces.append({'type':'bar', 'name':label, 'x':data.CPS.astype(str).tolist(),
            'y':[None if pd.isna(x) else float(x) for x in data[col]], 'marker':{'color':color},
            'hovertemplate':'%{x}<br>'+label+': %{y:,.1f} mbd<extra></extra>'})
    layout = chart_layout('Facility limit vs physical potential vs injection')
    layout.update(barmode='group', height=460)
    return {'data':traces, 'layout':layout}

def reservoir_comparison_spec(data):
    colors = ['#355C92', '#2463EB', '#F2AF35', '#58C996', '#64C7CD', '#9467bd']
    cols = ['RMT_TARGET', 'RMT_OE', 'AGREED_TARGET', 'WELL_CAPACITY', 'INJECTION', 'NETWORK_POTENTIAL']
    labels = ['RMT target', 'RMT+OE', 'Agreed target', 'Well capacity', 'Selected injection', 'Network-limited well potential']
    traces=[]
    for col, label, color in zip(cols, labels, colors):
        traces.append({'type':'bar', 'name':label, 'x':data.Reservoir.astype(str).tolist(),
                       'y':[None if pd.isna(x) else float(x) for x in data[col]], 'marker':{'color':color},
                       'hovertemplate':'%{x}<br>'+label+': %{y:,.1f} mbd<extra></extra>'})
    layout=chart_layout('Reservoir targets, capacity and injection')
    layout.update(barmode='group', height=480, margin={'l':65,'r':25,'t':80,'b':110})
    return {'data':traces, 'layout':layout}

def well_gap_spec(data, category, basis, title):
    # Gap is stored negative in the reference table and positive as lost mbd here.
    field = 'WELL_GAP_LOSS' if category == 'Reservoir' else GAPS[0]
    values = [None if pd.isna(x) else max(0., float(x) if category == 'Reservoir' else -float(x)) for x in data[field]]
    layout = chart_layout(title, horizontal=True)
    layout.update(height=max(360, len(values)*48+130))
    return {'data':[{'type':'bar', 'orientation':'h', 'name':'Unmet '+basis,
                     'x':values, 'y':data[category].astype(str).tolist(),
                     'marker':{'color':'#EC8569'},
                     'text':[None if x is None else f'{x:,.1f}' for x in values],
                     'textposition':'outside',
                     'hovertemplate':'%{y}<br>Well gap vs '+basis+': %{x:,.1f} mbd<extra></extra>'}],
            'layout':layout}

def show_charts(st, table, title, audit, reservoir_average=None, gap_basis='RMT+OE'):
    import plotly.io as pio
    cps = chart_data(table)
    reservoir = reservoir_summary(audit) if reservoir_average is None else reservoir_average
    figures = []
    st.caption('All rates are mbd. Waterfalls show contributions to the total. PWRI is counted as a separate facility, once only. Well capacity/injection may be partial if BMT is missing.')
    cps_tab, reservoir_tab, ullage_tab, mf_tab, gap_tab = st.tabs([
        'CPS charts', 'Reservoir charts', 'CPS Ullage', 'Manifold detail', 'Capacity shortfalls'])
    def draw(spec, key):
        figures.append(spec)
        st.plotly_chart(spec, use_container_width=True, key=title+key,
                        config={'displaylogo':False, 'toImageButtonOptions':{'format':'png', 'scale':2}})
    with cps_tab:
        for metric, label in [('RMT_TARGET','RMT Target'), ('RMT_OE','RMT+OE Target'),
                              ('WELL_CAPACITY','Well Capacity'), ('AGREED_TARGET','Agreed Target')]:
            draw(waterfall_spec(cps.CPS, cps[metric], label+' by CPS'), metric+'cps')
        draw(comparison_spec(cps, 'CPS', 'Maximum CPS Capacity', 'IIC', 'Demand, capacity and injection by CPS'), 'cpscompare')
        draw(well_gap_spec(cps, 'CPS', gap_basis, 'Well capacity gap to '+gap_basis+' by CPS'), 'cpswellgap')
    with reservoir_tab:
        for metric, label in [('RMT_TARGET','RMT Target'), ('RMT_OE','RMT+OE Target'),
                              ('WELL_CAPACITY','Well Capacity'), ('AGREED_TARGET','Agreed Target')]:
            draw(waterfall_spec(reservoir.Reservoir, reservoir[metric], label+' by Reservoir'), metric+'res')
        draw(reservoir_comparison_spec(reservoir), 'rescompare')
        draw(well_gap_spec(reservoir, 'Reservoir', gap_basis, 'Well capacity gap to '+gap_basis+' by Reservoir'), 'reswellgap')
        st.caption('Reservoir charts aggregate wells by the Reservoir column. Facility capacity is shared between reservoirs and is NOT assigned to individual reservoirs. Network-limited potential is allocated to wells proportionally under shared constraints.')
        st.dataframe(reservoir, use_container_width=True, hide_index=True)
    with ullage_tab:
        unusable = table.drop_duplicates('CPS')['Unutilizable CPS Ullage']
        if unusable.notna().all():
            st.metric('Total structurally unutilizable CPS/PWRI ullage (mbd)', f'{unusable.sum():,.1f}')
        st.markdown('**Unutilizable ullage** is the part of CPS/PWRI design capacity that available BMT well capacities and manifold constraints cannot support, **even if targets were increased**. The remaining unused capacity is split into target/operational headroom and optional efficiency losses.')
        if (table['Missing BMT wells'] > 0).any():
            st.warning('One or more facilities have unassessed BMT wells. Their ullage breakdown is n.a. rather than assuming missing capacity is zero. Select the missing-BMT policy to change this deliberately.')
        draw(ullage_spec(cps), 'ullagestack')
        draw(capacity_headroom_spec(cps), 'ullagecomparison')
        view_cols = ['CPS', 'Maximum CPS Capacity', *ULLAGE, 'IIC']
        st.dataframe(cps[view_cols], hide_index=True, use_container_width=True)
        st.download_button('Download CPS ullage CSV', cps[view_cols].to_csv(index=False).encode('utf-8-sig'),
                           'WI_CPS_Ullage.csv', 'text/csv', key=title+'ullagecsv')
        st.caption('Total ullage = facility capacity − selected final injection. Unutilizable = facility capacity − network-constrained physical well potential. Target/operational = physical potential − pre-efficiency injection. Efficiency effect = pre-efficiency − final injection. CPS and PWRI remain separate; do not double-count routed volumes. These values use the selected scenario, not an independently specified operating rate.')
    with mf_tab:
        chosen = st.selectbox('Select CPS / PWRI facility', ['All facilities', *cps.CPS.tolist()], key=title+'chartfacility')
        subset = table.copy() if chosen == 'All facilities' else table[table.CPS.eq(chosen)].copy()
        subset['Chart label'] = subset.Manifold.str.replace('PWRI1_', 'PWRI1 / ', regex=False).str.replace('PWRI2_', 'PWRI2 / ', regex=False)
        metric = st.selectbox('Manifold waterfall metric', ['RMT_TARGET','RMT_OE','WELL_CAPACITY','AGREED_TARGET'], key=title+'chartmetric')
        draw(waterfall_spec(subset['Chart label'], subset[metric], metric.replace('_',' ')+' by Manifold'), 'mfwaterfall')
        draw(comparison_spec(subset, 'Chart label', 'Maximum Manifold Capacity', 'Manifold Injection Rate', 'Demand and capacity by Manifold'), 'mfcompare')
        st.caption('Manifold injection uses the selected scenario after manifold constraint, before CPS constraint and any additional efficiency.')
    with gap_tab:
        draw(shortfall_spec(cps, 'CPS', GAPS[3], GAPS[4], 'CPS capacity shortfalls'), 'cpsgap')
        draw(shortfall_spec(table, 'Manifold', GAPS[1], GAPS[2], 'Manifold capacity shortfalls'), 'mfgap')
        st.caption('Facility shortfalls still use Agreed Target and RMT+OE; the new RMT basis selector changes only the WELL constraint gap. Shortfall = max(target demand − facility capacity, 0). CPS/manifold shortfalls overlap and must not be summed.')
    parts = ['<html><head><meta charset="utf-8"></head><body><h1>'+html.escape(title)+'</h1>']
    for i, spec in enumerate(figures):
        parts.append(pio.to_html(spec, full_html=False, include_plotlyjs=True if i == 0 else False))
    parts.append('</body></html>')
    st.download_button('Download interactive charts (HTML)', '\n'.join(parts).encode(),
                       'WI_Gap_Charts.html', 'text/html', key=title+'chartshtml')
    st.download_button('Download CPS chart data (CSV)', cps.to_csv(index=False).encode('utf-8-sig'),
                       'WI_CPS_Chart_Data.csv', 'text/csv', key=title+'chartcsv')

def gap_color(value, positive=False):
    if not np.isfinite(value):
        return '#eeeeee'
    severity = value if positive else -value
    if severity > 20:
        return '#d85b00'
    if severity > 5:
        return '#ffc000'
    return '#dff1ce'

def reference_html(table, total, title, threshold=20., gap_basis='RMT+OE'):
    cps_cols = {'Maximum CPS Capacity', 'CPS Injection Rate', 'IIC', 'INJ_RATE @ OE', *GAPS[-2:]}
    headers = [('CPS', 'CPS'), ('Manifold', 'Manifold'), ('RMT_TARGET','RMT Target'), *[(c, c) for c in METRICS[:9]],
               ('High Gap', 'High Gap'), *[(c, c) for c in GAPS]]
    def esc(s): return html.escape(str(s))
    out = ['<style>.gapwrap{overflow:auto}.gaptable{border-collapse:collapse;font:12px Arial;width:100%;min-width:1550px;background:white;color:#111}.gaptable th,.gaptable td{border:1px solid #777;padding:7px;text-align:right}.gaptable th{text-align:center;font-weight:bold}.gaptable tr.block td{border-top:2px solid #333}.gaptable .label{text-align:left}.gaptable .total{font-weight:bold;border-top:3px double #333}.gaptable caption{text-align:left;font: bold 18px Arial;padding:12px}</style><div class="gapwrap"><table class="gaptable">',
           '<caption>' + esc(title) + ' — mbd</caption>',
           '<thead><tr><th colspan="2"></th><th colspan="2">RESERVOIR</th><th colspan="2">WELL</th><th colspan="2">MANIFOLD</th><th colspan="2">CPS</th><th>FINAL INJ</th><th rowspan="2">INJ_RATE @ OE</th><th colspan="6">Capacity Gap_TABLE</th></tr><tr>',
           '<th>CPS</th><th>Manifold</th><th>RMT_TARGET</th><th>RMT_OE</th><th>WELL_CAPACITY</th><th>AGREED_TARGET</th><th>Maximum Manifold Capacity</th><th>Manifold Injection Rate</th><th>Maximum CPS Capacity</th><th>CPS Injection Rate</th><th>IIC</th><th>High Gap</th>' + ''.join('<th>' + esc('Well Constraint to '+('RMT Target' if gap_basis == 'RMT Target' else 'RMT+OE Target') if c == GAPS[0] else c) + '</th>' for c in GAPS) + '</tr></thead><tbody>']
    for fac, group in table.groupby('CPS', sort=False):
        for j, (_, row) in enumerate(group.iterrows()):
            out.append('<tr' + (' class="block"' if j == 0 else '') + '>')
            for key, label in headers:
                if key == 'CPS' or key in cps_cols:
                    if j > 0: continue
                    span = f' rowspan="{len(group)}"'
                else: span = ''
                if key == 'High Gap':
                    severity = max([0., -row[GAPS[0]] if pd.notna(row[GAPS[0]]) else 0.,
                                    *[row[g] if pd.notna(row[g]) else 0. for g in GAPS[1:]]])
                    value = 'HIGH' if severity >= threshold else ''
                elif key == 'Manifold':
                    value = row[key].removeprefix(fac + '_')
                else: value = row[key]
                color = f' style="background:{gap_color(float(value), positive=key != GAPS[0])}"' if key in GAPS else ''
                text = ('n.a.' if not np.isfinite(value) else f'{value:,.1f}') if isinstance(value, (float, np.floating)) else esc(value)
                out.append(f'<td{span}{color}>' + text + '</td>')
            out.append('</tr>')
    out.append('<tr class="total">')
    for key, _ in headers:
        value = total.get(key, '')
        text = ('n.a.' if not np.isfinite(value) else f'{value:,.1f}') if isinstance(value, (float, np.floating)) else esc(value)
        out.append('<td>' + text + '</td>')
    out.append('</tr></tbody></table></div>')
    return ''.join(out)

def excel_bytes(sheets):
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine='xlsxwriter', datetime_format='yyyy-mm-dd') as writer:
        for name, data in sheets.items():
            data.to_excel(writer, sheet_name=name[:31], index=False)
            ws = writer.sheets[name[:31]]
            ws.freeze_panes(1, 2)
            ws.autofilter(0, 0, len(data), len(data.columns)-1)
            ws.set_column(0, len(data.columns)-1, 19)
            ws.set_row(0, 42)
            header = writer.book.add_format({'bold':True, 'bg_color':'#17365D', 'font_color':'white', 'text_wrap':True, 'valign':'vcenter'})
            for j, label in enumerate(data.columns):
                ws.write(0, j, label, header)
            fmt = writer.book.add_format({'num_format':'0.0'})
            for j, label in enumerate(data.columns):
                if pd.api.types.is_numeric_dtype(data[label]):
                    ws.set_column(j, j, 20, fmt)
                if label in GAPS or label.startswith('Well Constraint to RMT'):
                    intervals = [(-1e12, -20.000001, '#d85b00'),(-20, -5.000001, '#ffc000'),(-5, 0, '#dff1ce')] if label == GAPS[0] or label.startswith('Well Constraint to RMT') else [(20.000001, 1e12, '#d85b00'),(5.000001, 20, '#ffc000'),(0, 5, '#dff1ce')]
                    for low, high, fill in intervals:
                        ws.conditional_format(1, j, len(data), j, {'type':'cell','criteria':'between','minimum':low,'maximum':high,'format':writer.book.add_format({'bg_color':fill})})
    return buffer.getvalue()

def show_report(st, table, total, title, audit, unknown, extra=None, gap_basis='RMT+OE'):
    full = pd.concat([table, total.to_frame().T], ignore_index=True)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric('RMT+OE demand (mbd)', f'{total.RMT_OE:,.1f}')
    incomplete = total.get('Missing BMT wells', 0) > 0
    c2.metric('Assessed-well injection (mbd)' if incomplete else 'Final injection (mbd)', 'n.a.' if pd.isna(total.IIC) else f'{total.IIC:,.1f}')
    gap = total[GAPS[2]]
    c3.metric('Manifold shortfall to RMT+OE (mbd)', 'n.a.' if pd.isna(gap) else f'{gap:,.1f}')
    c4.metric('Total unused CPS capacity (mbd)', 'n.a.' if pd.isna(total.get('Total CPS Ullage', np.nan)) else f"{total['Total CPS Ullage']:,.1f}")
    if incomplete:
        st.warning(f"All well demand is included. BMT is unknown for {total['Missing BMT wells']:g} wells: "
                   f"{total['Unassessed RMT+OE']:,.3f} mbd RMT+OE and {total['Unassessed Agreed Target']:,.3f} mbd Agreed Target. "
                   'Well gaps are unavailable where BMT is missing; injection/well-capacity figures cover assessed wells only. Direct manifold/CPS shortfalls still use all demand.')
    reference, charts, flat, well = st.tabs(['Reference-style gap table', 'Charts', 'Flat output table', 'Well audit'])
    markup = reference_html(table, total, title, gap_basis=gap_basis)
    with reference:
        st.markdown(markup, unsafe_allow_html=True)
        st.caption('Well gap: negative capacity loss. MF/CPS shortfall: positive max(total demand − facility capacity, 0), before well constraints. These independent shortfalls overlap and must not be added to the well gap. CPS totals are counted once per facility. Units: mbd.')
    gap_display = 'Well Constraint to '+('RMT Target' if gap_basis == 'RMT Target' else 'RMT+OE Target')
    display_full = full.rename(columns={GAPS[0]:gap_display})
    with flat:
        st.dataframe(display_full, use_container_width=True, hide_index=True)
    with well:
        st.dataframe(audit, use_container_width=True, hide_index=True)
        st.caption('Well audit rates are bbl/day. OE and agreed pipelines are calculated independently.')
    with charts:
        show_charts(st, table, title, audit, extra.get('Average reservoir mbd') if extra else None, gap_basis)
    if len(unknown):
        st.warning(f'{len(unknown)} wells have missing BMT capacity. Current missing-capacity policy applies to this report.')
        with st.expander('Missing BMT capacity wells'):
            st.dataframe(unknown[['Well', 'CPS', 'SW Manifold', 'Record Date', *RATE_COLUMNS.values()]], hide_index=True)
    sheets = {'Gap table mbd': display_full, 'Well audit bpd': audit,
              'CPS ullage mbd': chart_data(table)[['CPS', 'Maximum CPS Capacity', *ULLAGE, 'IIC']],
              'Reservoir mbd': extra['Average reservoir mbd'] if extra and 'Average reservoir mbd' in extra else reservoir_summary(audit),
              'Configuration':pd.DataFrame({'Setting':['Well gap basis', 'Injection scenario'],
                                             'Value':[gap_basis, audit['Selected scenario'].dropna().iloc[0] if audit['Selected scenario'].notna().any() else 'n.a.']})}
    if len(unknown): sheets['Missing BMT'] = unknown
    if extra: sheets.update(extra)
    st.download_button('Download Excel output tables', excel_bytes(sheets), 'WI_Gap_Analysis.xlsx',
                       'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet', key=title+'xlsx')
    st.download_button('Download gap table CSV', display_full.to_csv(index=False).encode('utf-8-sig'), 'WI_Gap_Table.csv', 'text/csv', key=title+'csv')
    st.download_button('Download reference-style HTML table', markup.encode(), 'WI_Gap_Table.html', 'text/html', key=title+'html')

def main():
    import streamlit as st
    st.set_page_config(page_title='WI System Gap Analysis', layout='wide')
    st.title('Water Injection — System Gap Analysis')
    st.caption('Well → manifold → CPS constraints, with date-dependent facility capacity and well-by-well decline forecast.')
    with st.sidebar:
        st.header('Upload data')
        uploaded = st.file_uploader('Browse: well targets CSV', type=['csv'])
        facility_file = st.file_uploader('Browse: facility.csv', type=['csv'])
        date_format = st.selectbox('Well CSV date format', ['Month/day/year', 'Day/month/year'])
        units = st.selectbox('Facility CAP units', ['mbd', 'bbl/day'])
        facility_date_format = st.selectbox('Facility date format (non-ISO)', ['Month/day/year', 'Day/month/year'])
        missing = st.selectbox('Missing BMT capacity', ['Exclude and flag', 'Treat as zero', 'Stop analysis'])
        route_label = st.selectbox('Water routing', ['Separate PWRI facilities (default)', 'SW only'],
            help='Separate PWRI assigns wells with PWRI Manifold to PWRI1 (CPS1/4/5) or PWRI2 (CPS7), once only. Their demand is removed from the corresponding SW group. PWRI uses its own facility limits.')
        route = 'PWRI priority' if route_label.startswith('Separate') else 'SW'
        st.caption('PWRI1 is shown separately with CPS1_SM, CPS4_SM and CPS5_SM branches when present in the well CSV.')
        scenario = st.selectbox('Displayed injection scenario', ['Agreed Target', 'RMT+OE'])
        gap_basis = st.selectbox('Well-capacity GAP calculated against', ['RMT+OE', 'RMT Target'],
            help='Choose only the reference used for the WELL constraint gap. Does not alter Agreed/RMT+OE facility shortfalls or the selected injection scenario.')
        efficiency = st.checkbox('Apply additional Well Efficiency to final injection', value=False,
            help='Leave off if Agreed Target already includes operational efficiency, to avoid applying it twice.')
    with st.expander('Calculation rules and input format'):
        st.markdown('''**facility.csv:** `FAC_NAME,Manifold,DATE,CAP`. CPS totals have blank Manifold; manifold rows use identifiers such as `CPS1_FM05` or `CPS1_SM`. Your earlier long capacity table is supported. Capacity is carried forward from its effective date; missing capacities stop the calculation.

For each target scenario separately: **well rate = min(target, BMT constrained capacity)**; sum by manifold, cap at manifold capacity; sum by CPS, cap at CPS capacity. Constraint reductions are allocated proportionally back to wells. Final injection applies Well Efficiency after the constraints when enabled. Facility constraints are assumed to use the same pre-efficiency rate basis as BMT capacity and RMT+OE.

Well gap uses the separate selected basis: min(RMT or RMT+OE, BMT constrained capacity) − chosen target. The four manifold/CPS shortfall columns use max(total target demand − facility capacity, 0), independently of the BMT-constrained injection pipeline. CPS demand is summed across its branches. These shortfalls overlap and must not be added together or to well losses. Incremental constraint losses remain in the well audit. `RMT_OE` is the RMT+OE export column. `IIC` and `INJ_RATE @ OE` show the selected scenario's final injection. The screenshot does not provide underlying formulas; these explicit rules reproduce its layout and constraint stages. High Gap flags losses of at least 20 mbd; colors use 5 and 20 mbd thresholds. Ullage is calculated independently of the chosen well gap basis from the available BMT capacity, manifold limits and CPS capacity; shortages due to BMT/mf constraints are physically unusable with the current wells.

Historical dates use only well records dated on/before the selection. A latest-only CSV cannot reconstruct earlier records. Forecast holds targets, efficiency and routing fixed at the chosen baseline, declines each well's BMT capacity from forecast start, and updates facility limits through time. Blank BMT values are never silently inferred from targets. All demand remains included when BMT is missing; only capacity assessment excludes those wells. Affected well gaps are unavailable; direct facility shortfalls remain calculable when facility capacity is available. Quality columns quantify unassessed BMT demand. PM01 is mapped to the corresponding CPS SM manifold for PWRI routing.''')
    if not uploaded or not facility_file:
        st.info('Upload both CSV files to begin. Well rates must be bbl/day. Facility CAP defaults to mbd.')
        return
    try:
        wells = prepare_wells(read_csv(uploaded), date_format == 'Day/month/year', route)
        facilities = prepare_facilities(read_csv(facility_file), units, facility_date_format == 'Day/month/year')
        first, last = wells.date.min().date(), wells.date.max().date()
        st.caption(f'{wells.Well.nunique()} wells; uploaded record dates {first} to {last}.')
        if wells.Well.nunique() == len(wells):
            st.info('This upload contains one record per well. Earlier dates may omit wells; later dates carry this snapshot forward. Upload historical records to see actual well-target changes.')
        snapshot_tab, forecast_tab = st.tabs(['Date / historical gap analysis', 'Decline forecast / period average'])
        kwargs = dict(missing=missing, scenario=scenario, efficiency=efficiency, gap_basis=gap_basis)
        with snapshot_tab:
            selected = st.date_input('Analysis date', last, min_value=first, max_value=last+timedelta(days=3650))
            t, total, audit, unknown = calculate(wells, facilities, selected, **kwargs)
            omitted = wells.Well.nunique()-audit.Well.nunique()
            if omitted: st.warning(f'{omitted} wells have no record on/before this date and are absent.')
            show_report(st, t, total, f'Gap analysis {selected}', audit, unknown, gap_basis=gap_basis)
            if st.checkbox('Compare with another date'):
                other = st.date_input('Comparison date', first, min_value=first, max_value=last+timedelta(days=3650))
                old, old_total, _, _ = calculate(wells, facilities, other, **kwargs)
                diff = t.set_index(['CPS','Manifold'])[METRICS].subtract(old.set_index(['CPS','Manifold'])[METRICS], fill_value=0).reset_index()
                st.caption('Selected date minus comparison date, mbd. Absent groups are treated as zero for the comparison only.')
                st.dataframe(diff, hide_index=True, use_container_width=True)
                st.download_button('Download date comparison CSV', diff.to_csv(index=False).encode('utf-8-sig'), 'Date_Comparison.csv', 'text/csv')
        with forecast_tab:
            with st.form('forecast_settings'):
                c1,c2,c3 = st.columns(3)
                baseline = c1.date_input('Well baseline date', last, min_value=first, max_value=last)
                start = c2.date_input('Forecast start date', last, min_value=first, max_value=last+timedelta(days=3650))
                end = c3.date_input('Forecast end date', last+timedelta(days=365), min_value=first, max_value=last+timedelta(days=3650))
                c1,c2 = st.columns(2)
                decline = c1.number_input('BMT constrained capacity decline (%)', min_value=0., max_value=99.9, value=6., step=0.5)
                basis = c2.selectbox('Decline basis', ['Annual effective', 'Monthly effective'])
                run = st.form_submit_button('Run forecast and average gap analysis')
            st.caption('Annual: capacity(t) = baseline capacity × (1 − decline/100)^(days/365.25). Monthly uses days/30.4375. The first forecast day has no decline. The average includes every calendar day from start to end.')
            if run:
                if end < start or baseline > start:
                    st.error('End must be on/after start, and baseline must be on/before forecast start.')
                else:
                    with st.spinner('Calculating daily well, manifold and CPS constraints…'):
                        result = forecast(wells, facilities, baseline, start, end, decline, 365.25 if basis.startswith('Annual') else 30.4375, **kwargs)
                    st.session_state['forecast_result'] = result
                    st.session_state['forecast_config'] = ("charts-v2", uploaded.getvalue(), facility_file.getvalue(), date_format, units, facility_date_format, route, missing, scenario, efficiency, gap_basis)
                    st.session_state['forecast_title'] = f'Average gap analysis {start} to {end} ({decline:g}% {basis.lower()})'
            config = ("charts-v2", uploaded.getvalue(), facility_file.getvalue(), date_format, units, facility_date_format, route, missing, scenario, efficiency, gap_basis)
            if st.session_state.get('forecast_config') == config:
                avg, total, daily, trend, audit, unknown, reservoir_average = st.session_state['forecast_result']
                st.line_chart(trend.set_index('DATE')[['RMT_OE', 'WELL_CAPACITY', 'AGREED_TARGET', 'IIC']])
                st.caption('Trend units: mbd. Targets remain fixed; declining well capacity and changing facility limits drive the forecast.')
                show_report(st, avg, total, st.session_state['forecast_title'], audit, unknown, {'Daily manifold mbd': daily, 'Daily system mbd': trend, 'Average reservoir mbd': reservoir_average}, gap_basis=gap_basis)
                st.caption('The well audit shows the forecast end date. Average gap tables are means of daily constrained results, not constraints applied to mean capacities.')
            elif 'forecast_result' in st.session_state:
                st.info('Inputs changed. Run the forecast again to refresh the results.')
    except (ValueError, KeyError, pd.errors.ParserError) as exc:
        st.error(str(exc))

if __name__ == '__main__':
    main()
