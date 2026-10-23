"""Network-constrained Water Injection allocation optimizer.

Rates are bbl/day. Pattern VoW is a screening coefficient, *not* incremental oil
prediction. No cross-CPS routing. Re-route requires an explicit approved edge.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Optional
import numpy as np
import pandas as pd
from scipy.optimize import milp, Bounds, LinearConstraint
from scipy.sparse import coo_matrix

from wi_gap import normal_cps, normal_manifold, capacity_map, latest_wells, calculate
from pattern_core import (standardise, analytical_waterflood, support_scenario,
                          BP_ANALYTICAL_METHOD, DEFAULT_BO)


def norm_res(value):
    s = re.sub(r'[^A-Z0-9]', '', str(value).upper())
    return s if s not in ('','NAN','UNSPECIFIED','NONE') else ''


def norm_mf(value, cps=None):
    s = str(value or '').strip()
    if not s or s.upper() in ('UNMAPPED','NONE','NAN','UNKNOWN'):
        return ''
    if s.upper().startswith('PWRI'):
        return ''  # PWRI kept completely separate from ordinary CPS routing.
    if re.search(r'(FM|SM|PM)\s*[-_ ]*\d*',s,re.I) is None:
        return ''
    prefix = normal_cps(s)
    if prefix == 'UNASSIGNED' and cps:
        prefix = normal_cps(cps)
    if prefix == 'UNASSIGNED':
        return ''
    try:
        return normal_manifold(prefix + '_' + s)
    except ValueError:
        return ''


def prepare_vow(pattern_raw: pd.DataFrame, method: str = BP_ANALYTICAL_METHOD,
                bo: Optional[dict] = None, bwi: float = 1., bwp: float = 1.,
                delta_gor: float = 0., bg: float = .005,
                support_share: float = 20.) -> tuple[pd.DataFrame, list[str]]:
    """Use original Pattern Analytics exact standardisation & BP/support definitions."""
    df, missing, _ = standardise(pattern_raw)
    if missing:
        raise ValueError('Pattern input lacks ' + ', '.join(missing))
    if df.empty:
        raise ValueError('Pattern input contains no valid rows.')
    errors=[]
    df['CPS'] = df['CPS'].map(normal_cps)
    df['mf_key'] = [norm_mf(m,c) for m,c in zip(df['Manifold'],df['CPS'])]
    mismatched = df['mf_key'].ne('') & (df['mf_key'].str.extract(r'^(CPS\d+)',expand=False)!=df['CPS'])
    if mismatched.any():
        errors.append(f'{int(mismatched.sum())} Pattern records have inconsistent CPS / Manifold IDs and were excluded from optimization matching (all levels).')
        df=df.loc[~mismatched].copy()
    df['Existing Supported'] = support_scenario(df,method,support_share)['Supported']
    df = analytical_waterflood(df, bo or DEFAULT_BO,bwi,bwp,delta_gor,bg)
    df['Supported'] = df['BP Oil Supported (stb)'] if method==BP_ANALYTICAL_METHOD else df['Existing Supported']
    for c in ['Injection','Supported']:
        df[c]=pd.to_numeric(df[c],errors='coerce')
    invalid = df[['Injection','Supported']].isna().any(axis=1) | df['Injection'].le(0) | df['Injection'].lt(0) | df['Supported'].lt(0)
    if invalid.any():
        errors.append(f'{int(invalid.sum())} Pattern records excluded from VoW coefficients due to non-positive injection or invalid supported-oil volumes.')
    df=df.loc[~invalid].copy()
    df['res_key']=df['Reservoir'].map(norm_res)
    df['pat_key']=df['Pattern'].astype(str).str.strip().str.upper()
    df['raw_vow']=df['Supported']/df['Injection']
    return df, errors


def _indexes(df,names):
    g=df.groupby(names,dropna=False)[['Supported','Injection']].sum(min_count=1).reset_index()
    g=g.loc[g.Injection.gt(0)].copy()
    return {tuple(rec[:-1]):float(rec[-1]) for rec in
            g.assign(score=g.Supported/g.Injection)[names+['score']].itertuples(index=False,name=None)}


def well_scores(wells:pd.DataFrame, pattern:pd.DataFrame,
                well_pattern_map:Optional[pd.DataFrame]=None,
                fallback='Manifold + CPS') -> tuple[pd.DataFrame, list[str]]:
    """Assign score to original well/pattern, never the destination manifold."""
    notes=[]
    indexed={
        'pat':_indexes(pattern,['pat_key','res_key']),
        'mfres':_indexes(pattern.loc[pattern.mf_key.ne('')],['mf_key','res_key']),
        'mf':_indexes(pattern.loc[pattern.mf_key.ne('')],['mf_key']),
        'cpsres':_indexes(pattern,['CPS','res_key']),
        'cps':_indexes(pattern,['CPS']),
    }
    direct={}
    if well_pattern_map is not None and not well_pattern_map.empty:
        m=well_pattern_map.copy()
        names={re.sub(r'[^a-z0-9]','',str(c).lower()):c for c in m.columns}
        if not {'well','pattern'}.issubset(names):
            raise ValueError('Well↔Pattern mapping must contain Well, Pattern columns; optional Reservoir, Weight.')
        wellcol,patcol=names['well'],names['pattern']
        rescol=names.get('reservoir'); wcol=names.get('weight')
        m=m.dropna(subset=[wellcol,patcol]).copy()
        m['_well']=m[wellcol].astype(str).str.strip().str.upper()
        m['_pat']=m[patcol].astype(str).str.strip().str.upper()
        if wcol:
            m['_weight']=pd.to_numeric(m[wcol],errors='coerce')
            if m['_weight'].isna().any() or m['_weight'].lt(0).any():
                raise ValueError('Well↔Pattern Weight must be numeric and nonnegative.')
        else:
            ct=m.groupby('_well')['_pat'].transform('size')
            if (ct>1).any():
                raise ValueError('Multiple patterns for one well: provide a Weight column with fractions summing to 1.')
            m['_weight']=1.
        totals=m.groupby('_well')['_weight'].sum()
        if ((totals-1).abs()>0.001).any():
            raise ValueError('Pattern mapping weights per Well must sum to 1.')
        for key, group in m.groupby('_well'):
            wr=wells.loc[wells.Well.astype(str).str.upper().eq(key)]
            if wr.empty:
                notes.append(f'Well {key} in pattern mapping is not in current WI well snapshot.')
                continue
            res=norm_res(wr.iloc[0].get('Reservoir',''))
            score=0.; matched=True
            for _,row in group.iterrows():
                rkey=norm_res(row[rescol]) if rescol else res
                raw=indexed['pat'].get((row['_pat'],rkey))
                if raw is None: matched=False; break
                score+=float(row['_weight'])*raw
            if matched:direct[key]=score
            else:notes.append(f'No unique Pattern + Reservoir score for well {key}; fallback attempted.')
    output=wells.copy()
    scores=[]; levels=[]; qualities=[]
    for rec in output.to_dict('records'):
        well=str(rec['Well']).upper(); f=rec['facility']; mf=rec['manifold']; res=norm_res(rec.get('Reservoir',''))
        if well in direct:
            a=direct[well]; level='Well ↔ Pattern'; conf=1.
        else:
            lookups=[('mfres',(mf,res),'Manifold + Reservoir',.9),('mf',(mf,),'Manifold',.75)]
            if fallback=='Manifold + CPS':
                lookups += [('cpsres',(f,res),'CPS + Reservoir',.55),('cps',(f,),'CPS only',.35)]
            a=np.nan; level='Unscored'; conf=0.
            for typ, key, label, confidence in lookups:
                if key in indexed[typ]:
                    a=indexed[typ][key]; level=label; conf=confidence; break
        scores.append(a);levels.append(level);qualities.append(conf)
    output['VoW raw']=scores
    output['VoW source']=levels
    output['Score confidence']=qualities
    return output,notes


def approved_routes(wells:pd.DataFrame, caps:dict, connection_raw:Optional[pd.DataFrame]):
    """Only explicit, approved same-CPS route changes; PWRI never eligible."""
    notes=[]
    groups={str(w):[(f,m)] for w,f,m in wells[['Well','facility','manifold']].itertuples(index=False,name=None)}
    if connection_raw is None or connection_raw.empty:
        return groups,notes
    d=connection_raw.copy()
    lookup={re.sub(r'[^a-z0-9]','',str(c).lower()):c for c in d.columns}
    needed=['well','allowedmanifold','approved']
    if not all(k in lookup for k in needed):
        raise ValueError('Approved route CSV needs: Well, Allowed Manifold, Approved (Yes/No); optional CPS.')
    wellcol,manicol,appcol=[lookup[x] for x in needed]
    cpscol=lookup.get('cps')
    valid={'yes','true','approved','1','y'}
    for _,rec in d.iterrows():
        w=str(rec[wellcol]).strip(); value=str(rec[manicol]).strip()
        if w not in groups:
            notes.append(f'Ignored route for unknown well {w}');continue
        f,home=groups[w][0]
        if not str(rec[appcol]).strip().lower() in valid:
            continue
        if f.startswith('PWRI'):
            notes.append(f'Ignored {w} rerouting: PWRI is separate and locked.');continue
        if cpscol is not None and normal_cps(rec[cpscol]) != f:
            notes.append(f'Ignored {w}: CSV CPS does not match original CPS.');continue
        target=norm_mf(value,f)
        if not target or not target.startswith(f+'_'):
            notes.append(f'Ignored {w} -> {value}: cross-CPS or unrecognised manifold.');continue
        if (f,target) not in caps:
            notes.append(f'Ignored {w} -> {target}: no dated capacity record for target manifold.');continue
        if (f,target) not in groups[w]:
            groups[w].append((f,target))
    return groups,notes


@dataclass
class OptimizerSettings:
    mode: str = 'Conserve water'          # or 'Additional supply'
    additional_supply_bpd: float = 0.
    extra_by_facility: dict = field(default_factory=dict)  # explicitly allotted additions; never reallocate CPS water implicitly
    demand_limit: str = 'BMT capacity'     # or RMT+OE / Agreed Target
    max_increase_pct: float = 200.
    max_decrease_pct: float = 100.
    allow_zero_start: bool = False
    min_score_confidence: float = .70
    change_penalty_vow: float = .002
    route_penalty_vow: float = .01
    solver_time_limit_sec: float = 45.


def _make_optimisation(wells, caps, routes, config, score_objective=True,
                       total_override=None):
    d=wells.reset_index(drop=True).copy()
    if d.Well.duplicated().any():
        raise ValueError('Duplicate wells in optimization snapshot.')
    needed=['baseline','capacity','facility','manifold','VoW raw','Score confidence']
    if any(x not in d.columns for x in needed):
        raise ValueError('Optimizer input missing: '+str([x for x in needed if x not in d]))
    d['baseline']=pd.to_numeric(d.baseline,errors='coerce')
    if d.baseline.isna().any() or d.baseline.lt(0).any():
        raise ValueError('Baseline injection must be complete, nonnegative bbl/day values.')
    limit=d.capacity.astype(float).clip(lower=0)
    if config.demand_limit=='RMT+OE':limit=np.minimum(limit,d.oe.clip(lower=0))
    elif config.demand_limit=='Agreed Target':limit=np.minimum(limit,d.agreed.clip(lower=0))
    d['physical_upper']=limit
    # A well's existing VoW coefficient is not changed by its destination manifold.
    scored=d['VoW raw'].notna() & d['Score confidence'].ge(config.min_score_confidence) & d['VoW raw'].ge(0)
    inc=d.baseline*(1+config.max_increase_pct/100)
    if config.allow_zero_start:
        inc=np.where(d.baseline.eq(0),limit,inc)
    lb=(d.baseline*(1-config.max_decrease_pct/100)).clip(lower=0)
    ub=np.minimum(limit,inc)
    # Unknown VoW holds current allocation, except exceeding verified well capacity.
    unknown_baseline=np.minimum(d.baseline,limit)
    lb=np.where(scored,lb,unknown_baseline)
    ub=np.where(scored,ub,unknown_baseline)
    ub=np.maximum(0,ub)
    if (lb>ub+1e-7).any():
        offenders=d.loc[lb>ub+1e-7,'Well'].tolist()
        raise ValueError('Well minimum rate exceeds cap/ramp at: '+', '.join(map(str,offenders[:12]))+'. Relax decrease limit or investigate overloaded wells.')
    d['min_rate']=lb;d['max_rate']=ub;d['score_valid']=scored
    bounds_l=[];bounds_u=[];c=[];integer=[];x_cols={};y_cols={};dev_cols={};
    def new_var(lo,up,cost=0,integrality=0):
        k=len(c);bounds_l.append(float(lo));bounds_u.append(float(up));c.append(float(cost));integer.append(integrality);return k
    for i, rec in enumerate(d.itertuples(index=False)):
        well=d.at[i,'Well']; candidates=routes[well]
        for f,m in candidates:
            if (f,m) not in caps:raise ValueError(f'Missing manifold capacity {(f,m)}')
            if (f,'') not in caps:raise ValueError(f'Missing CPS capacity {f}')
            score=float(d.at[i,'VoW raw']) if scored.iloc[i] else 0.
            x_cols[(i,f,m)]=new_var(0,ub[i],-(score if score_objective else 1.))
        if len(candidates)>1:
            for f,m in candidates:
                other=(f,m)!=(d.at[i,'facility'],d.at[i,'manifold'])
                cost=(config.route_penalty_vow*max(float(d.at[i,'baseline']),1)) if other and score_objective else 0
                y_cols[(i,f,m)]=new_var(0,1,cost,1)
        if score_objective:
            dev_cols[i]=(new_var(0,np.inf,config.change_penalty_vow),
                         new_var(0,np.inf,config.change_penalty_vow))
    n=len(c)
    row_id=[];col_id=[];data=[];lows=[];ups=[]
    def row(parts,low=-np.inf,up=np.inf):
        k=len(lows)
        for j,v in parts:
            if v:
                row_id.append(k);col_id.append(j);data.append(float(v))
        lows.append(float(low));ups.append(float(up))
    for i,rec in d.iterrows():
        cand=routes[rec.Well]
        xs=[x_cols[(i,f,m)] for f,m in cand]
        row([(j,1) for j in xs],rec.min_rate,rec.max_rate)
        if len(cand)>1:
            row([(y_cols[(i,f,m)],1) for f,m in cand],1,1)
            for f,m in cand:
                row([(x_cols[(i,f,m)],1),(y_cols[(i,f,m)],-float(rec.max_rate))],up=0)
        if score_objective:
            dp,dn=dev_cols[i]
            row([(j,1) for j in xs]+[(dp,-1),(dn,1)],float(rec.baseline),float(rec.baseline))
    all_x=list(x_cols.items())
    for (f,m), limit in caps.items():
        cols=[k for (i,fac,mf),k in all_x if fac==f and (not m or mf==m)]
        if cols:
            row([(k,1) for k in cols],up=limit)
    # Water supply is isolated BY FACILITY. Conserve within CPS/PWRI, never
    # borrow water from a different CPS just because it has spare well capacity.
    # More water at a CPS requires an explicit extra supply allotment for that CPS.
    if score_objective:
        for facility, group in d.groupby('facility'):
            baseline_fac=float(group.baseline.sum())
            extra_fac=float(config.extra_by_facility.get(facility, 0.))
            if extra_fac<0:raise ValueError(f'Negative extra water allocation for {facility}')
            fac_x=[j for (i,f,m),j in all_x if f==facility]
            if config.mode=='Conserve water':
                row([(j,1) for j in fac_x], low=baseline_fac, up=baseline_fac)
            else:
                row([(j,1) for j in fac_x], up=baseline_fac+extra_fac)
    available=(float(d.baseline.sum()) if config.mode=='Conserve water'
               else float(d.baseline.sum())+sum(float(v) for v in config.extra_by_facility.values()))
    if total_override is not None:
        available=total_override
    if score_objective:
        row([(j,1) for _,j in all_x],
            low=available if config.mode=='Conserve water' else -np.inf,up=available)
    # Physical maximum scenario is unconstrained by supply, but constrained by wells/manifolds/CPS.
    A=coo_matrix((data,(row_id,col_id)),shape=(len(lows),n)).tocsr()
    result=milp(c=np.array(c),integrality=np.array(integer),
                bounds=Bounds(bounds_l,bounds_u),
                constraints=LinearConstraint(A,np.array(lows),np.array(ups)),
                options={'time_limit':float(config.solver_time_limit_sec),'mip_rel_gap':0.001})
    if result.x is None or result.status not in (0,1):
        raise ValueError(f'Optimization infeasible or solver failure ({result.message}). Check available water, minimum flow, connectivity, and facility capacities.')
    if result.status==1:
        # Time-limited feasible incumbent is usable but cannot be claimed optimal.
        status='Feasible (time limit; not proven optimal)'
    else:
        status='Optimal'
    out=d.copy();out['Optimized injection']=0.;out['Recommended manifold']=out['manifold']
    if score_objective:
        for i,rec in d.iterrows():
            cand=routes[rec.Well]
            vals=[max(0.,float(result.x[x_cols[(i,f,m)]])) for f,m in cand]
            out.at[i,'Optimized injection']=sum(vals)
            if len(cand)>1:
                yvals=[float(result.x[y_cols[(i,f,m)]]) for f,m in cand]
                selected=int(np.argmax(yvals))
                out.at[i,'Recommended manifold']=cand[selected][1] if sum(vals)>=1 else d.at[i,'manifold']
            else:out.at[i,'Recommended manifold']=cand[0][1]
        out['Delta injection']=out['Optimized injection']-out['baseline']
    return (out,status,float(sum(float(result.x[j]) for _,j in all_x)))


def optimize(wells:pd.DataFrame, caps:dict, routes:dict, settings:OptimizerSettings):
    recommendations,status,planned=_make_optimisation(wells,caps,routes,settings,True)
    physical_max_status='Not calculated'
    # Estimate attainable maximum using same well/ramp/route controls with no supply restriction.
    try:
        _, physical_max_status, physical_max=_make_optimisation(wells,caps,routes,settings,False)
    except ValueError:
        physical_max=np.nan
    recommendations['Action']=np.select([
        recommendations['Recommended manifold'].ne(recommendations.manifold),
        recommendations['Delta injection'].gt(1),
        recommendations['Delta injection'].lt(-1)],
        ['Approved reroute + adjust rate','Increase injection','Decrease injection'],default='Maintain')
    recommendations['Oil-supported proxy change (stb/d)']=(recommendations['Delta injection']*recommendations['VoW raw']).where(recommendations['score_valid'],0)
    source_group=(recommendations.groupby(['facility','manifold'],dropna=False)
                  .agg(**{'Current injection':('baseline','sum')}).reset_index().rename(columns={'facility':'CPS','manifold':'Manifold'}))
    dest_group=(recommendations.groupby(['facility','Recommended manifold'],dropna=False)
                .agg(**{'Optimized injection':('Optimized injection','sum'),
                        'Oil-supported proxy change (stb/d)':('Oil-supported proxy change (stb/d)','sum')})
                .reset_index().rename(columns={'facility':'CPS','Recommended manifold':'Manifold'}))
    mf=pd.merge(source_group,dest_group,on=['CPS','Manifold'],how='outer').fillna(0)
    mf['Capacity']=[caps.get((f,m),np.nan) for f,m in zip(mf.CPS,mf.Manifold)]
    mf['Before ullage']=mf.Capacity-mf['Current injection']
    mf['After ullage']=mf.Capacity-mf['Optimized injection']
    mf['Change']=mf['Optimized injection']-mf['Current injection']
    cps_before=recommendations.groupby('facility').baseline.sum()
    cps_after=recommendations.groupby('facility')['Optimized injection'].sum()
    fac=pd.DataFrame({'CPS':sorted(set(cps_before.index)|set(cps_after.index))})
    fac['Current injection']=fac.CPS.map(cps_before).fillna(0)
    fac['Optimized injection']=fac.CPS.map(cps_after).fillna(0)
    fac['Capacity']=[caps.get((c,''),np.nan) for c in fac.CPS]
    fac['Before ullage']=fac.Capacity-fac['Current injection']
    fac['After ullage']=fac.Capacity-fac['Optimized injection']
    fac['Change']=fac['Optimized injection']-fac['Current injection']
    return {'wells':recommendations, 'manifold':mf, 'cps':fac,
            'status':status,'physical_max':physical_max,
            'physical_max_status':physical_max_status,
            'planned_total':planned}
