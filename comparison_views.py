"""Before/After decision analysis from the REAL MILP well allocation.

All internal rate values are bbl/day; result display uses mbd. Never recalculate
an 'after' scenario from invented rates. VoW is a FIXED screening coefficient,
not a response curve or a production forecast. Source pattern-volume records are
not silently used as comparable daily well injection measurements.
"""
from __future__ import annotations
import re
from typing import Optional
import numpy as np
import pandas as pd
import plotly.graph_objects as go

NAVY="#173553"
BLUE="#2A6EBB"
GREEN="#069B78"
ORANGE="#ED9B37"
RED="#D34C5B"
GREY="#728397"
LIGHT="#E6EEF5"


def _sum(series):
    return float(pd.to_numeric(series, errors="coerce").sum())


def _group_values(w, key):
    """Aggregate targets and BMT deliverability in a physical routing state."""
    if w.empty:
        return pd.DataFrame(columns=[*key,"RMT Target","RMT+OE Target","Agreed Target","BMT Well Capacity","Wells"])
    agg=(w.groupby(key,dropna=False)
         .agg(**{'RMT Target':('rmt','sum'), 'RMT+OE Target':('oe','sum'),
                 'Agreed Target':('agreed','sum'),
                 'BMT Well Capacity':('capacity','sum'), 'Wells':('Well','nunique')})
         .reset_index())
    return agg


def capacity_gap_comparison(wells: pd.DataFrame, caps: dict,
                            source_gap: Optional[pd.DataFrame]=None,
                            gap_target='RMT+OE Target') -> tuple[pd.DataFrame,pd.DataFrame]:
    """Recomputed delivery gap by ORIGINAL and RECOMMENDED physical manifolds.

    Target and BMT ownership move with a rerouted well. CPS totals NEVER move.
    Structural shortfall = demand above capacity (independent of actual injection).
    Delivery gap = unmet target at the displayed date/rate allocation.
    """
    if gap_target not in ('RMT Target','RMT+OE Target','Agreed Target'):
        raise ValueError('Invalid gap target')
    w=wells.copy()
    if w.Well.duplicated().any():
        raise ValueError('Comparison requires one dated record per well.')
    w['Before Manifold']=w['manifold']; w['After Manifold']=w['Recommended manifold']
    for col in ['rmt','oe','agreed','capacity','baseline','Optimized injection']:
        w[col]=pd.to_numeric(w[col],errors='coerce')
    if w[['rmt','oe','agreed','baseline','Optimized injection']].isna().any().any():
        raise ValueError('Missing well rates/target values for comparison.')
    w['Selected target']=w['oe'] if gap_target=='RMT+OE Target' else w['rmt'] if gap_target=='RMT Target' else w['agreed']
    w['Well Physical Shortfall']=(w['Selected target']-w['capacity']).clip(lower=0)
    before=w.rename(columns={'facility':'CPS','Before Manifold':'Manifold'}).copy()
    after=w.rename(columns={'facility':'CPS','After Manifold':'Manifold'}).copy()
    kb=['CPS','Manifold']
    bt=_group_values(before,kb).rename(columns={x:x+' Before' for x in ['RMT Target','RMT+OE Target','Agreed Target','BMT Well Capacity','Wells']})
    at=_group_values(after,kb).rename(columns={x:x+' After' for x in ['RMT Target','RMT+OE Target','Agreed Target','BMT Well Capacity','Wells']})
    b_wgap=before.groupby(kb,dropna=False)['Well Physical Shortfall'].sum().rename('Well Physical Shortfall Before').reset_index()
    a_wgap=after.groupby(kb,dropna=False)['Well Physical Shortfall'].sum().rename('Well Physical Shortfall After').reset_index()
    b_inj=before.groupby(kb,dropna=False)['baseline'].sum().rename('Injection Before').reset_index()
    a_inj=after.groupby(kb,dropna=False)['Optimized injection'].sum().rename('Injection After').reset_index()
    frames=[bt,at,b_inj,a_inj,b_wgap,a_wgap]
    keys=set()
    for f in frames:
        keys.update(map(tuple,f[kb].itertuples(index=False,name=None)))
    keys.update((f,m) for (f,m) in caps if m)
    out=pd.DataFrame(sorted(keys),columns=kb)
    for f in frames:out=out.merge(f,on=kb,how='left')
    nums=[c for c in out if c not in kb]
    out[nums]=out[nums].fillna(0)
    out['MF Capacity']=[float(caps.get((f,m),np.nan)) for f,m in out[kb].itertuples(index=False,name=None)]
    if out['MF Capacity'].isna().any():
        raise ValueError('Missing manifold maximum capacity in Comparison.')
    for state in ('Before','After'):
        target=out[f'{gap_target} {state}']
        inj=out[f'Injection {state}']
        out[f'Delivery Gap {state}']=(target-inj).clip(lower=0)
        out[f'Target Surplus {state}']=(inj-target).clip(lower=0)
        out[f'MF Target Shortfall {state}']=(target-out['MF Capacity']).clip(lower=0)
        out[f'MF Ullage {state}']=out['MF Capacity']-inj
        # BMT-only well limit, independent of existing manifold capacity.
    out['Injection Change']=out['Injection After']-out['Injection Before']
    # Preserve source report's missing assessment information (raw gap report is mbd).
    if source_gap is not None and not source_gap.empty:
        cols=['CPS','Manifold','Missing BMT wells','Unassessed RMT Target','Unassessed RMT+OE','Unassessed Agreed Target']
        present=[c for c in cols if c in source_gap]
        if 'CPS' in present and 'Manifold' in present:
            out=out.merge(source_gap[present].drop_duplicates(kb),on=kb,how='left')
            if 'Missing BMT wells' in out:
                out['Missing BMT wells']=out['Missing BMT wells'].fillna(0)
    # CPS aggregation deliberately independent of MF join so CPS limits counted exactly once.
    bfac=_group_values(before,['CPS']).rename(columns={x:x+' Before' for x in ['RMT Target','RMT+OE Target','Agreed Target','BMT Well Capacity','Wells']})
    afac=_group_values(after,['CPS']).rename(columns={x:x+' After' for x in ['RMT Target','RMT+OE Target','Agreed Target','BMT Well Capacity','Wells']})
    bef=before.groupby('CPS')['baseline'].sum().rename('Injection Before').reset_index()
    aft=after.groupby('CPS')['Optimized injection'].sum().rename('Injection After').reset_index()
    fac=pd.DataFrame({'CPS':sorted(set(bef.CPS)|set(aft.CPS))})
    for f in (bfac,afac,bef,aft):fac=fac.merge(f,on='CPS',how='left')
    fac['CPS Capacity']=[float(caps.get((f,''),np.nan)) for f in fac.CPS]
    if fac['CPS Capacity'].isna().any():raise ValueError('Missing CPS maximum capacity in Comparison.')
    for state in ['Before','After']:
        target=fac[f'{gap_target} {state}']; inj=fac[f'Injection {state}']
        fac[f'Delivery Gap {state}']=(target-inj).clip(lower=0)
        fac[f'Target Surplus {state}']=(inj-target).clip(lower=0)
        fac[f'CPS Target Shortfall {state}']=(target-fac['CPS Capacity']).clip(lower=0)
        fac[f'CPS Ullage {state}']=fac['CPS Capacity']-inj
    fac['Injection Change']=fac['Injection After']-fac['Injection Before']
    # Structural limitation of well BMT capacities: do not confuse with delivery gap.
    well_shortfall=w.groupby('facility')['Well Physical Shortfall'].sum()
    fac['Well Physical Shortfall']=fac.CPS.map(well_shortfall).fillna(0)
    return out,fac


def allocation_rollup(wells: pd.DataFrame, level='CPS',
                      only_scored=True)->pd.DataFrame:
    """Original vs optimized allocated injection and SAME well VoW coefficients.

    No operating rate changes to VoW. Scored coverage is reported explicitly.
    """
    if level not in ('CPS','Manifold','Well'):
        raise ValueError('level needs CPS/Manifold/Well')
    w=wells.copy()
    if 'score_valid' not in w:
        w['score_valid']=w['VoW raw'].notna()
    # A strict score_valid flag is used in BOTH scenarios. Unknown scores are not zero.
    w['score_valid']=w['score_valid'].fillna(False).astype(bool)
    w['valid_vo']=pd.to_numeric(w['VoW raw'],errors='coerce').where(w.score_valid)
    keys=['CPS'] if level=='CPS' else ['CPS','Manifold'] if level=='Manifold' else ['CPS','Well']
    if level=='CPS':
        w['CPS']=w.facility
        original=w.copy();opt=w.copy()
    elif level=='Manifold':
        original=w.assign(CPS=w.facility,Manifold=w.manifold)
        opt=w.assign(CPS=w.facility,Manifold=w['Recommended manifold'])
    else:
        original=w.assign(CPS=w.facility)
        opt=original.copy()
    def extract(src,rate,scenario):
        d=src[keys].copy()
        d['Injection (mbd)']=pd.to_numeric(src[rate],errors='coerce')/1000
        d['Scored injection (mbd)']=d['Injection (mbd)'].where(src.score_valid,0.)
        d['Oil supported proxy (mbd)']=(d['Injection (mbd)']*src.valid_vo).fillna(0.)
        d['Wells']=src['Well'].values
        g=(d.groupby(keys,dropna=False)
           .agg({'Injection (mbd)':'sum','Scored injection (mbd)':'sum',
                 'Oil supported proxy (mbd)':'sum','Wells':'nunique'})
           .reset_index())
        g['VoW raw (scored only)']=np.where(g['Scored injection (mbd)']>1e-9,
                                            g['Oil supported proxy (mbd)']/g['Scored injection (mbd)'],np.nan)
        g['Scored coverage (%)']=np.where(g['Injection (mbd)']>1e-9,
                                          g['Scored injection (mbd)']/g['Injection (mbd)']*100,np.nan)
        g['Scenario']=scenario
        return g
    return pd.concat([extract(original,'baseline','Original'),
                      extract(opt,'Optimized injection','Optimized')],ignore_index=True)


def pattern_allocation_rollup(wells:pd.DataFrame,pattern:pd.DataFrame,
                              mapping:Optional[pd.DataFrame])->tuple[pd.DataFrame,list[str]]:
    """AFTER Pattern allocation ONLY when explicit Well-to-Pattern mapping exists.

    Pattern original/optimized BOTH reconstruct well allocation from the same
    well snapshot. They are not the original source historical pattern averages.
    """
    issues=[]
    if mapping is None or mapping.empty:return pd.DataFrame(),['Upload Well → Pattern mapping to enable defensible Pattern Original/Optimized comparisons.']
    lookup={re.sub(r'[^a-z0-9]','',str(c).lower()):c for c in mapping.columns}
    if not {'well','pattern'}.issubset(lookup):raise ValueError('Pattern mapping must have Well and Pattern columns.')
    cols=[lookup['well'],lookup['pattern']]
    m=mapping.copy().dropna(subset=cols)
    m['Well']=m[lookup['well']].astype(str).str.strip().str.upper()
    m['Pattern']=m[lookup['pattern']].astype(str).str.strip().str.upper()
    weightcol=lookup.get('weight')
    m['Weight']=pd.to_numeric(m[weightcol],errors='coerce') if weightcol else 1.
    if not weightcol and m.Well.duplicated().any():
        raise ValueError('More than one mapped Pattern per well: Weight column required.')
    if m['Weight'].isna().any() or m['Weight'].lt(0).any() or (m.groupby('Well').Weight.sum().sub(1.).abs()>0.001).any():
        raise ValueError('All pattern mapping weights must be nonnegative and sum to 1 per well.')
    pat=pattern.copy()
    if 'pat_key' not in pat:pat['pat_key']=pat.Pattern.astype(str).str.strip().str.upper()
    # Reservoir-aware mapping is essential if the same pattern ID exists across
    # multiple reservoirs. Do not mix unrelated reservoir waterflood responses.
    with_res=('Reservoir' in pat.columns and 'Reservoir' in wells.columns)
    if with_res:
        pat['_res']=pat['Reservoir'].map(lambda z:re.sub(r'[^A-Z0-9]','',str(z).upper()))
        gp=(pat.groupby(['pat_key','CPS','_res'],dropna=False)[['Injection','Supported']].sum().reset_index())
    else:
        gp=(pat.groupby(['pat_key','CPS'],dropna=False)[['Injection','Supported']].sum().reset_index())
    gp=gp[gp.Injection>0].copy()
    gp['Source VoW']=gp.Supported/gp.Injection
    m=m.merge(wells[['Well','facility','Reservoir','baseline','Optimized injection']].assign(
        Well=lambda x:x.Well.astype(str).str.upper()).rename(columns={'Reservoir':'_well_reservoir'}),
        on='Well',how='inner')
    if m.empty:return pd.DataFrame(),['None of the uploaded Well-to-Pattern mapping IDs match the selected well snapshot.']
    if with_res:
        map_res_column=lookup.get('reservoir')
        m['_res']=(m[map_res_column] if map_res_column else m['_well_reservoir']).map(
            lambda z:re.sub(r'[^A-Z0-9]','',str(z).upper()))
        m=m.merge(gp,left_on=['Pattern','facility','_res'],right_on=['pat_key','CPS','_res'],how='left')
    else:
        m=m.merge(gp,left_on=['Pattern','facility'],right_on=['pat_key','CPS'],how='left')
    if m['Source VoW'].isna().any():
        bad=m.loc[m['Source VoW'].isna(),['Well','Pattern']].drop_duplicates()
        issues += [f'No source pattern VoW / inconsistent CPS for {r.Well} → {r.Pattern}; excluded from Pattern comparison.' for r in bad.itertuples(index=False)]
    m=m[m['Source VoW'].notna()].copy()
    if m.empty:return pd.DataFrame(),issues
    frames=[]
    for scenario,col in [('Original','baseline'),('Optimized','Optimized injection')]:
        p=m[['Pattern','facility','Source VoW']].copy()
        p['Injection (mbd)']=m[col].astype(float)*m.Weight.astype(float)/1000
        p['Oil supported proxy (mbd)']=p['Injection (mbd)']*p['Source VoW']
        # A row-per-well pattern score is recomputed from total numerator/denominator.
        p=p.groupby(['Pattern','facility'],as_index=False).agg({'Injection (mbd)':'sum','Oil supported proxy (mbd)':'sum'})
        p['VoW raw (scored only)']=np.where(p['Injection (mbd)']>0,p['Oil supported proxy (mbd)']/p['Injection (mbd)'],np.nan)
        p=p.rename(columns={'facility':'CPS'})
        p['Scored coverage (%)']=100.
        p['Scenario']=scenario
        frames.append(p)
    return pd.concat(frames,ignore_index=True),issues


def _base_figure(title,xtitle,ytitle, height=480):
    fig=go.Figure()
    fig.update_layout(template='plotly_white',title=dict(text=title,x=.02,font=dict(size=20,color=NAVY)),
         paper_bgcolor='white',plot_bgcolor='white',font=dict(family='Arial',size=12,color=NAVY),
         margin=dict(l=60,r=30,t=65,b=70),height=height,
         legend=dict(orientation='h',x=0,y=-.19),
         xaxis=dict(title=xtitle,showgrid=True,gridcolor=LIGHT,zeroline=True,zerolinecolor='#becddd'),
         yaxis=dict(title=ytitle,showgrid=True,gridcolor=LIGHT,zeroline=True,zerolinecolor='#becddd'))
    return fig


def bar_before_after(frame:pd.DataFrame,category,before,after,title,ytitle='Rate (mbd)',
                    horizontal=False,top=None):
    d=frame.copy()
    if top and len(d)>top:
        d=d.assign(_change=(d[after]-d[before]).abs()).nlargest(top,'_change')
    labels=d[category].astype(str).tolist()
    fig=_base_figure(title,category if not horizontal else ytitle,ytitle if not horizontal else category,
                     height=max(410,min(970,80+31*len(labels))) if horizontal else 480)
    kwargs={'orientation':'h'} if horizontal else {}
    catarg='y' if horizontal else 'x'
    valarg='x' if horizontal else 'y'
    for field, name, color in ((before,'Original',BLUE),(after,'Optimized',GREEN)):
        fig.add_trace(go.Bar(name=name,marker_color=color,
                  customdata=np.stack([d[after]-d[before]],axis=-1),
                  hovertemplate='%{'+catarg+'}<br>'+name+': %{'+valarg+':,.2f} mbd<br>Δ Optimized - Original: %{customdata[0]:+,.2f} mbd<extra></extra>',
                  **{catarg:labels,valarg:d[field].to_list()},**kwargs))
    fig.update_layout(barmode='group',bargap=.27)
    if horizontal:fig.update_yaxes(autorange='reversed')
    return fig


def injection_vs_support(frame:pd.DataFrame,keys:list[str],title:str,only_changed=False):
    """One connected path per CPS / Manifold / Pattern, with same VoW source basis."""
    if frame.empty:return _base_figure(title,'Injection (mbd)','Oil supported proxy (mbd)')
    d=frame.copy()
    d['Key']=d[keys].fillna('').astype(str).agg(' / '.join,axis=1)
    idx=d.pivot_table(index='Key',columns='Scenario',values='Injection (mbd)',aggfunc='sum')
    both=idx.dropna(subset=['Original','Optimized']) if {'Original','Optimized'}.issubset(idx.columns) else pd.DataFrame()
    if only_changed:
        kept=both.index[(both.Optimized-both.Original).abs()>.05] if not both.empty else []
        d=d[d.Key.isin(kept)]
    else:
        d=d[d.Key.isin(both.index)]
    f=_base_figure(title,'Water injection (mbd)','Oil supported proxy (mbd)',520)
    orig=d[d.Scenario.eq('Original')].set_index('Key'); aft=d[d.Scenario.eq('Optimized')].set_index('Key')
    for name in sorted(set(orig.index)&set(aft.index)):
        u=orig.loc[name];v=aft.loc[name]
        if isinstance(u,pd.DataFrame) or isinstance(v,pd.DataFrame):continue
        if abs(float(u['Injection (mbd)'])-float(v['Injection (mbd)']))>.001:
            f.add_trace(go.Scatter(x=[u['Injection (mbd)'],v['Injection (mbd)']],
                            y=[u['Oil supported proxy (mbd)'],v['Oil supported proxy (mbd)']],
                            mode='lines',line=dict(color='#A6B6C7',width=1.4,dash='dot'),
                            showlegend=False,hoverinfo='skip'))
    # One shared fixed normalization reference for both scenarios, avoiding a
    # misleading different color scale for Original vs Optimized.
    raw=pd.to_numeric(d['VoW raw (scored only)'],errors='coerce')
    valid=raw.replace([np.inf,-np.inf],np.nan).dropna()
    ref=max(float(valid.max()),1e-9) if not valid.empty else 1.
    for scen,sym in [('Original','circle'),('Optimized','diamond')]:
        gg=d[d.Scenario.eq(scen)].copy()
        if gg.empty:continue
        raw_vo=pd.to_numeric(gg['VoW raw (scored only)'],errors='coerce')
        normalized=(raw_vo/ref).clip(0,1)
        custom=np.array(list(zip(gg.Key.astype(str),
                gg['Scored coverage (%)'].to_numpy(float),raw_vo.to_numpy(float),
                normalized.to_numpy(float))),dtype=object)
        f.add_trace(go.Scatter(x=gg['Injection (mbd)'],y=gg['Oil supported proxy (mbd)'],mode='markers+text',
            name=scen,text=gg.Key if len(gg)<=13 else None,textposition='top center',textfont=dict(size=10),
            marker=dict(symbol=sym,color=normalized,size=np.clip(9+gg['Injection (mbd)'].fillna(0)*.07,10,25),
                        cmin=0,cmax=1,colorscale='Viridis',showscale=scen=='Optimized',
                        colorbar=dict(title='VoW (0–1)',len=.73,thickness=15),
                        line=dict(width=2,color=BLUE if scen=='Original' else GREEN)),
            customdata=custom,hovertemplate='<b>%{customdata[0]}</b><br>'+scen+'<br>Water: %{x:,.2f} mbd<br>Oil-supported proxy: %{y:,.2f} mbd<br>Raw VoW: %{customdata[2]:.3f}<br>Normalized VoW: %{customdata[3]:.3f}<br>Scored injection coverage: %{customdata[1]:.0f}%<extra></extra>'))
    return f


def vow_vs_supported(frame:pd.DataFrame,keys:list[str],title:str,only_changed=False):
    d=frame.copy();d['Key']=d[keys].fillna('').astype(str).agg(' / '.join,axis=1)
    idx=d.pivot_table(index='Key',columns='Scenario',values='Injection (mbd)',aggfunc='sum')
    if {'Original','Optimized'}.issubset(idx.columns):
        idx=idx.dropna(subset=['Original','Optimized'])
        if only_changed:idx=idx[(idx.Original-idx.Optimized).abs()>.05]
        d=d[d.Key.isin(idx.index)]
    else:d=d.iloc[0:0]
    f=_base_figure(title,'Injection-weighted raw VoW (scored wells)','Oil supported proxy (mbd)')
    for scen,color in [('Original',BLUE),('Optimized',GREEN)]:
        g=d[d.Scenario.eq(scen)]
        f.add_trace(go.Scatter(x=g['VoW raw (scored only)'],y=g['Oil supported proxy (mbd)'],
            mode='markers+text',name=scen,text=g.Key if len(g)<=12 else None,
            textposition='top center',marker=dict(size=np.clip(10+g['Injection (mbd)'].fillna(0)*.08,10,28),color=color,opacity=.85),
            customdata=np.column_stack([g.Key,g['Injection (mbd)'],g['Scored coverage (%)']]) if len(g) else None,
            hovertemplate='<b>%{customdata[0]}</b><br>'+scen+'<br>VoW: %{x:.3f}<br>Supported proxy: %{y:.2f} mbd<br>Injected water: %{customdata[1]:.2f} mbd<br>Scored coverage: %{customdata[2]:.0f}%<extra></extra>'))
    return f


def source_pattern_reference(pattern:pd.DataFrame):
    """Unaltered historical period volumes: NO fictitious after optimization."""
    ref=(pattern.groupby(['Pattern','CPS'],dropna=False)[['Injection','Supported']].sum().reset_index())
    ref=ref[ref.Injection>0]
    ref['Raw VoW']=ref.Supported/ref.Injection
    fig=_base_figure('Source Pattern record • original period totals (not a prediction)',
                     'Source injection volume (uploaded units)','Source oil supported (uploaded units)')
    fig.add_trace(go.Scatter(x=ref.Injection,y=ref.Supported,mode='markers',
        marker=dict(size=10,color=ref['Raw VoW'],colorscale='Viridis',showscale=True,colorbar=dict(title='Raw VoW')),
        text=ref.Pattern,hovertemplate='%{text}<br>Injected: %{x:,.1f}<br>Supported: %{y:,.1f}<extra></extra>'))
    return fig


def summary_metrics(wells:pd.DataFrame, fac:pd.DataFrame)->dict:
    w=wells.copy()
    active=w.get('score_valid',w['VoW raw'].notna()).fillna(False)
    raw=pd.to_numeric(w['VoW raw'],errors='coerce').where(active,0).fillna(0)
    b=pd.to_numeric(w.baseline,errors='coerce')
    a=pd.to_numeric(w['Optimized injection'],errors='coerce')
    return {
        'injection_before_mbd':_sum(b)/1000,
        'injection_after_mbd':_sum(a)/1000,
        'support_before_mbd':_sum(b*raw)/1000,
        'support_after_mbd':_sum(a*raw)/1000,
        'score_coverage_before':_sum(b.where(active,0))/max(_sum(b),1)*100,
        'score_coverage_after':_sum(a.where(active,0))/max(_sum(a),1)*100,
        'cps_delivery_gap_before_mbd':_sum(fac['Delivery Gap Before'])/1000,
        'cps_delivery_gap_after_mbd':_sum(fac['Delivery Gap After'])/1000,
        'cps_ullage_before_mbd':_sum(fac['CPS Ullage Before'])/1000,
        'cps_ullage_after_mbd':_sum(fac['CPS Ullage After'])/1000,
    }


def prepare_comparisons(result:dict, caps:dict, source_gap:Optional[pd.DataFrame],
                        pattern:pd.DataFrame, mapping:Optional[pd.DataFrame],
                        target='RMT+OE Target')->dict:
    w=result['wells']
    mf,fac=capacity_gap_comparison(w,caps,source_gap,target)
    cp_scatter=allocation_rollup(w,'CPS')
    mf_scatter=allocation_rollup(w,'Manifold')
    pat_scatter,notes=pattern_allocation_rollup(w,pattern,mapping)
    return {'capacity_manifold':mf,'capacity_cps':fac,'scatter_cps':cp_scatter,
            'scatter_manifold':mf_scatter,'scatter_pattern':pat_scatter,
            'pattern_issues':notes,'metrics':summary_metrics(w,fac),
            'selected_target':target}
