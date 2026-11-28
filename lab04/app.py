"""ЛР4. Интерактивный аналитический дашборд в стиле ИТМО (Economics of Cities)."""
import html

import numpy as np
import pandas as pd
import streamlit as st

from charts import (
    ab_diff_chart,
    correlation_heatmap,
    dbscan_heatmap,
    dendrogram_chart,
    feature_distribution_chart,
    k_distance_chart,
    method_distribution_chart,
    scatter_chart,
    silhouette_profile_chart,
    single_conclusion_metric_chart,
    single_histogram,
    single_k_metric_chart,
    variance_bar_chart,
    z_profile_chart,
)
from model import (
    FEATURES,
    LINKAGES,
    MISSING_CODE,
    NOISE,
    SOURCES,
    UNITS,
    ari_summary,
    benchmark_params,
    comparison_table,
    crosstab,
    dendrogram_data,
    describe_group,
    describe_noise_cities,
    disagreeing_cities,
    filter_cities,
    group_color,
    hidden_imputed_reason,
    k_distance,
    load_raw,
    load_results,
    partition,
    profile,
    projection,
    results_mtime,
    sort_groups,
    standardize,
    typical_cities,
    variance_shares,
)

ITMO_LOGO_SVG = (
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 106.627 29.972" '
    'width="74" height="21" aria-label="ИТМО">'
    '<g fill="#000000">'
    '<path fill-rule="nonzero" d="M28.143 8.105 L28.143 13.234 L35.936 13.234 L35.936 29.261 '
    'L41.640 29.261 L41.640 13.234 L49.575 13.234 L49.575 8.105 L28.143 8.105 Z"/>'
    '<path fill-rule="nonzero" d="M20.545 8.060H26.301V29.260H20.545Z"/>'
    '<path fill-rule="nonzero" d="M0.500 8.061 L0.500 29.261 L6.256 29.261 L6.256 8.061 L0.500 8.061 Z"/>'
    '<path fill-rule="nonzero" d="M18.250 8.061 L14.937 8.061 L6.256 29.261 L12.191 29.261 '
    'L20.540 8.061 L18.250 8.061 Z"/>'
    '<path fill-rule="nonzero" d="M78.281 8.061 L76.758 8.061 L75.992 8.061 L72.678 8.061 '
    'L66.965 22.014 L61.252 8.061 L57.939 8.061 L55.649 8.061 L51.416 8.061 L51.416 29.261 '
    'L57.173 29.261 L57.173 11.929 L63.998 29.261 L69.932 29.261 L76.758 11.929 L76.758 29.261 '
    'L82.514 29.261 L82.514 8.061 L78.281 8.061 Z"/>'
    '<path fill-rule="nonzero" d="M23.420 0.500 C21.829 0.500 20.539 1.764 20.539 3.323 '
    'C20.539 4.883 21.829 6.147 23.420 6.147 C25.011 6.147 26.301 4.883 26.301 3.323 '
    'C26.301 1.764 25.011 0.500 23.420 0.500 Z"/>'
    '<path fill-rule="nonzero" d="M106.127 18.681 C106.113 14.499 104.392 11.379 100.998 9.406 '
    'C99.269 8.401 97.320 7.892 95.205 7.892 C94.692 7.892 94.161 7.922 93.625 7.981 '
    'C90.536 8.322 88.051 9.611 86.239 11.814 C84.691 13.696 83.945 15.950 83.964 18.683 '
    'L83.965 18.683 C83.946 21.414 84.692 23.668 86.239 25.549 C88.051 27.752 90.536 29.042 '
    '93.625 29.383 C94.161 29.442 94.693 29.472 95.205 29.472 L95.206 29.472 '
    'C97.320 29.472 99.269 28.962 100.998 27.957 C104.393 25.984 106.114 22.864 106.127 18.681 '
    'M95.085 24.346 L95.075 24.346 C93.431 24.346 92.045 23.836 91.064 22.872 '
    'C90.089 21.913 89.558 20.545 89.528 18.918 C89.527 18.838 89.528 18.760 89.528 18.681 '
    'C89.527 18.603 89.526 18.525 89.528 18.445 C89.558 16.818 90.089 15.451 91.064 14.491 '
    'C92.044 13.527 93.431 13.018 95.075 13.018 L95.084 13.018 C96.682 13.020 98.045 13.550 '
    '99.027 14.551 C100.019 15.562 100.555 17.026 100.549 18.683 C100.555 20.339 100.019 21.802 '
    '99.027 22.813 C98.045 23.814 96.682 24.344 95.085 24.346 Z"/>'
    '</g></svg>'
)

ITMO_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Golos+Text:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&family=Unbounded:wght@500;600;700&display=swap');

:root {
  --itmo-bg: #FAF9F8;
  --itmo-card: #FFFFFF;
  --itmo-ink: #000000;
  --itmo-ink2: #212023;
  --itmo-muted: #6E6F7A;
  --itmo-line: #E6E4EC;
  --itmo-purple: #950CFF;
  --itmo-purple-l1: #F2E6FF;
  --itmo-lime: #C3F168;
  --itmo-orange: #FF9A00;
  --itmo-dark: #201F22;
}

html, body, [class*="css"], .stApp {
  font-family: 'Golos Text', -apple-system, BlinkMacSystemFont, sans-serif;
  background-color: var(--itmo-bg);
  color: var(--itmo-ink2);
}

header[data-testid="stHeader"] {
  background: transparent !important;
  height: 2.0rem !important;
}

.block-container {
  padding-top: 2.1rem !important;
  padding-bottom: 2.2rem !important;
  max-width: 1480px;
}

h1, h2, h3 {
  font-family: 'Unbounded', 'Golos Text', sans-serif !important;
  letter-spacing: -0.02em;
  color: var(--itmo-ink);
}

.itmo-topbar {
  background: var(--itmo-card);
  border: 1px solid var(--itmo-line);
  border-radius: 14px;
  padding: 12px 18px;
  margin-bottom: 12px;
  box-shadow: 0 4px 18px rgba(27, 31, 59, 0.045);
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  flex-wrap: wrap;
}
.itmo-topbar-left {
  display: flex;
  align-items: center;
  gap: 14px;
  flex-wrap: wrap;
}
.itmo-divider {
  width: 1px;
  height: 26px;
  background: var(--itmo-line);
}
.itmo-title-wrap {
  display: flex;
  flex-direction: column;
  gap: 2px;
}
.itmo-title {
  font-family: 'Unbounded', sans-serif;
  font-size: 15.5px;
  font-weight: 600;
  color: var(--itmo-ink);
  line-height: 1.15;
}
.itmo-subtitle {
  font-size: 12px;
  color: var(--itmo-muted);
}
.itmo-topbar-right {
  display: flex;
  align-items: center;
  gap: 8px;
  flex-wrap: wrap;
}
.itmo-pill {
  display: inline-flex;
  align-items: center;
  gap: 5px;
  padding: 5px 10px;
  border-radius: 999px;
  font-size: 11.5px;
  font-weight: 600;
  line-height: 1.2;
}
.itmo-pill-lime { background: var(--itmo-lime); color: var(--itmo-ink); }
.itmo-pill-orange { background: var(--itmo-orange); color: var(--itmo-ink); }
.itmo-pill-neutral { background: var(--itmo-bg); color: var(--itmo-ink2); border: 1px solid var(--itmo-line); }
.itmo-profile {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  background: var(--itmo-bg);
  border: 1px solid var(--itmo-line);
  border-radius: 999px;
  padding: 4px 12px 4px 5px;
}
.itmo-avatar {
  width: 26px;
  height: 26px;
  border-radius: 50%;
  background: var(--itmo-dark);
  color: #FFFFFF;
  font-family: 'Unbounded', sans-serif;
  font-size: 9.5px;
  font-weight: 600;
  display: inline-flex;
  align-items: center;
  justify-content: center;
}
.itmo-profile-meta {
  display: flex;
  flex-direction: column;
  line-height: 1.15;
}
.itmo-profile-name {
  font-size: 12px;
  font-weight: 600;
  color: var(--itmo-ink);
}
.itmo-profile-isu {
  font-size: 10.5px;
  color: var(--itmo-muted);
  font-family: 'JetBrains Mono', monospace;
}

.method-col-head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 6px;
  margin-bottom: 8px;
  padding-bottom: 8px;
  border-bottom: 1px solid var(--itmo-line);
  flex-wrap: wrap;
}
.method-col-name {
  font-family: 'Unbounded', sans-serif;
  font-size: 13.5px;
  font-weight: 600;
  color: var(--itmo-ink);
  white-space: nowrap;
}
.method-col-tag {
  font-size: 10.5px;
  font-family: 'JetBrains Mono', monospace;
  color: var(--itmo-purple);
  background: var(--itmo-purple-l1);
  padding: 2px 8px;
  border-radius: 999px;
  font-weight: 600;
  white-space: nowrap;
}

.method-kpi-grid {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 6px;
  margin: 6px 0 10px 0;
}
.method-kpi-cell {
  background: var(--itmo-bg);
  border: 1px solid var(--itmo-line);
  border-radius: 10px;
  padding: 6px 7px;
  min-width: 0;
}
.method-kpi-label {
  font-size: 10px;
  color: var(--itmo-muted);
  line-height: 1.2;
}
.method-kpi-val {
  font-family: 'Unbounded', sans-serif;
  font-size: 12px;
  font-weight: 600;
  color: var(--itmo-ink);
  margin-top: 2px;
  line-height: 1.25;
}
.method-kpi-delta {
  font-size: 9.5px;
  font-weight: 600;
  color: var(--itmo-purple);
  font-family: 'JetBrains Mono', monospace;
  min-height: 13px;
}
.method-rationale {
  font-size: 12.5px;
  line-height: 1.48;
  color: var(--itmo-ink2);
  background: var(--itmo-bg);
  border-radius: 10px;
  padding: 9px 11px;
  margin: 4px 0 8px 0;
}

.itmo-box {
  border-radius: 14px;
  padding: 14px 16px;
  margin-bottom: 10px;
}
.itmo-box-white {
  background: var(--itmo-card);
  border: 1px solid var(--itmo-line);
  box-shadow: 0 4px 16px rgba(27, 31, 59, 0.04);
}
.itmo-box-equal {
  min-height: 225px;
}
.itmo-box-lime {
  background: var(--itmo-lime);
  color: var(--itmo-ink);
}
.itmo-box-dark {
  background: var(--itmo-dark);
  color: #FFFFFF;
}
.itmo-box-dark .itmo-box-title { color: var(--itmo-lime); }
.itmo-box-lilac {
  background: var(--itmo-purple-l1);
  border: 1px solid #E3C9FF;
  color: var(--itmo-ink2);
}
.itmo-box-orange {
  background: #FFF4E0;
  border: 1px solid #FFD28A;
  color: var(--itmo-ink2);
}
.itmo-box-title {
  font-family: 'Unbounded', sans-serif;
  font-size: 12.5px;
  font-weight: 600;
  margin-bottom: 6px;
  color: var(--itmo-ink);
  line-height: 1.3;
}
.itmo-box-purple-title { color: var(--itmo-purple); }
.itmo-kv {
  display: flex;
  justify-content: space-between;
  align-items: baseline;
  gap: 10px;
  padding: 6px 0;
  border-bottom: 1px solid var(--itmo-line);
  font-size: 12.5px;
}
.itmo-kv:last-child { border-bottom: none; }
.itmo-kv-k { font-weight: 600; color: var(--itmo-ink); }
.itmo-kv-v { color: var(--itmo-purple); font-weight: 600; font-family: 'JetBrains Mono', monospace; text-align: right; }

.itmo-legend-row {
  display: flex;
  flex-wrap: wrap;
  gap: 7px;
  margin: 4px 0 8px 0;
  font-size: 12px;
  color: var(--itmo-ink2);
}
.itmo-legend-item {
  display: inline-flex;
  align-items: center;
  gap: 5px;
  background: var(--itmo-card);
  border: 1px solid var(--itmo-line);
  padding: 3px 10px;
  border-radius: 999px;
}
.itmo-dot {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  display: inline-block;
}
</style>
"""

NAV_VIEWS = [
    'Сравнение 3 методов',
    'Данные',
    'Интерпретация',
    'Выводы',
]

AXIS_OPTIONS = {
    'PC1 × PC2': ('PC1', 'PC2'),
    'Price × Salary': ('Price', 'Salary'),
    'Work × Salary': ('Work', 'Salary'),
    'Work × Price': ('Work', 'Price'),
}

PERSISTED_KEYS = (
    'ward_k', 'ward_linkage', 'kmeans_k', 'dbscan_eps', 'dbscan_ms',
    'board_axes', 'col_chart_view', 'board_w', 'board_p', 'board_s', 'board_imputed',
    'interp_method', 'interp_groups', 'interp_feature', 'data_color_method',
    'cmp_a', 'cmp_b', 'grp_a', 'grp_b',
)


@st.cache_data(show_spinner=False)
def get_raw(mtime):
    return load_raw()


@st.cache_data(show_spinner=False)
def get_benchmark(mtime):
    return load_results()


@st.cache_data(show_spinner=False)
def get_pca_info(mtime):
    X = load_raw()
    Z, _ = standardize(X)
    pc_df, var_ratio = projection(Z)
    domains = {
        'PC1': [float(pc_df['PC1'].min()), float(pc_df['PC1'].max())],
        'PC2': [float(pc_df['PC2'].min()), float(pc_df['PC2'].max())],
    }
    for f in FEATURES:
        domains[f] = [float(X[f].min()), float(X[f].max())]
    return var_ratio, domains


@st.cache_data(show_spinner=False)
def get_partition(mtime, method, params_items):
    X = load_raw()
    bench = load_results()
    return partition(X, method, dict(params_items), benchmark=bench)


@st.cache_data(show_spinner=False)
def get_dendro(mtime, linkage_name, k, city_labels_items):
    X = load_raw()
    Z, _ = standardize(X)
    labels_series = pd.Series(dict(city_labels_items))
    return dendrogram_data(Z, linkage_name=linkage_name, k=k, labels_by_city=labels_series)


@st.cache_data(show_spinner=False)
def get_k_dist(mtime, min_samples):
    X = load_raw()
    Z, _ = standardize(X)
    return k_distance(Z, min_samples=min_samples)


def reset_to_benchmark(bparams):
    st.session_state['ward_k'] = bparams['Ward']['k']
    st.session_state['ward_linkage'] = bparams['Ward']['linkage']
    st.session_state['kmeans_k'] = bparams['k-means']['k']
    st.session_state['dbscan_eps'] = bparams['DBSCAN']['eps']
    st.session_state['dbscan_ms'] = bparams['DBSCAN']['min_samples']


def current_results(mtime, bench_results, is_sandbox):
    if not is_sandbox:
        return bench_results
    return {
        'Ward': get_partition(mtime, 'Ward', (('k', st.session_state['ward_k']), ('linkage', st.session_state['ward_linkage']))),
        'k-means': get_partition(mtime, 'k-means', (('k', st.session_state['kmeans_k']),)),
        'DBSCAN': get_partition(mtime, 'DBSCAN', (('eps', round(st.session_state['dbscan_eps'], 2)), ('min_samples', st.session_state['dbscan_ms']))),
    }


def axis_titles(x_col, y_col, var_ratio):
    def _t(col):
        if col == 'PC1':
            return f'PC1 ({var_ratio[0]:.1%})'
        if col == 'PC2':
            return f'PC2 ({var_ratio[1]:.1%})'
        return f'{col} ({UNITS[col]})'
    return _t(x_col), _t(y_col)


def render_legend_html(groups):
    items = []
    for g in sort_groups(groups):
        c = group_color(g)
        marker = (
            f'<span style="color:{c};font-weight:700;">✕</span>'
            if g == NOISE
            else f'<span class="itmo-dot" style="background:{c};"></span>'
        )
        items.append(f'<span class="itmo-legend-item">{marker} {html.escape(g)}</span>')
    return f'<div class="itmo-legend-row">{"".join(items)}</div>'


def format_ari_badge(results):
    min_val, n_def, n_tot, missing = ari_summary(results)
    if min_val is None:
        return 'ARI не опред.'
    if n_def == n_tot:
        return f'Мин. ARI = {min_val:.3f}'
    missing_methods = sorted({m for pair in missing for m in pair if m == 'DBSCAN'})
    suffix = 'с DBSCAN не опред.' if missing_methods else f'{n_def} из {n_tot} пар'
    return f'ARI = {min_val:.3f} ({suffix})'


def render_topbar(slot, mode, live_results, bench_results, n_cities):
    is_sandbox = mode == 'Песочница'
    best_k = bench_results['k-means']['summary']['clusters']
    mode_pill = (
        '<span class="itmo-pill itmo-pill-orange">Песочница · пересчёт</span>'
        if is_sandbox
        else '<span class="itmo-pill itmo-pill-lime">Эталон ЛР1–3</span>'
    )
    ari_txt = format_ari_badge(live_results)
    slot.markdown(
        f"""
        <div class="itmo-topbar">
          <div class="itmo-topbar-left">
            {ITMO_LOGO_SVG}
            <div class="itmo-divider"></div>
            <div class="itmo-title-wrap">
              <div class="itmo-title">Economics of Cities</div>
              <div class="itmo-subtitle">ЛР4 · {n_cities} городов, 1991 г. · Ответ: {best_k} типа городов (k-means)</div>
            </div>
          </div>
          <div class="itmo-topbar-right">
            <span class="itmo-pill itmo-pill-neutral">{html.escape(ari_txt)}</span>
            {mode_pill}
            <div class="itmo-profile">
              <div class="itmo-avatar">МС</div>
              <div class="itmo-profile-meta">
                <span class="itmo-profile-name">Макаров Семён</span>
                <span class="itmo-profile-isu">ИСУ 335173</span>
              </div>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_city_inspector(X, results, highlight_cities):
    if not highlight_cities:
        return
    rows = []
    missing_notes = []
    for city in highlight_cities:
        if city not in X.index:
            continue
        row = {
            'Город': city,
            'Work': X.loc[city, 'Work'],
            'Price': X.loc[city, 'Price'],
            'Salary': X.loc[city, 'Salary'],
        }
        na_feats = [f for f in FEATURES if pd.isna(X.loc[city, f])]
        if na_feats:
            missing_notes.append(f"{city} (нет наблюдённого {', '.join(na_feats)})")
        for method in SOURCES:
            row[method] = results[method]['cities'].loc[city, 'group']
        rows.append(row)
    if rows:
        with st.container(border=True):
            st.markdown('**Инспектор выбранных городов (сравнение по трём методам)**')
            st.dataframe(
                pd.DataFrame(rows).set_index('Город'),
                width='stretch',
                column_config={
                    'Work': st.column_config.NumberColumn('Work (ч)', format='%.0f'),
                    'Price': st.column_config.NumberColumn('Price', format='%.1f'),
                    'Salary': st.column_config.NumberColumn('Salary', format='%.1f'),
                },
            )
            if missing_notes:
                st.caption(
                    'Пропуски в исходных данных: ' + ', '.join(missing_notes)
                    + ' — на диаграммах с этими осями точка не отображается.'
                )


def render_method_kpi_strip(method, res, bench_res, is_sandbox):
    s, bs = res['summary'], bench_res['summary']
    m, bm = s.get('metrics'), bs.get('metrics')

    cl_val = f"{s['clusters']}"
    cl_delta = f"{s['clusters'] - bs['clusters']:+d}" if (is_sandbox and s['clusters'] != bs['clusters']) else ''

    if m is None:
        sil_val, sil_lbl, sil_delta = '—', 'Silh. ↑', 'нужно ≥ 2'
    else:
        sil_val = f"{m['silhouette']:.3f}"
        sil_lbl = f"Silh. ({s['metrics_n']})" if method == 'DBSCAN' else 'Silh. ↑'
        sil_delta = ''
        if is_sandbox and bm:
            diff = m['silhouette'] - bm['silhouette']
            if abs(diff) > 1e-6:
                sil_delta = f'{diff:+.3f}'.replace('-', '−')
        elif method == 'DBSCAN' and s.get('kmeans_same_subset_metrics'):
            d = m['silhouette'] - s['kmeans_same_subset_metrics']['silhouette']
            sil_delta = f'{d:+.3f} vs km'.replace('-', '−')

    if method == 'DBSCAN':
        third_lbl = 'Шум'
        third_val = f"{s['noise']} ({s['noise'] / s['n']:.0%})"
        third_delta = f"{s['noise'] - bs['noise']:+d}" if (is_sandbox and s['noise'] != bs['noise']) else ''
    else:
        sizes = res['cities'][res['cities'].cluster != -1].cluster.value_counts().sort_index()
        third_lbl = 'Размеры'
        third_val = '/'.join(map(str, sizes)) if len(sizes) else '—'
        third_delta = ''

    st.markdown(
        f"""
        <div class="method-kpi-grid">
          <div class="method-kpi-cell">
            <div class="method-kpi-label">Кл.</div>
            <div class="method-kpi-val">{html.escape(cl_val)}</div>
            <div class="method-kpi-delta">{html.escape(cl_delta)}</div>
          </div>
          <div class="method-kpi-cell">
            <div class="method-kpi-label">{html.escape(sil_lbl)}</div>
            <div class="method-kpi-val">{html.escape(sil_val)}</div>
            <div class="method-kpi-delta">{html.escape(sil_delta)}</div>
          </div>
          <div class="method-kpi-cell">
            <div class="method-kpi-label">{html.escape(third_lbl)}</div>
            <div class="method-kpi-val">{html.escape(third_val)}</div>
            <div class="method-kpi-delta">{html.escape(third_delta)}</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_shared_range_filter(X):
    with st.expander('Фильтр отображения городов по признакам (действует на все 3 столбца)', expanded=False):
        sc1, sc2, sc3, sc4 = st.columns([1.2, 1.2, 1.2, 1.0])
        w_min, w_max = int(X.Work.min()), int(X.Work.max())
        p_min, p_max = float(X.Price.min()), float(X.Price.max())
        s_min, s_max = float(X.Salary.min()), float(X.Salary.max())
        st.session_state.setdefault('board_w', (w_min, w_max))
        st.session_state.setdefault('board_p', (p_min, p_max))
        st.session_state.setdefault('board_s', (s_min, s_max))
        st.session_state.setdefault('board_imputed', True)
        ranges = {
            'Work': sc1.slider('Work (часы)', w_min, w_max, step=1, format='%d', key='board_w'),
            'Price': sc2.slider('Price (цены)', p_min, p_max, step=0.5, format='%.1f', key='board_p'),
            'Salary': sc3.slider('Salary (зарплата)', s_min, s_max, step=0.5, format='%.1f', key='board_s'),
        }
        missing_names = ', '.join(X.index[X.isna().any(axis=1)].tolist())
        include_imputed = sc4.checkbox(f'Показывать {missing_names}', key='board_imputed')
        return ranges, include_imputed


def _method_tag(method, params):
    if method == 'Ward':
        return f"{params['linkage']} · k={params['k']}"
    if method == 'k-means':
        return f"k={params['k']} · k-means++"
    return f"eps={params['eps']:.2f} · M={params['min_samples']}"


def render_method_column(
    method, res, bench_res, is_sandbox, ranges, include_imputed,
    axes_choice, x_col, y_col, xt, yt, x_dom, y_dom,
    highlight_cities, col_chart_view, mtime,
):
    s = res['summary']
    cities = res['cities']
    with st.container(border=True):
        st.markdown(
            f"""
            <div class="method-col-head">
              <span class="method-col-name">{html.escape(method)}</span>
              <span class="method-col-tag">{html.escape(_method_tag(method, s['params']))}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if is_sandbox:
            if method == 'Ward':
                p1, p2 = st.columns([1.15, 1.0])
                p1.slider('Кластеров (k)', 2, 8, key='ward_k')
                p2.selectbox('Связь', LINKAGES, key='ward_linkage')
            elif method == 'k-means':
                st.slider(
                    'Кластеров (k)', 2, 8, key='kmeans_k',
                    help='k-means++, n_init = 50, random_state = 42',
                )
            else:
                d1, d2 = st.columns([1.2, 1.0])
                d1.slider('Радиус eps', 0.30, 1.30, step=0.05, format='%.2f', key='dbscan_eps')
                d2.slider('Соседи (M)', 3, 9, key='dbscan_ms')

        render_method_kpi_strip(method, res, bench_res, is_sandbox)

        shown = filter_cities(cities, cities['group'].unique(), ranges, include_imputed)
        if shown.empty:
            st.info('Нет городов в выбранном диапазоне.')
        else:
            st.altair_chart(
                scatter_chart(
                    shown, x_col, y_col, xt, yt,
                    highlight_cities=highlight_cities,
                    title=axes_choice,
                    height=275, x_domain=x_dom, y_domain=y_dom, show_legend=False,
                ),
                width='stretch',
                key=f'col_{method}_scatter',
            )

        if col_chart_view == 'Силуэт':
            if s.get('metrics') is None:
                st.info(f"Кластеров: {s['clusters']}. Для силуэта нужно ≥ 2 кластеров.")
            else:
                sil_chart = silhouette_profile_chart(shown, s['metrics']['silhouette'], height=225)
                if sil_chart is None:
                    st.info('В фильтре нет не-шумовых городов для силуэта.')
                else:
                    st.altair_chart(sil_chart, width='stretch', key=f'col_{method}_sil')
        else:
            if method == 'Ward':
                city_labels = tuple(cities['cluster'].items())
                dendro = get_dendro(mtime, s['params']['linkage'], s['clusters'], city_labels)
                st.altair_chart(
                    dendrogram_chart(dendro, highlight_cities=highlight_cities, title='Дендрограмма', height=225),
                    width='stretch',
                    key='col_ward_dendro',
                )
            elif method == 'k-means':
                st.altair_chart(
                    single_k_metric_chart(res['selection'], 'inertia', 'Осыпь W(k) ↓', s['clusters'], height=225),
                    width='stretch',
                    key='col_km_elbow',
                )
            else:
                st.altair_chart(
                    k_distance_chart(get_k_dist(mtime, s['params']['min_samples']), s['params']['eps'], s['params']['min_samples'], height=225),
                    width='stretch',
                    key='col_db_kdist',
                )

        st.markdown(f'<div class="method-rationale">{html.escape(s["rationale"])}</div>', unsafe_allow_html=True)

        with st.expander(f'Детали, подбор и таблица ({method})', expanded=False):
            if method == 'Ward':
                heights_df = pd.DataFrame({'k': range(1, len(s.get('merge_heights', [])) + 1), 'height': s.get('merge_heights', [])})
                sel = res['selection']
                link = s['params']['linkage']
                w_sel = sel[sel['method'] == link] if 'method' in sel.columns and (sel['method'] == link).any() else sel
                st.altair_chart(single_k_metric_chart(heights_df, 'height', 'Высота слияния ↓', s['clusters'], height=130), width='stretch', key='col_w_h')
                st.altair_chart(single_k_metric_chart(w_sel.dropna(subset=['silhouette']), 'silhouette', 'Silhouette ↑', s['clusters'], height=130), width='stretch', key='col_w_s')
                if 'method' in sel.columns and sel['method'].nunique() > 1:
                    st.markdown('**Связи (Silhouette из ЛР1)**')
                    st.dataframe(sel.pivot(index='method', columns='k', values='silhouette').round(3), width='stretch')
            elif method == 'k-means':
                sel = res['selection']
                st.altair_chart(single_k_metric_chart(sel.dropna(subset=['silhouette']), 'silhouette', 'Silhouette ↑', s['clusters'], height=130), width='stretch', key='col_km_s')
                st.altair_chart(single_k_metric_chart(sel.dropna(subset=['CH']), 'CH', 'Calinski–Harabasz ↑', s['clusters'], height=130), width='stretch', key='col_km_ch')
                st.altair_chart(single_k_metric_chart(sel.dropna(subset=['DB']), 'DB', 'Davies–Bouldin ↓', s['clusters'], height=130), width='stretch', key='col_km_db')
            else:
                noise_df = describe_noise_cities(cities)
                if noise_df.empty:
                    st.info('При текущих параметрах шумовых городов нет.')
                else:
                    st.markdown(f"**Шумовые города ({len(noise_df)})**")
                    st.dataframe(noise_df.set_index('City'), width='stretch', height=180)
            st.dataframe(
                shown[['group', *FEATURES, 'silhouette']].sort_values(['group', 'Salary'], ascending=[True, False]),
                width='stretch', height=180,
            )


def view_three_columns(X, live_results, bench_results, is_sandbox, var_ratio, domains, highlight_cities, mtime):
    bar_l, bar_r = st.columns([1.3, 1.0], vertical_alignment='center')
    with bar_l:
        axes_choice = st.segmented_control(
            'Оси всех трёх графиков',
            list(AXIS_OPTIONS),
            key='board_axes',
            required=True,
            label_visibility='collapsed',
        )
    with bar_r:
        col_chart_view = st.segmented_control(
            'Нижний график столбцов',
            ['Подбор параметров', 'Силуэт'],
            key='col_chart_view',
            required=True,
            label_visibility='collapsed',
        )

    ranges, include_imputed = render_shared_range_filter(X)
    x_col, y_col = AXIS_OPTIONS[axes_choice]
    xt, yt = axis_titles(x_col, y_col, var_ratio)
    x_dom, y_dom = domains[x_col], domains[y_col]

    all_groups = set()
    for m_name in SOURCES:
        all_groups.update(live_results[m_name]['cities']['group'].unique())
    st.markdown(render_legend_html(all_groups), unsafe_allow_html=True)

    sample_reason = hidden_imputed_reason(
        live_results['k-means']['cities'],
        live_results['k-means']['cities']['group'].unique(),
        ranges,
        include_imputed,
    )
    if sample_reason:
        st.info(sample_reason)

    cols = st.columns(3, gap='medium')
    for col, m_name in zip(cols, SOURCES):
        with col:
            render_method_column(
                m_name, live_results[m_name], bench_results[m_name], is_sandbox,
                ranges, include_imputed, axes_choice, x_col, y_col, xt, yt, x_dom, y_dom,
                highlight_cities, col_chart_view, mtime,
            )

    with st.container(border=True):
        st.markdown('**Сводная таблица качества и сопоставление алгоритмов**')
        table = comparison_table(live_results).rename(columns={
            'Размеры кластеров': 'Размеры',
            'Silhouette': 'Silhouette ↑',
            'Davies-Bouldin': 'DB ↓',
            'Calinski-Harabasz': 'CH ↑',
            'Метрики по объектам': 'n для метрик',
        })
        st.dataframe(
            table.style.format({
                'Доля шума': '{:.0%}',
                'Silhouette ↑': lambda v: '—' if pd.isna(v) else f'{v:.3f}',
                'DB ↓': lambda v: '—' if pd.isna(v) else f'{v:.3f}',
                'CH ↑': lambda v: '—' if pd.isna(v) else f'{v:.1f}',
            }),
            width='stretch',
        )

        sd = live_results['DBSCAN']['summary']
        sk = live_results['k-means']['summary']
        db_m, km_m, sub_m = sd.get('metrics'), sk.get('metrics'), sd.get('kmeans_same_subset_metrics')
        if db_m is None:
            st.info(
                f"У DBSCAN сейчас {sd['clusters']} кластер(а/ов) и {sd['noise']} шумовых городов — "
                'внутренние метрики (Silhouette, DB, CH) не определены.'
            )
        elif sub_m and km_m:
            diff_sub = abs(db_m['silhouette'] - sub_m['silhouette'])
            verdict = (
                'совпадают — разница объясняется исключением шумовых городов.'
                if diff_sub < 1e-3
                else f"различаются на {db_m['silhouette'] - sub_m['silhouette']:+.3f}."
            )
            st.markdown(
                f"""
                <div class="itmo-box itmo-box-orange" style="margin-top:8px;margin-bottom:8px;">
                  <div class="itmo-box-title">Корректное сопоставление DBSCAN</div>
                  <div style="font-size:13px;line-height:1.45;">
                    Метрики DBSCAN посчитаны по <b>{sd['metrics_n']}</b> не-шумовым городам из {sd['n']}
                    (Silhouette = <b>{db_m['silhouette']:.3f}</b>). На тех же {sd['metrics_n']} городах разбиение k-means (k={sd['clusters']})
                    даёт <b>{sub_m['silhouette']:.3f}</b> — значения {verdict}
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        bot_l, bot_r = st.columns([1.0, 1.2])
        with bot_l:
            st.altair_chart(method_distribution_chart(live_results), width='stretch', key='cmp_dist')
            with st.expander('Тепловая карта сетки DBSCAN (eps × M)', expanded=False):
                st.altair_chart(
                    dbscan_heatmap(bench_results['DBSCAN']['selection'], sd['params']['eps'], sd['params']['min_samples']),
                    width='stretch',
                    key='cmp_db_heat',
                )
        with bot_r:
            st.markdown('**Матрица совпадений (A × B) и расхождения**')
            sa, sb = st.columns(2)
            methods_list = list(SOURCES)
            first = sa.selectbox('Алгоритм A', methods_list, key='cmp_a')
            second = sb.selectbox('Алгоритм B', methods_list, key='cmp_b')
            ct, ari, n_common = crosstab(live_results, first, second)
            st.dataframe(ct, width='stretch')
            if ari is None:
                st.caption(f'Общих не-шумовых городов: {n_common} — ARI не определён.')
            else:
                st.caption(f'ARI = **{ari:.3f}** по **{n_common}** не-шумовым городам (не зависит от нумерации кластеров).')
            dis = disagreeing_cities(live_results[first]['cities'], live_results[second]['cities'], first, second)
            if dis.empty:
                st.success(f'Разбиения {first} и {second} полностью совпадают на всех {len(X)} городах.')
            else:
                with st.expander(f'Расходящиеся города между {first} и {second} ({len(dis)})', expanded=False):
                    st.dataframe(
                        dis,
                        width='stretch',
                        column_config={
                            'Work': st.column_config.NumberColumn('Work', format='%.0f'),
                            'Price': st.column_config.NumberColumn('Price', format='%.1f'),
                            'Salary': st.column_config.NumberColumn('Salary', format='%.1f'),
                        },
                    )


def view_data(X, live_results):
    missing_cities = X.index[X.isna().any(axis=1)].tolist()
    missing_cols = [f for f in FEATURES if X[f].isna().any()]
    n_missing = int(X.isna().sum().sum())
    var_df = variance_shares(X)
    work_before = var_df[(var_df['Признак'] == 'Work') & (var_df['Этап'] == 'До стандартизации')]['Доля дисперсии'].iloc[0]

    left, right = st.columns([1.0, 1.35])
    with left:
        st.markdown(
            f"""
            <div class="itmo-box itmo-box-white">
              <div class="itmo-box-title itmo-box-purple-title">Паспорт набора данных</div>
              <div class="itmo-kv"><span class="itmo-kv-k">Наблюдений</span><span class="itmo-kv-v">{len(X)} городов</span></div>
              <div class="itmo-kv"><span class="itmo-kv-k">Признаков</span><span class="itmo-kv-v">{len(FEATURES)} ({', '.join(FEATURES)})</span></div>
              <div class="itmo-kv"><span class="itmo-kv-k">Пропуски ({MISSING_CODE})</span><span class="itmo-kv-v">{n_missing} ({html.escape(', '.join(missing_cities))})</span></div>
              <div class="itmo-kv"><span class="itmo-kv-k">Источник</span><span class="itmo-kv-v">UBS · 1991 г.</span></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with right:
        with st.container(border=True):
            st.altair_chart(variance_bar_chart(var_df), width='stretch')

    with st.container(border=True):
        h_l, h_r = st.columns([1.3, 1.0], vertical_alignment='center')
        h_l.markdown('**Распределения признаков по группам выбранного алгоритма**')
        data_method = h_r.segmented_control(
            'Окраска по методу',
            list(SOURCES),
            key='data_color_method',
            required=True,
            label_visibility='collapsed',
        )
        active_cities = live_results[data_method]['cities']
        st.markdown(render_legend_html(active_cities['group'].unique()), unsafe_allow_html=True)
        for idx, (col, feat) in enumerate(zip(st.columns(3), FEATURES)):
            col.altair_chart(single_histogram(X, active_cities, feat, show_y=(idx == 0)), width='stretch')

    c1, c2 = st.columns([1.0, 1.2])
    with c1:
        with st.container(border=True):
            st.altair_chart(correlation_heatmap(X), width='stretch')
    with c2:
        med_parts = ', '.join(f'{f} = <b>{X[f].median():g}</b>' for f in missing_cols)
        st.markdown(
            f"""
            <div class="itmo-box itmo-box-dark">
              <div class="itmo-box-title">Предобработка (единая для ЛР1–3)</div>
              <div style="font-size:13px;line-height:1.55;">
                1. Код <code>{MISSING_CODE}</code> заменён на пропуск у <b>{html.escape(', '.join(missing_cities))}</b> ({html.escape(', '.join(missing_cols))}).<br>
                2. Пропуски заполнены <b>медианой</b>: {med_parts}.<br>
                3. Выполнена <b>z-стандартизация</b>: без неё на Work приходится <b>{work_before:.1%}</b> всей дисперсии.<br>
                4. Кластеризация идёт по 3 стандартизованным признакам (евклидово расстояние); PCA нужен только для 2D-проекций.
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with st.expander(f'Описательная статистика (наблюдённые значения) и таблица {len(X)} городов', expanded=False):
        desc = X.describe().T.rename(columns={
            'count': 'Наблюдений', 'mean': 'Среднее', 'std': 'Ст. откл.',
            'min': 'Мин.', '25%': '25 %', '50%': 'Медиана', '75%': '75 %', 'max': 'Макс.',
        })
        st.dataframe(desc.round(2), width='stretch')
        st.dataframe(
            X.assign(
                Ward=live_results['Ward']['cities']['group'],
                **{'k-means': live_results['k-means']['cities']['group']},
                DBSCAN=live_results['DBSCAN']['cities']['group'],
            ),
            width='stretch',
        )


def view_interpretation(X, live_results, highlight_cities):
    top_l, top_r = st.columns([1.0, 1.5])
    with top_l:
        method = st.segmented_control(
            'Алгоритм для интерпретации',
            list(SOURCES),
            key='interp_method',
            required=True,
        )
    cities = live_results[method]['cities']

    options = sort_groups(cities.group.unique())
    if 'interp_groups' not in st.session_state or st.session_state.get('interp_last_options') != options:
        st.session_state['interp_last_options'] = options
        st.session_state['interp_groups'] = options

    with top_r:
        groups = st.multiselect('Кластеры для анализа', options, key='interp_groups')

    full_ranges = {f: (float(X[f].min()), float(X[f].max())) for f in FEATURES}
    shown = filter_cities(cities, groups, full_ranges, include_imputed=True)
    if shown.empty:
        st.info('Нет городов для отображения: выберите хотя бы одну группу выше.')
        return

    prof = profile(shown)
    examples = typical_cities(shown)

    left, right = st.columns([1.0, 1.15])
    with left:
        with st.container(border=True):
            z_chart = z_profile_chart(prof)
            if z_chart is None:
                st.info('Средние z-оценки строятся только для кластеров: у шума общего профиля нет.')
            else:
                st.altair_chart(z_chart, width='stretch')

    with right:
        cluster_groups = [g for g in prof.index if g != NOISE]
        if not cluster_groups:
            st.info('В текущем выборе нет кластеров (выбран только шум или 0 кластеров).')
        for g in cluster_groups:
            p = prof.loc[g]
            color = group_color(g)
            desc = describe_group(p)
            typ = ', '.join(examples.get(g, [])) or '—'
            st.markdown(
                f"""
                <div class="itmo-box itmo-box-white" style="border-left:4px solid {color};">
                  <div class="itmo-box-title" style="display:flex;justify-content:space-between;">
                    <span>{html.escape(g)}</span>
                    <span style="color:{color};">{int(p['Городов'])} гор.</span>
                  </div>
                  <div style="font-size:13px;line-height:1.45;margin-bottom:6px;">{html.escape(desc)}</div>
                  <div style="font-size:12px;color:#6E6F7A;">
                    Средние: Salary <b>{p['Среднее Salary']:.1f}</b> · Price <b>{p['Среднее Price']:.1f}</b> · Work <b>{p['Среднее Work']:.0f} ч</b><br>
                    Типичные: <b>{html.escape(typ)}</b>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        if NOISE in prof.index:
            noise_df = describe_noise_cities(shown)
            sample_reasons = '; '.join(
                f"{r['City']} ({r['Главное отклонение']})" for _, r in noise_df.head(5).iterrows()
            )
            more = f' и ещё {len(noise_df) - 5}' if len(noise_df) > 5 else ''
            st.markdown(
                f"""
                <div class="itmo-box itmo-box-lilac">
                  <div class="itmo-box-title">Шум ({int(prof.loc[NOISE, 'Городов'])} гор.) — не отдельный кластер</div>
                  <div style="font-size:12.5px;line-height:1.45;">
                    Разнородные крайние города: {html.escape(sample_reasons)}{html.escape(more)}.
                    Общий средний профиль для шума не строится.
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    bot_l, bot_r = st.columns([1.1, 1.0])
    with bot_l:
        with st.container(border=True):
            feat = st.segmented_control(
                'Признак для сравнения распределений',
                FEATURES,
                key='interp_feature',
                required=True,
            )
            f_chart = feature_distribution_chart(shown, feat, highlight_cities=highlight_cities)
            if f_chart is not None:
                st.altair_chart(f_chart, width='stretch')

    with bot_r:
        with st.container(border=True):
            st.markdown('**Сравнение двух групп (A vs B)**')
            grp_list = list(prof.index)
            if len(grp_list) < 2:
                st.info('Для сравнения нужно хотя бы две группы на экране.')
            else:
                ga_col, gb_col = st.columns(2)
                if st.session_state.get('grp_a') not in grp_list:
                    st.session_state['grp_a'] = grp_list[0]
                ga = ga_col.selectbox('Группа A', grp_list, key='grp_a')
                other_groups = [g for g in grp_list if g != ga]
                if st.session_state.get('grp_b') not in other_groups:
                    st.session_state['grp_b'] = other_groups[0]
                gb = gb_col.selectbox('Группа B', other_groups, key='grp_b')

                st.altair_chart(ab_diff_chart(prof, ga, gb), width='stretch')
                means = [f'Среднее {f}' for f in FEATURES]
                diff = pd.DataFrame(
                    {'A': prof.loc[ga, means].values, 'B': prof.loc[gb, means].values},
                    index=FEATURES,
                )
                diff['B − A'] = diff['B'] - diff['A']
                diff['B / A'] = diff['B'] / diff['A'].replace(0, np.nan)
                st.caption(f'A = {ga} · B = {gb}')
                st.dataframe(
                    diff.round(2),
                    width='stretch',
                    column_config={
                        'A': st.column_config.NumberColumn('A', format='%.1f'),
                        'B': st.column_config.NumberColumn('B', format='%.1f'),
                        'B − A': st.column_config.NumberColumn('B − A', format='%.1f'),
                        'B / A': st.column_config.NumberColumn('B / A', format='%.2f'),
                    },
                )

    with st.expander('Полная таблица профилей (наблюдённые средние, медианы и z-оценки)', expanded=False):
        st.caption('Заполненные пропуски исключены из расчёта средних, медиан и z-оценок. Строка «Шум» справочная.')
        st.dataframe(prof.round(2), width='stretch')


def view_conclusions(bench_results, live_results, X, is_sandbox):
    t = comparison_table(bench_results)
    wk_ari = crosstab(bench_results, 'Ward', 'k-means')[1]
    min_ari, _, _, _ = ari_summary(bench_results)

    km_s = bench_results['k-means']['summary']
    db_s = bench_results['DBSCAN']['summary']
    wd_s = bench_results['Ward']['summary']
    km_sel = bench_results['k-means']['selection'].set_index('k')
    best_k = km_s['clusters']
    drop_k = km_sel.loc[best_k, 'падение inertia, %']
    next_max_drop = km_sel.loc[best_k + 1 :, 'падение inertia, %'].max()
    best_sil_k = int(km_sel['silhouette'].idxmax())
    best_ch_k = int(km_sel['CH'].idxmax())
    best_db_k = int(km_sel['DB'].idxmin())
    heights = wd_s['merge_heights']
    missing_names = ', '.join(X.index[X.isna().any(axis=1)].tolist())
    r_ps = X.Price.corr(X.Salary)

    same_wk_text = (
        'Ward и k-means дают идентичное разбиение (ARI = 1.000) и одинаковые метрики. '
        if wk_ari is not None and wk_ari > 0.999
        else f'Согласие Ward и k-means составляет ARI = {wk_ari:.3f}. '
    )

    st.markdown(
        f"""
        <div class="itmo-box itmo-box-lime">
          <div class="itmo-box-title">Главный вывод по эталону ЛР1–3: {best_k} типа городов · базовый метод — k-means</div>
          <div style="font-size:13.5px;line-height:1.55;">
            {same_wk_text}Для итоговой типологии выбран <b>k-means (k = {best_k})</b>: он относит к группе каждый город и задаёт
            понятные центры в исходных единицах. <b>Ward</b> дополняет картину деревом слияний, а <b>DBSCAN</b>
            выявляет <b>{db_s['noise']}</b> нетипичных городов ({db_s['noise'] / db_s['n']:.0%} шума).
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    a1, a2, a3 = st.columns(3)
    with a1:
        st.markdown(
            f"""
            <div class="itmo-box itmo-box-white itmo-box-equal">
              <div class="itmo-box-title itmo-box-purple-title">1. Выбор k</div>
              <div style="font-size:12.5px;line-height:1.55;">
                › Излом осыпи W(k): переход 1 → {best_k} снижает W на <b>{drop_k:.0f} %</b>, дальше ≤ <b>{next_max_drop:.0f} %</b>.<br>
                › Скачок дендрограммы Ward: <b>{heights[0]:.2f}</b> против <b>{heights[1]:.2f}</b>.<br>
                › На {km_s['n']} городах: Silhouette <b>{t.loc['k-means', 'Silhouette']:.3f}</b> (лучший при k={best_sil_k}),
                CH <b>{t.loc['k-means', 'Calinski-Harabasz']:.1f}</b> (k={best_ch_k}), DB <b>{t.loc['k-means', 'Davies-Bouldin']:.3f}</b>
                (ниже только при k={best_db_k}: {km_sel.loc[best_db_k, 'DB']:.3f}, мин. кластер из {km_sel.loc[best_db_k, 'min_size']:.0f} гор.).
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with a2:
        st.markdown(
            f"""
            <div class="itmo-box itmo-box-white itmo-box-equal">
              <div class="itmo-box-title itmo-box-purple-title">2. Устойчивость</div>
              <div style="font-size:12.5px;line-height:1.55;">
                › Три метода выделяют одни и те же группы: мин. ARI <b>{min_ari:.2f}</b>.<br>
                › Silhouette DBSCAN (<b>{db_s['metrics']['silhouette']:.3f}</b>) получен на {db_s['metrics_n']} городах
                и совпадает с k-means на той же подвыборке (<b>{db_s['kmeans_same_subset_metrics']['silhouette']:.3f}</b>).<br>
                › По ЛР1–3: разбиение на {best_k} группы сохраняется на шкале [0, 1] и без {html.escape(missing_names)};
                при k = {best_k + 1} граница между группами у Ward и k-means расходится.
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with a3:
        st.markdown(
            f"""
            <div class="itmo-box itmo-box-white itmo-box-equal">
              <div class="itmo-box-title itmo-box-purple-title">3. Применение</div>
              <div style="font-size:12.5px;line-height:1.55;">
                › k-means и Ward дают полную типологию всех {len(X)} городов с контрастными профилями.<br>
                › DBSCAN оставляет <b>{t.loc['DBSCAN', 'Доля шума']:.0%}</b> городов без группы — полезно для поиска выбросов,
                но неудобно как основная классификация.<br>
                › Ограничения: {len(X)} городов, 1991 г.; у <b>{html.escape(missing_names)}</b> заполнены 2 из 3 признаков;
                Price и Salary сильно коррелируют (<b>r = {r_ps:.2f}</b>).
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with st.container(border=True):
        st.markdown(
            f"**Сводка внутренних метрик эталона ЛР1–3 (у DBSCAN — по {db_s['metrics_n']} из {db_s['n']} городов; акцент — рекомендуемый k-means)**"
        )
        mc1, mc2, mc3 = st.columns(3)
        mc1.altair_chart(single_conclusion_metric_chart(bench_results, 'silhouette', 'Silhouette ↑'), width='stretch')
        mc2.altair_chart(single_conclusion_metric_chart(bench_results, 'CH', 'Calinski–Harabasz ↑'), width='stretch')
        mc3.altair_chart(single_conclusion_metric_chart(bench_results, 'DB', 'Davies–Bouldin ↓'), width='stretch')

    if is_sandbox:
        diff_rows = []
        moved_summary = []
        for m_name in SOURCES:
            bs = bench_results[m_name]['summary']
            ls = live_results[m_name]['summary']
            _, ari_vs_bench, _ = crosstab({'B': bench_results[m_name], 'L': live_results[m_name]}, 'B', 'L')
            dis = disagreeing_cities(bench_results[m_name]['cities'], live_results[m_name]['cities'], 'Эталон', 'Песочница')
            bm, lm = bs.get('metrics'), ls.get('metrics')
            d_sil = (lm['silhouette'] - bm['silhouette']) if (lm and bm) else np.nan
            diff_rows.append({
                'Алгоритм': m_name,
                'Кластеров (эталон → сейчас)': f"{bs['clusters']} → {ls['clusters']}",
                'Шум (эталон → сейчас)': f"{bs['noise']} → {ls['noise']}",
                'ARI с эталоном': ari_vs_bench if ari_vs_bench is not None else np.nan,
                'Δ Silhouette': d_sil,
                'Городов изменили группу': len(dis),
            })
            if not dis.empty:
                moved_summary.append(f"**{m_name}** ({len(dis)}): " + ', '.join(dis.index.tolist()))

        st.markdown(
            """
            <div class="itmo-box itmo-box-orange">
              <div class="itmo-box-title">Что изменилось в песочнице относительно эталона ЛР1–3</div>
              <div style="font-size:13px;">Расхождение текущих параметров с защищаемым эталоном ноутбуков.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.dataframe(
            pd.DataFrame(diff_rows).set_index('Алгоритм').style.format({
                'ARI с эталоном': lambda v: '—' if pd.isna(v) else f'{v:.3f}',
                'Δ Silhouette': lambda v: '—' if pd.isna(v) else f'{v:+.3f}',
            }),
            width='stretch',
        )
        if moved_summary:
            st.caption('Изменили группу или статус шума — ' + ' · '.join(moved_summary))


def main():
    st.set_page_config(
        page_title='ИТМО · ЛР4 · Economics of Cities',
        layout='wide',
        initial_sidebar_state='collapsed',
    )
    st.markdown(ITMO_CSS, unsafe_allow_html=True)

    # Сохраняем состояние виджетов, которые скрываются при переключении разделов
    for key in PERSISTED_KEYS:
        if key in st.session_state:
            st.session_state[key] = st.session_state[key]

    mtime = results_mtime()
    try:
        X = get_raw(mtime)
        bench_results = get_benchmark(mtime)
        var_ratio, domains = get_pca_info(mtime)
    except (ValueError, FileNotFoundError) as exc:
        st.error(f'Ошибка загрузки данных ЛР1–3: {exc}')
        st.stop()

    bparams = benchmark_params(bench_results)
    st.session_state.setdefault('ward_k', bparams['Ward']['k'])
    st.session_state.setdefault('ward_linkage', bparams['Ward']['linkage'])
    st.session_state.setdefault('kmeans_k', bparams['k-means']['k'])
    st.session_state.setdefault('dbscan_eps', bparams['DBSCAN']['eps'])
    st.session_state.setdefault('dbscan_ms', bparams['DBSCAN']['min_samples'])
    st.session_state.setdefault('board_axes', 'PC1 × PC2')
    st.session_state.setdefault('col_chart_view', 'Подбор параметров')
    st.session_state.setdefault('data_color_method', 'k-means')
    st.session_state.setdefault('interp_method', 'k-means')
    st.session_state.setdefault('interp_feature', 'Salary')
    st.session_state.setdefault('cmp_a', 'k-means')
    st.session_state.setdefault('cmp_b', 'DBSCAN')

    topbar_slot = st.empty()

    ctrl_mode, ctrl_cities, ctrl_action, ctrl_help = st.columns([1.4, 1.45, 0.72, 0.45], vertical_alignment='center')
    with ctrl_mode:
        mode = st.segmented_control(
            'Режим анализа',
            ['Эталон ЛР1–3', 'Песочница'],
            default='Эталон ЛР1–3',
            key='mode',
            required=True,
            label_visibility='collapsed',
        )
    is_sandbox = mode == 'Песочница'
    live_results = current_results(mtime, bench_results, is_sandbox)

    render_topbar(topbar_slot, mode, live_results, bench_results, len(X))

    with ctrl_cities:
        highlight_cities = st.multiselect(
            'Подсветить города',
            sorted(X.index.tolist()),
            default=[],
            key='highlight_cities',
            placeholder='Подсветить города на графиках…',
            label_visibility='collapsed',
        )

    with ctrl_action:
        if is_sandbox:
            st.button(
                'Сбросить',
                on_click=reset_to_benchmark,
                args=(bparams,),
                width='stretch',
                key='reset_btn',
            )

    with ctrl_help:
        with st.popover('?', width='stretch'):
            st.markdown(
                '**Как пользоваться дашбордом:**\n'
                '1. **Эталон ЛР1–3** — точные результаты ноутбуков.\n'
                '2. **Песочница** — открывает слайдеры прямо внутри каждого из трёх столбцов (**Ward**, **k-means**, **DBSCAN**).\n'
                '3. **Сравнение 3 методов** — три алгоритма рядом в единых осях и одной палитре.\n'
                '4. **Подсветить города** — выделяет города на графиках и выводит сравнительную таблицу.'
            )

    active_view = st.segmented_control(
        'Раздел',
        NAV_VIEWS,
        default=NAV_VIEWS[0],
        key='nav_view',
        required=True,
        label_visibility='collapsed',
        width='stretch',
    )

    render_city_inspector(X, live_results, highlight_cities)

    if active_view == 'Сравнение 3 методов':
        view_three_columns(X, live_results, bench_results, is_sandbox, var_ratio, domains, highlight_cities, mtime)
    elif active_view == 'Данные':
        view_data(X, live_results)
    elif active_view == 'Интерпретация':
        view_interpretation(X, live_results, highlight_cities)
    else:
        view_conclusions(bench_results, live_results, X, is_sandbox)


if __name__ == '__main__':
    main()
