"""Графики Altair в светлом стиле ИТМО для дашборда ЛР4."""
import altair as alt
import numpy as np
import pandas as pd

from model import (
    FEATURES,
    NOISE,
    SOURCES,
    UNITS,
    group_color,
    sort_groups,
)

ITMO_PURPLE = '#950CFF'
ITMO_PURPLE_L2 = '#E5CCFF'
ITMO_ORANGE = '#FF9A00'
ITMO_LIME = '#C3F168'
ITMO_DARK = '#201F22'
ITMO_SLATE = '#474A60'
ITMO_MUTED = '#6E6F7A'
ITMO_FAINT = '#B2B3C1'
ITMO_LINE = '#E6E4EC'

FONT_BODY = 'Golos Text, -apple-system, BlinkMacSystemFont, sans-serif'
FONT_HEAD = 'Unbounded, Golos Text, -apple-system, sans-serif'


def configure_chart(chart):
    """Единая конфигурация Altair под светлый стиль ИТМО."""
    return (
        chart.configure_view(strokeWidth=0, fill='#FFFFFF')
        .configure_axis(
            labelFont=FONT_BODY,
            titleFont=FONT_BODY,
            labelColor=ITMO_DARK,
            titleColor=ITMO_MUTED,
            labelFontSize=11,
            titleFontSize=11,
            gridColor=ITMO_LINE,
            domainColor=ITMO_LINE,
            tickColor=ITMO_LINE,
        )
        .configure_title(
            font=FONT_HEAD,
            fontSize=12,
            fontWeight=600,
            color=ITMO_DARK,
            anchor='start',
            offset=8,
        )
        .configure_legend(
            labelFont=FONT_BODY,
            titleFont=FONT_BODY,
            labelColor=ITMO_DARK,
            titleColor=ITMO_MUTED,
            labelFontSize=11,
            titleFontSize=11,
            orient='bottom',
        )
    )


def color_scale(groups):
    """Динамическая цветовая шкала по фактическим группам в числовом порядке."""
    domain = sort_groups(groups)
    return alt.Scale(domain=domain, range=[group_color(g) for g in domain])


def variance_bar_chart(var_df):
    """Вклад признаков в дисперсию до и после z-стандартизации (нейтральная шкала)."""
    df = var_df.copy()
    df['Этап'] = df['Этап'].replace({
        'До стандартизации': 'До z',
        'После z-стандартизации': 'После z',
    })
    df['ord'] = df['Признак'].map({f: i for i, f in enumerate(FEATURES)})
    neutral_scale = alt.Scale(domain=FEATURES, range=[ITMO_DARK, ITMO_MUTED, '#D4D3DC'])
    chart = (
        alt.Chart(df, title='Вклад признаков в дисперсию')
        .mark_bar(cornerRadius=4, height=22)
        .encode(
            y=alt.Y(
                'Этап:N',
                sort=['До z', 'После z'],
                title=None,
                axis=alt.Axis(labelAngle=0, labelLimit=70, labelPadding=6),
            ),
            x=alt.X('Доля дисперсии:Q', stack='normalize', axis=alt.Axis(format='.0%'), title=None),
            color=alt.Color(
                'Признак:N',
                scale=neutral_scale,
                sort=FEATURES,
                legend=alt.Legend(title=None, orient='top'),
            ),
            order=alt.Order('ord:Q'),
            tooltip=[
                alt.Tooltip('Этап:N', title='Этап'),
                alt.Tooltip('Признак:N', title='Признак'),
                alt.Tooltip('Доля дисперсии:Q', format='.1%', title='Доля'),
            ],
        )
        .properties(height=180)
    )
    return configure_chart(chart)


def single_histogram(X_full, shown, feature, show_y=True):
    """Гистограмма признака с фиксированными границами бинов."""
    df = shown.reset_index()
    scale = color_scale(df['group'])
    lo, hi = float(X_full[feature].min()), float(X_full[feature].max())
    pad = (hi - lo) * 0.02 if hi > lo else 1.0
    chart = (
        alt.Chart(df, title=feature)
        .mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3)
        .encode(
            x=alt.X(
                f'{feature}:Q',
                bin=alt.Bin(maxbins=12, extent=[lo - pad, hi + pad]),
                title=UNITS[feature],
                axis=alt.Axis(labelAngle=0),
            ),
            y=alt.Y('count():Q', title='Городов' if show_y else None),
            color=alt.Color('group:N', scale=scale, legend=None),
            tooltip=[
                alt.Tooltip('group:N', title='Группа'),
                alt.Tooltip('count():Q', title='Городов'),
            ],
        )
        .properties(height=200)
    )
    return configure_chart(chart)


def correlation_heatmap(X):
    """Тепловая карта корреляций Пирсона 3×3."""
    corr = X[FEATURES].corr().reset_index().melt('index', var_name='Признак 2', value_name='r')
    corr = corr.rename(columns={'index': 'Признак 1'})
    base = alt.Chart(corr, title='Корреляции признаков (Пирсон)').encode(
        x=alt.X('Признак 1:N', sort=FEATURES, title=None, axis=alt.Axis(labelAngle=0, labelLimit=0)),
        y=alt.Y('Признак 2:N', sort=FEATURES, title=None, axis=alt.Axis(labelAngle=0, labelLimit=0)),
    )
    rects = base.mark_rect(cornerRadius=6, stroke='#FFFFFF', strokeWidth=3).encode(
        color=alt.Color(
            'r:Q',
            scale=alt.Scale(domain=[-1, 0, 1], range=[ITMO_SLATE, '#F5F3F7', ITMO_PURPLE]),
            legend=None,
        ),
        tooltip=[
            alt.Tooltip('Признак 1:N'),
            alt.Tooltip('Признак 2:N'),
            alt.Tooltip('r:Q', format='.2f', title='Корреляция r'),
        ],
    )
    text = base.mark_text(font=FONT_HEAD, fontSize=12, fontWeight=600).encode(
        text=alt.Text('r:Q', format='.2f'),
        color=alt.condition('abs(datum.r) > 0.50', alt.value('#FFFFFF'), alt.value(ITMO_DARK)),
    )
    return configure_chart((rects + text).properties(height=215))


def scatter_chart(
    cities, x_col, y_col, x_title, y_title,
    highlight_cities=None, title='', height=280,
    x_domain=None, y_domain=None, show_legend=False,
):
    """Интерактивная диаграмма рассеяния с общей шкалой осей и подсветкой городов."""
    df = cities.reset_index().dropna(subset=[x_col, y_col]).copy()
    highlight = set(highlight_cities or [])
    scale = color_scale(df['group'])

    tooltips = [
        alt.Tooltip('City:N', title='Город'),
        alt.Tooltip('group:N', title='Группа'),
        alt.Tooltip('Work:Q', format='.0f', title='Work (ч)'),
        alt.Tooltip('Price:Q', format='.1f', title='Price'),
        alt.Tooltip('Salary:Q', format='.1f', title='Salary'),
    ]
    if 'silhouette' in df.columns:
        tooltips.append(alt.Tooltip('silhouette:Q', format='.3f', title='Силуэт'))
    tooltips.append(alt.Tooltip('imputed:N', title='Пропуск заполнен'))

    x_scale = alt.Scale(domain=x_domain, padding=10) if x_domain else alt.Scale(zero=False, padding=14)
    y_scale = alt.Scale(domain=y_domain, padding=10) if y_domain else alt.Scale(zero=False, padding=14)
    legend = alt.Legend(title='Группа', orient='bottom') if show_legend else None

    base = alt.Chart(df, title=title).encode(
        x=alt.X(f'{x_col}:Q', scale=x_scale, title=x_title),
        y=alt.Y(f'{y_col}:Q', scale=y_scale, title=y_title),
    )
    points = base.mark_point(filled=True, size=82, opacity=0.92).encode(
        color=alt.Color('group:N', scale=scale, legend=legend),
        shape=alt.condition(alt.datum.group == NOISE, alt.value('cross'), alt.value('circle')),
        stroke=alt.condition(alt.datum.group == NOISE, alt.value(ITMO_MUTED), alt.value('#FFFFFF')),
        strokeWidth=alt.condition(alt.datum.group == NOISE, alt.value(1.8), alt.value(0.8)),
        tooltip=tooltips,
    )

    layers = [points]
    if highlight:
        hi_df = df[df['City'].isin(highlight)]
        if not hi_df.empty:
            hi_base = alt.Chart(hi_df).encode(
                x=alt.X(f'{x_col}:Q', scale=x_scale),
                y=alt.Y(f'{y_col}:Q', scale=y_scale),
            )
            ring = hi_base.mark_point(filled=False, size=220, stroke=ITMO_DARK, strokeWidth=2.5).encode(
                tooltip=tooltips
            )
            labels = hi_base.mark_text(
                align='left', dx=9, dy=-5, font=FONT_BODY, fontSize=10.5, fontWeight=700, color=ITMO_DARK
            ).encode(text='City:N')
            layers.extend([ring, labels])

    return configure_chart(alt.layer(*layers).properties(height=height).interactive())


def dendrogram_chart(dendro, highlight_cities=None, title='Дендрограмма', height=225):
    """Дендрограмма иерархической кластеризации (полилинии, линия разреза, листья)."""
    segs = dendro['segments']
    leaves = dendro['leaves'].copy()
    leaves['y'] = 0.0
    cut_y = float(dendro['cut_height'])
    highlight = set(highlight_cities or [])
    scale = color_scale(leaves['group'])
    x_max = float(leaves['x'].max()) + 5.0
    y_max = float(segs['y'].max()) * 1.08

    x_axis = alt.Axis(labels=False, ticks=False, title=None, domain=False, grid=False)
    x_scale = alt.Scale(domain=[0, x_max])
    y_scale = alt.Scale(domain=[-y_max * 0.06, y_max])

    branches = (
        alt.Chart(segs)
        .mark_line(color=ITMO_DARK, strokeWidth=1.35, opacity=0.88)
        .encode(
            x=alt.X('x:Q', axis=x_axis, scale=x_scale),
            y=alt.Y('y:Q', title='Высота слияния', scale=y_scale),
            detail='branch_id:N',
            order='pt:Q',
        )
    )

    cut_line_df = pd.DataFrame({'x': [0.0, x_max], 'y': [cut_y, cut_y]})
    cut_line = (
        alt.Chart(cut_line_df)
        .mark_line(color=ITMO_ORANGE, strokeDash=[5, 4], strokeWidth=2)
        .encode(x=alt.X('x:Q', axis=x_axis, scale=x_scale), y=alt.Y('y:Q', scale=y_scale))
    )

    leaf_dots = (
        alt.Chart(leaves)
        .mark_circle(size=52)
        .encode(
            x=alt.X('x:Q', axis=x_axis, scale=x_scale),
            y=alt.Y('y:Q', scale=y_scale),
            color=alt.Color('group:N', scale=scale, legend=None),
            tooltip=[alt.Tooltip('City:N', title='Город'), alt.Tooltip('group:N', title='Группа')],
        )
    )

    layers = [branches, cut_line, leaf_dots]
    if highlight:
        hi_leaves = leaves[leaves['City'].isin(highlight)]
        if not hi_leaves.empty:
            hi_ring = (
                alt.Chart(hi_leaves)
                .mark_point(filled=False, size=150, stroke=ITMO_DARK, strokeWidth=2.2)
                .encode(x=alt.X('x:Q', axis=x_axis, scale=x_scale), y=alt.Y('y:Q', scale=y_scale))
            )
            hi_txt = (
                alt.Chart(hi_leaves)
                .mark_text(dy=-11, font=FONT_BODY, fontSize=10, fontWeight=700, color=ITMO_DARK)
                .encode(x=alt.X('x:Q', axis=x_axis, scale=x_scale), y=alt.Y('y:Q', scale=y_scale), text='City:N')
            )
            layers.extend([hi_ring, hi_txt])

    return configure_chart(alt.layer(*layers, title=f'{title} ({cut_y:.2f})').properties(height=height))


def single_k_metric_chart(df, col, title, current_k, height=225):
    """График метрики по числу кластеров k с маркером текущего значения."""
    k_min, k_max = int(df['k'].min()), int(df['k'].max())
    x_scale = alt.Scale(domain=[k_min - 0.3, k_max + 0.3])
    rule_df = pd.DataFrame({'k': [int(current_k)]})
    base = (
        alt.Chart(df, title=title)
        .mark_line(color=ITMO_PURPLE, strokeWidth=2.2, point=alt.OverlayMarkDef(color=ITMO_PURPLE, size=48))
        .encode(
            x=alt.X('k:Q', title='k', scale=x_scale, axis=alt.Axis(labelAngle=0, tickMinStep=1, format='d')),
            y=alt.Y(f'{col}:Q', title=None, scale=alt.Scale(zero=False)),
            tooltip=[alt.Tooltip('k:Q', format='d', title='k'), alt.Tooltip(f'{col}:Q', format='.3f', title=title)],
        )
    )
    rule = alt.Chart(rule_df).mark_rule(color=ITMO_ORANGE, strokeDash=[4, 3], strokeWidth=1.8).encode(x=alt.X('k:Q', scale=x_scale))
    cur_pt = (
        alt.Chart(df[df['k'] == int(current_k)])
        .mark_circle(color=ITMO_ORANGE, size=90, stroke='#FFFFFF', strokeWidth=1.5)
        .encode(x=alt.X('k:Q', scale=x_scale), y=f'{col}:Q')
    )
    return configure_chart((base + rule + cur_pt).properties(height=height))


def dbscan_heatmap(grid_df, current_eps, current_ms):
    """Тепловая карта сетки DBSCAN (eps × min_samples) с выделением текущей пары."""
    df = grid_df.copy()
    eps_sorted = sorted(df['eps'].unique())
    step_eps = float(np.median(np.diff(eps_sorted))) if len(eps_sorted) > 1 else 0.05
    eps_min, eps_max = float(eps_sorted[0]), float(eps_sorted[-1])
    ms_min, ms_max = int(df['min_samples'].min()), int(df['min_samples'].max())
    dark_text_threshold = float(df['noise'].max()) * 0.55

    df['eps_lo'] = df['eps'] - step_eps * 0.46
    df['eps_hi'] = df['eps'] + step_eps * 0.46
    df['ms_lo'] = df['min_samples'] - 0.45
    df['ms_hi'] = df['min_samples'] + 0.45

    x_scale = alt.Scale(domain=[eps_min - step_eps * 0.6, eps_max + step_eps * 0.6])
    y_scale = alt.Scale(domain=[ms_min - 0.55, ms_max + 0.55])

    rects = (
        alt.Chart(df)
        .mark_rect(cornerRadius=3, stroke='#FFFFFF', strokeWidth=1)
        .encode(
            x=alt.X('eps_lo:Q', scale=x_scale, title='Радиус eps', axis=alt.Axis(format='.2f', tickCount=11)),
            x2='eps_hi:Q',
            y=alt.Y('ms_lo:Q', scale=y_scale, title='min_samples (M)', axis=alt.Axis(format='d', tickMinStep=1)),
            y2='ms_hi:Q',
            color=alt.Color(
                'noise:Q',
                scale=alt.Scale(range=[ITMO_LIME, '#F5F3F7', ITMO_PURPLE_L2, ITMO_PURPLE]),
                legend=alt.Legend(title='Шум', orient='right'),
            ),
            tooltip=[
                alt.Tooltip('eps:Q', format='.2f', title='eps'),
                alt.Tooltip('min_samples:Q', format='d', title='min_samples'),
                alt.Tooltip('clusters:Q', title='Кластеров'),
                alt.Tooltip('noise:Q', title='Шумовых городов'),
                alt.Tooltip('silhouette:Q', format='.3f', title='Silhouette (без шума)'),
            ],
        )
    )
    labels = (
        alt.Chart(df)
        .mark_text(font=FONT_BODY, fontSize=10, fontWeight=600)
        .encode(
            x=alt.X('eps:Q', scale=x_scale),
            y=alt.Y('min_samples:Q', scale=y_scale),
            text='clusters:N',
            color=alt.condition(f'datum.noise > {dark_text_threshold:.1f}', alt.value('#FFFFFF'), alt.value(ITMO_DARK)),
        )
    )
    cur_df = df[(np.isclose(df['eps'], current_eps, atol=1e-6)) & (df['min_samples'] == int(current_ms))]
    layers = [rects, labels]
    if not cur_df.empty:
        sel_box = (
            alt.Chart(cur_df)
            .mark_rect(fillOpacity=0, stroke=ITMO_ORANGE, strokeWidth=2.8, cornerRadius=3)
            .encode(x=alt.X('eps_lo:Q', scale=x_scale), x2='eps_hi:Q', y=alt.Y('ms_lo:Q', scale=y_scale), y2='ms_hi:Q')
        )
        layers.append(sel_box)
    return configure_chart(alt.layer(*layers, title='Сетка DBSCAN (eps × M): число кластеров, цвет — шум').properties(height=290))


def k_distance_chart(kdist_df, current_eps, current_ms, height=225):
    """График расстояний до M-го соседа (с учётом самой точки) и горизонталь eps."""
    line = (
        alt.Chart(kdist_df, title=f'k-distance (M={int(current_ms)})')
        .mark_line(color=ITMO_PURPLE, strokeWidth=2.2, point=alt.OverlayMarkDef(color=ITMO_PURPLE, size=30))
        .encode(
            x=alt.X('rank:Q', title='Города (по возрастанию)'),
            y=alt.Y('distance:Q', title=f'До {int(current_ms)}-го соседа (с точкой)'),
            tooltip=[
                alt.Tooltip('City:N', title='Город'),
                alt.Tooltip('distance:Q', format='.3f', title='Расстояние'),
            ],
        )
    )
    rule_df = pd.DataFrame({'eps': [float(current_eps)]})
    rule = alt.Chart(rule_df).mark_rule(color=ITMO_ORANGE, strokeDash=[5, 4], strokeWidth=2).encode(y='eps:Q')
    return configure_chart((line + rule).properties(height=height))


def silhouette_profile_chart(cities, avg_sil, height=225, show_legend=False):
    """Профиль коэффициента силуэта по городам внутри кластеров."""
    df = cities[cities['group'] != NOISE].dropna(subset=['silhouette']).reset_index().copy()
    if df.empty:
        return None
    order_groups = sort_groups(df['group'].unique())
    df['group_ord'] = df['group'].map(lambda g: order_groups.index(g))
    df = df.sort_values(['group_ord', 'silhouette'], ascending=[True, False]).reset_index(drop=True)
    df['rank'] = np.arange(1, len(df) + 1)
    scale = color_scale(df['group'])
    y_min = min(-0.15, float(df['silhouette'].min()) - 0.08)
    legend = alt.Legend(title='Группа', orient='bottom') if show_legend else None

    bars = (
        alt.Chart(df)
        .mark_bar(cornerRadius=2, size=5)
        .encode(
            x=alt.X('rank:Q', title='Города в кластерах', scale=alt.Scale(domain=[0, len(df) + 1])),
            y=alt.Y('silhouette:Q', title='Силуэт', scale=alt.Scale(domain=[y_min, 0.85])),
            color=alt.Color('group:N', scale=scale, legend=legend),
            stroke=alt.condition('datum.silhouette < 0', alt.value(ITMO_DARK), alt.value(None)),
            strokeWidth=alt.condition('datum.silhouette < 0', alt.value(1.4), alt.value(0)),
            tooltip=[
                alt.Tooltip('City:N', title='Город'),
                alt.Tooltip('group:N', title='Группа'),
                alt.Tooltip('silhouette:Q', format='.3f', title='Силуэт'),
            ],
        )
    )
    zero_df = pd.DataFrame({'x': [0], 'x2': [len(df) + 1], 'y': [0.0], 'y2': [0.0]})
    zero_line = alt.Chart(zero_df).mark_rule(color=ITMO_DARK, strokeWidth=1).encode(x='x:Q', x2='x2:Q', y='y:Q', y2='y2:Q')
    layers = [bars, zero_line]
    title = 'Силуэт по городам'
    if avg_sil is not None and not np.isnan(avg_sil):
        avg_df = pd.DataFrame({'x': [0], 'x2': [len(df) + 1], 'y': [float(avg_sil)], 'y2': [float(avg_sil)]})
        avg_rule = (
            alt.Chart(avg_df)
            .mark_rule(color=ITMO_ORANGE, strokeDash=[4, 3], strokeWidth=1.8)
            .encode(x='x:Q', x2='x2:Q', y='y:Q', y2='y2:Q')
        )
        layers.append(avg_rule)
        title = f'Силуэт · среднее {avg_sil:.3f}'
    return configure_chart(alt.layer(*layers, title=title).properties(height=height))


def method_distribution_chart(results):
    """Составной столбик распределения городов по группам в едином порядке методов."""
    rows = []
    n_total = 0
    for method in SOURCES:
        if method not in results:
            continue
        cities = results[method]['cities']
        n_total = len(cities)
        counts = cities['group'].value_counts()
        for grp in sort_groups(counts.index):
            rows.append({'Алгоритм': method, 'group': grp, 'Городов': int(counts[grp])})
    df = pd.DataFrame(rows)
    scale = color_scale(df['group'])
    chart = (
        alt.Chart(df, title=f'Распределение {n_total} городов по группам')
        .mark_bar(height=28, cornerRadius=3)
        .encode(
            y=alt.Y('Алгоритм:N', sort=list(SOURCES), title=None, axis=alt.Axis(labelAngle=0, labelLimit=0)),
            x=alt.X('Городов:Q', stack='zero', title='Количество городов'),
            color=alt.Color('group:N', scale=scale, title='Группа'),
            tooltip=[
                alt.Tooltip('Алгоритм:N', title='Алгоритм'),
                alt.Tooltip('group:N', title='Группа'),
                alt.Tooltip('Городов:Q', title='Городов'),
            ],
        )
        .properties(height=165)
    )
    return configure_chart(chart)


def z_profile_chart(prof):
    """Средние z-оценки кластеров в порядке FEATURES с горизонтальными подписями."""
    clean = prof.drop(index=NOISE, errors='ignore')
    if clean.empty:
        return None
    z_cols = [f'{f} (z)' for f in FEATURES]
    df = clean[z_cols].reset_index().melt('Группа', var_name='Признак', value_name='z')
    df['Признак'] = df['Признак'].str.replace(' (z)', '', regex=False)
    scale = color_scale(df['Группа'])
    bars = (
        alt.Chart(df, title='Средние z-оценки по кластерам')
        .mark_bar(cornerRadius=4)
        .encode(
            x=alt.X('Признак:N', sort=FEATURES, title=None, axis=alt.Axis(labelAngle=0, labelLimit=0)),
            xOffset=alt.XOffset('Группа:N', sort=sort_groups(df['Группа'].unique())),
            y=alt.Y('z:Q', title='Отклонение (σ)'),
            color=alt.Color('Группа:N', scale=scale, title='Кластер'),
            tooltip=[
                alt.Tooltip('Группа:N', title='Кластер'),
                alt.Tooltip('Признак:N', title='Признак'),
                alt.Tooltip('z:Q', format='+.2f', title='z-оценка'),
            ],
        )
    )
    zero = alt.Chart(pd.DataFrame({'y': [0.0]})).mark_rule(color=ITMO_DARK, strokeWidth=1).encode(y='y:Q')
    return configure_chart((bars + zero).properties(height=280))


def feature_distribution_chart(shown, feature, highlight_cities=None):
    """Boxplot (для кластеров) и индивидуальные города точками (для шума только точки)."""
    df = shown.reset_index().dropna(subset=[feature]).copy()
    if df.empty:
        return None
    order = sort_groups(df['group'].unique())
    scale = color_scale(order)
    highlight = set(highlight_cities or [])

    clusters_df = df[df['group'] != NOISE]
    layers = []
    if not clusters_df.empty:
        box = (
            alt.Chart(clusters_df)
            .mark_boxplot(size=36, opacity=0.32, outliers=False)
            .encode(
                x=alt.X('group:N', sort=order, title=None, axis=alt.Axis(labelAngle=0, labelLimit=0)),
                y=alt.Y(f'{feature}:Q', scale=alt.Scale(zero=False), title=f'{feature} ({UNITS[feature]})'),
                color=alt.Color('group:N', scale=scale, legend=None),
            )
        )
        layers.append(box)

    rng = np.random.default_rng(42)
    df['jitter'] = rng.uniform(-0.18, 0.18, size=len(df))
    pts = (
        alt.Chart(df)
        .mark_point(filled=True, size=65, opacity=0.9)
        .encode(
            x=alt.X('group:N', sort=order, title=None, axis=alt.Axis(labelAngle=0, labelLimit=0)),
            xOffset='jitter:Q',
            y=alt.Y(f'{feature}:Q', scale=alt.Scale(zero=False), title=f'{feature} ({UNITS[feature]})'),
            color=alt.Color('group:N', scale=scale, title='Группа'),
            shape=alt.condition(alt.datum.group == NOISE, alt.value('cross'), alt.value('circle')),
            tooltip=[
                alt.Tooltip('City:N', title='Город'),
                alt.Tooltip('group:N', title='Группа'),
                alt.Tooltip(f'{feature}:Q', format='.1f', title=feature),
            ],
        )
    )
    layers.append(pts)

    if highlight:
        hi = df[df['City'].isin(highlight)]
        if not hi.empty:
            hi_pts = (
                alt.Chart(hi)
                .mark_point(filled=False, size=180, stroke=ITMO_DARK, strokeWidth=2.2)
                .encode(x=alt.X('group:N', sort=order), xOffset='jitter:Q', y=f'{feature}:Q')
            )
            layers.append(hi_pts)

    return configure_chart(alt.layer(*layers, title=f'{feature} по группам').properties(height=285))


def ab_diff_chart(prof, ga, gb):
    """График разницы двух групп (B − A) в σ, раскрашенный в цвета самих групп A и B."""
    rows = []
    for f in FEATURES:
        va = prof.loc[ga, f'{f} (z)']
        vb = prof.loc[gb, f'{f} (z)']
        rows.append({'Признак': f, 'Δz': float(vb - va)})
    df = pd.DataFrame(rows)
    lim = max(1.0, float(df['Δz'].abs().max()) * 1.15)
    col_b = group_color(gb)
    col_a = group_color(ga)
    bars = (
        alt.Chart(df)
        .mark_bar(cornerRadius=4, height=22)
        .encode(
            y=alt.Y('Признак:N', sort=FEATURES, title=None, axis=alt.Axis(labelAngle=0, labelLimit=0)),
            x=alt.X('Δz:Q', title='B − A (σ)', scale=alt.Scale(domain=[-lim, lim])),
            color=alt.condition(alt.datum['Δz'] >= 0, alt.value(col_b), alt.value(col_a)),
            tooltip=[
                alt.Tooltip('Признак:N'),
                alt.Tooltip('Δz:Q', format='+.2f', title='B − A (σ)'),
            ],
        )
    )
    zero = alt.Chart(pd.DataFrame({'x': [0.0]})).mark_rule(color=ITMO_DARK, strokeWidth=1).encode(x='x:Q')
    return configure_chart(alt.layer(bars, zero, title='Разница профилей (B − A, σ)').properties(height=165))


def single_conclusion_metric_chart(results, metric_key, title):
    """Мини-график внутренней метрики эталона (нейтральные тона, акцент на рекомендуемом k-means)."""
    rows = []
    order = []
    for method in SOURCES:
        m = results[method]['summary'].get('metrics')
        if not m:
            continue
        n_obs = results[method]['summary']['metrics_n']
        label = f'{method} ({n_obs})' if method == 'DBSCAN' else method
        order.append(label)
        rows.append({'Алгоритм': method, 'Подпись': label, 'n': n_obs, 'Значение': m[metric_key]})
    df = pd.DataFrame(rows)
    method_colors = alt.Scale(domain=list(SOURCES), range=['#C8C6D0', ITMO_PURPLE, '#D9D7E0'])
    chart = (
        alt.Chart(df, title=title)
        .mark_bar(cornerRadiusTopLeft=5, cornerRadiusTopRight=5, width=38)
        .encode(
            x=alt.X('Подпись:N', sort=order, title=None, axis=alt.Axis(labelAngle=0, labelLimit=0)),
            y=alt.Y('Значение:Q', title=None),
            color=alt.Color('Алгоритм:N', scale=method_colors, legend=None),
            tooltip=[
                alt.Tooltip('Подпись:N', title='Алгоритм'),
                alt.Tooltip('n:Q', title='Городов (n)'),
                alt.Tooltip('Значение:Q', format='.3f', title=title),
            ],
        )
        .properties(height=165)
    )
    return configure_chart(chart)
