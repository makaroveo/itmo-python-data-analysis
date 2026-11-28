"""Загрузка данных, валидация эталона ЛР1–3 и живой пересчёт моделей для дашборда ЛР4."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_samples,
    silhouette_score,
)
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / 'lab01' / 'Econom_Cities_data.csv'
SOURCES = {
    'Ward': 'lab01/results',
    'k-means': 'lab02_03/results_lab02',
    'DBSCAN': 'lab02_03/results_lab03',
}
FEATURES = ['Work', 'Price', 'Salary']
UNITS = {
    'Work': 'часы в год',
    'Price': 'индекс цен, Цюрих = 100',
    'Salary': 'индекс зарплаты, Цюрих = 100',
}
MISSING_CODE = -9999
NOISE = 'Шум'
LINKAGES = ['ward', 'average', 'complete', 'single']

PALETTE = [
    '#950CFF',
    '#FF9A00',
    '#018AB2',
    '#8FB904',
    '#0B68FE',
    '#E63980',
    '#6B4EFF',
    '#B85D00',
]
NOISE_COLOR = '#B2B3C1'

REQUIRED_SELECTION_COLS = {
    'Ward': {'method', 'k', 'silhouette', 'CH', 'DB', 'min_size'},
    'k-means': {'k', 'inertia', 'silhouette', 'CH', 'DB', 'min_size'},
    'DBSCAN': {'min_samples', 'eps', 'clusters', 'noise', 'min_size', 'silhouette'},
}


def _kmeans(k):
    return KMeans(n_clusters=int(k), init='k-means++', n_init=50, max_iter=300, random_state=42)


def _fmt_signed(val, precision=2):
    text = f'{val:+.{precision}f}'
    return text.replace('-', '−')


def cluster_name(label):
    return NOISE if int(label) == -1 else f'Кластер {int(label)}'


def group_sort_key(group):
    if group == NOISE:
        return (1, 0)
    parts = str(group).split()
    return (0, int(parts[-1])) if parts and parts[-1].isdigit() else (0, 999)


def sort_groups(groups):
    return sorted(set(groups), key=group_sort_key)


def group_color(group):
    if group == NOISE:
        return NOISE_COLOR
    _, idx = group_sort_key(group)
    return PALETTE[(max(idx, 1) - 1) % len(PALETTE)]


def load_raw():
    raw = pd.read_csv(DATA, sep=';', decimal=',').set_index('City')
    return raw[FEATURES].astype(float).replace(MISSING_CODE, np.nan)


def results_mtime():
    """Штампы времени CSV и результатов ЛР1–3 для инвалидации кэша."""
    paths = [DATA]
    for rel_path in SOURCES.values():
        folder = ROOT / rel_path
        paths.extend([folder / 'assignments.csv', folder / 'summary.json', folder / 'selection.csv'])
    return tuple(p.stat().st_mtime_ns if p.exists() else 0 for p in paths)


def benchmark_params(results):
    """Эталонные параметры, извлечённые напрямую из summary.json ЛР1–3."""
    return {
        'Ward': {
            'k': int(results['Ward']['summary']['params']['k']),
            'linkage': str(results['Ward']['summary']['params']['linkage']),
        },
        'k-means': {
            'k': int(results['k-means']['summary']['params']['k']),
        },
        'DBSCAN': {
            'eps': float(results['DBSCAN']['summary']['params']['eps']),
            'min_samples': int(results['DBSCAN']['summary']['params']['min_samples']),
        },
    }


def standardize(X):
    """Предобработка как в ЛР1–3: заполнение медианой → z-оценки."""
    filled_arr = SimpleImputer(strategy='median').fit_transform(X[FEATURES])
    filled = pd.DataFrame(filled_arr, index=X.index, columns=FEATURES)
    z_arr = StandardScaler().fit_transform(filled_arr)
    z_df = pd.DataFrame(z_arr, index=X.index, columns=FEATURES)
    return z_df, filled


def projection(Z):
    """PCA-проекция (2 компоненты) и доли объяснённой дисперсии."""
    pca = PCA(n_components=2).fit(Z.to_numpy())
    coords = pca.transform(Z.to_numpy())
    pc_df = pd.DataFrame(coords, index=Z.index, columns=['PC1', 'PC2'])
    return pc_df, tuple(float(v) for v in pca.explained_variance_ratio_)


def relabel_by_salary(labels, X):
    """Нумерация кластеров 1..k по убыванию медианной зарплаты; шум остаётся -1."""
    arr = np.asarray(labels, dtype=int)
    clusters = sorted(g for g in set(arr) if g != -1)

    def _key(g):
        mask = arr == g
        sal = X.Salary.to_numpy()[mask]
        prc = X.Price.to_numpy()[mask]
        sal_med = float(np.nanmedian(sal)) if not np.all(np.isnan(sal)) else -np.inf
        prc_med = float(np.nanmedian(prc)) if not np.all(np.isnan(prc)) else -np.inf
        return (-sal_med, -prc_med, int(g))

    mapping = {g: i + 1 for i, g in enumerate(sorted(clusters, key=_key))}
    mapping[-1] = -1
    return np.array([mapping[g] for g in arr], dtype=int)


def quality(Z, labels):
    """Внутренние метрики качества по не-шумовым объектам; None если кластеров < 2."""
    arr = np.asarray(labels, dtype=int)
    mask = arr != -1
    valid_labels = arr[mask]
    n_valid = int(mask.sum())
    n_clusters = len(set(valid_labels))
    if n_clusters < 2 or n_valid <= n_clusters:
        return None
    z_valid = Z.to_numpy()[mask]
    return {
        'silhouette': float(silhouette_score(z_valid, valid_labels)),
        'CH': float(calinski_harabasz_score(z_valid, valid_labels)),
        'DB': float(davies_bouldin_score(z_valid, valid_labels)),
    }


def silhouette_by_city(Z, labels):
    """Силуэт для каждого города; для шума и при <2 кластеров возвращает NaN."""
    arr = np.asarray(labels, dtype=int)
    result = pd.Series(np.nan, index=Z.index, dtype=float, name='silhouette')
    mask = arr != -1
    valid_labels = arr[mask]
    if len(set(valid_labels)) < 2 or int(mask.sum()) <= len(set(valid_labels)):
        return result
    scores = silhouette_samples(Z.to_numpy()[mask], valid_labels)
    result.iloc[np.where(mask)[0]] = scores
    return result


def load_results():
    """Загрузка и проверка согласованности эталонных результатов ЛР1–3."""
    current = load_raw()
    current_sorted = current.sort_index()
    z_df, _ = standardize(current)
    results = {}

    for method, rel_path in SOURCES.items():
        path = ROOT / rel_path
        cities = pd.read_csv(path / 'assignments.csv', index_col='City')
        summary = json.loads((path / 'summary.json').read_text(encoding='utf-8'))
        selection = pd.read_csv(path / 'selection.csv')

        saved = cities[FEATURES].sort_index()
        if not (saved.index.equals(current_sorted.index) and saved.equals(current_sorted)):
            raise ValueError(
                f'Результаты {method} посчитаны не на текущем {DATA.name}: перезапустите ноутбуки ЛР1–3.'
            )

        actual_clusters = int(len(set(cities.cluster) - {-1}))
        actual_noise = int((cities.cluster == -1).sum())
        if int(summary.get('clusters', -1)) != actual_clusters:
            raise ValueError(
                f'Рассогласование в {method}: summary.clusters={summary.get("clusters")} != {actual_clusters}.'
            )
        if int(summary.get('noise', -1)) != actual_noise:
            raise ValueError(
                f'Рассогласование в {method}: summary.noise={summary.get("noise")} != {actual_noise}.'
            )
        if 'noise_cities' in summary:
            expected_noise = sorted(cities.index[cities.cluster == -1].tolist())
            if sorted(summary['noise_cities']) != expected_noise:
                raise ValueError(f'Рассогласование в {method}: noise_cities не совпадают с assignments.csv.')

        missing_cols = REQUIRED_SELECTION_COLS[method] - set(selection.columns)
        if missing_cols:
            raise ValueError(f'В {rel_path}/selection.csv отсутствуют столбцы: {sorted(missing_cols)}.')

        cities['group'] = cities.cluster.map(cluster_name)
        cities['silhouette'] = silhouette_by_city(z_df.loc[cities.index], cities.cluster.to_numpy())
        results[method] = {'summary': summary, 'cities': cities, 'selection': selection}

    return results


def fit_labels(Z, X, method, k=2, linkage_name='ward', eps=0.9, min_samples=6):
    """Запуск выбранного метода кластеризации на стандартизованных данных Z."""
    z_arr = Z.to_numpy()
    if method == 'Ward':
        tree = linkage(z_arr, method=linkage_name)
        raw = fcluster(tree, t=int(k), criterion='maxclust')
        return relabel_by_salary(raw, X), tree
    if method == 'k-means':
        km = _kmeans(k)
        raw = km.fit_predict(z_arr) + 1
        return relabel_by_salary(raw, X), km
    if method == 'DBSCAN':
        db = DBSCAN(eps=float(eps), min_samples=int(min_samples), metric='euclidean')
        raw = db.fit_predict(z_arr)
        return relabel_by_salary(raw, X), db
    raise ValueError(f'Неизвестный метод кластеризации: {method}')


def k_sweep(Z, X, method, linkage_name='ward', ks=range(2, 9)):
    """Таблица качества и размеров кластеров по k для Ward или k-means."""
    z_arr = Z.to_numpy()
    rows = []
    if method == 'k-means':
        km1 = _kmeans(1).fit(z_arr)
        prev_inertia = float(km1.inertia_)
        rows.append({
            'k': 1, 'inertia': prev_inertia, 'silhouette': np.nan,
            'CH': np.nan, 'DB': np.nan, 'min_size': len(Z), 'падение inertia, %': np.nan,
        })
        for k in ks:
            km = _kmeans(k).fit(z_arr)
            labels = relabel_by_salary(km.labels_ + 1, X)
            q = quality(Z, labels) or {}
            drop = (prev_inertia - km.inertia_) / prev_inertia * 100.0
            sizes = pd.Series(labels).value_counts()
            rows.append({
                'k': int(k), 'inertia': float(km.inertia_),
                'silhouette': q.get('silhouette', np.nan),
                'CH': q.get('CH', np.nan), 'DB': q.get('DB', np.nan),
                'min_size': int(sizes.min()), 'падение inertia, %': round(float(drop), 1),
            })
            prev_inertia = float(km.inertia_)
        return pd.DataFrame(rows)

    tree = linkage(z_arr, method=linkage_name)
    for k in ks:
        labels = relabel_by_salary(fcluster(tree, t=int(k), criterion='maxclust'), X)
        q = quality(Z, labels) or {}
        sizes = pd.Series(labels).value_counts()
        rows.append({
            'method': linkage_name, 'k': int(k),
            'silhouette': q.get('silhouette', np.nan),
            'CH': q.get('CH', np.nan), 'DB': q.get('DB', np.nan),
            'min_size': int(sizes.min()),
        })
    return pd.DataFrame(rows)


def k_distance(Z, min_samples=6):
    """Отсортированные расстояния до min_samples-го соседа (с учётом самой точки)."""
    ms = max(2, min(int(min_samples), len(Z)))
    nn = NearestNeighbors(n_neighbors=ms, metric='euclidean').fit(Z.to_numpy())
    distances, _ = nn.kneighbors(Z.to_numpy())
    kth = distances[:, -1]
    order = np.argsort(kth)
    return pd.DataFrame({
        'rank': np.arange(1, len(Z) + 1),
        'City': Z.index.to_numpy()[order],
        'distance': kth[order],
    })


def dendrogram_data(Z, linkage_name='ward', k=2, labels_by_city=None):
    """Полилинии дендрограммы, порядок листьев и высота разреза для Altair."""
    tree = linkage(Z.to_numpy(), method=linkage_name)
    ddata = dendrogram(tree, labels=Z.index.tolist(), no_plot=True)

    segments = []
    for bid, (xs, ys) in enumerate(zip(ddata['icoord'], ddata['dcoord'])):
        for pt in range(4):
            segments.append({
                'branch_id': bid,
                'pt': pt,
                'x': float(xs[pt]),
                'y': float(ys[pt]),
            })

    leaf_cities = ddata['ivl']
    leaf_x = [5.0 + 10.0 * i for i in range(len(leaf_cities))]
    leaves = pd.DataFrame({'City': leaf_cities, 'x': leaf_x})
    if labels_by_city is not None:
        if isinstance(labels_by_city, pd.Series):
            label_map = labels_by_city.map(cluster_name)
        else:
            label_map = pd.Series(labels_by_city).map(cluster_name)
        leaves['group'] = leaves['City'].map(label_map)
    else:
        leaves['group'] = 'Кластер 1'

    heights = tree[:, 2]
    k_int = max(2, min(int(k), len(Z)))
    upper = float(heights[-(k_int - 1)])
    lower = float(heights[-k_int]) if k_int <= len(heights) else 0.0
    cut_height = 0.5 * (upper + lower)

    merge_heights = [float(h) for h in heights[::-1][:10]]
    return {
        'segments': pd.DataFrame(segments),
        'leaves': leaves,
        'cut_height': cut_height,
        'merge_heights': merge_heights,
    }


def partition(X, method, params, benchmark=None):
    """Пересчёт одного алгоритма в едином формате load_results()[method]."""
    z_df, _ = standardize(X)
    pc_df, _ = projection(z_df)

    if method == 'Ward':
        k = int(params.get('k', 2))
        link = params.get('linkage', 'ward')
        labels, tree = fit_labels(z_df, X, 'Ward', k=k, linkage_name=link)
        merge_heights = [float(h) for h in tree[:, 2][::-1][:10]]
        summary_params = {'linkage': link, 'metric': 'euclidean', 'k': k}
        selection = benchmark['Ward']['selection'] if (benchmark and link == 'ward') else k_sweep(z_df, X, 'Ward', link)
        if benchmark and link == 'ward' and k == benchmark['Ward']['summary']['clusters']:
            rationale = benchmark['Ward']['summary']['rationale']
        else:
            hi_h = merge_heights[k - 2] if k - 2 < len(merge_heights) else merge_heights[-1]
            lo_h = merge_heights[k - 1] if k - 1 < len(merge_heights) else 0.0
            rationale = (
                f'Пересчёт Ward: связь {link}, k = {k}. '
                f'Высоты слияния на границе k: {hi_h:.2f} → {lo_h:.2f}.'
            )
    elif method == 'k-means':
        k = int(params.get('k', 2))
        labels, _ = fit_labels(z_df, X, 'k-means', k=k)
        merge_heights = None
        summary_params = {'k': k, 'init': 'k-means++', 'n_init': 50, 'max_iter': 300, 'random_state': 42}
        selection = benchmark['k-means']['selection'] if benchmark else k_sweep(z_df, X, 'k-means')
        if benchmark and k == benchmark['k-means']['summary']['clusters']:
            rationale = benchmark['k-means']['summary']['rationale']
        else:
            row = selection.set_index('k').loc[k] if k in selection['k'].values else None
            drop_txt = (
                f" (падение W: −{row['падение inertia, %']:.0f} %)"
                if row is not None and pd.notna(row['падение inertia, %'])
                else ''
            )
            rationale = f'Пересчёт k-means при k = {k}{drop_txt}; k-means++, n_init = 50.'
    elif method == 'DBSCAN':
        eps = float(params.get('eps', 0.9))
        ms = int(params.get('min_samples', 6))
        labels, _ = fit_labels(z_df, X, 'DBSCAN', eps=eps, min_samples=ms)
        merge_heights = None
        summary_params = {'eps': round(eps, 2), 'min_samples': ms, 'metric': 'euclidean'}
        selection = benchmark['DBSCAN']['selection'] if benchmark else pd.DataFrame()
        is_bench = (
            benchmark is not None
            and abs(eps - benchmark['DBSCAN']['summary']['params']['eps']) < 1e-9
            and ms == benchmark['DBSCAN']['summary']['params']['min_samples']
        )
        if is_bench:
            rationale = benchmark['DBSCAN']['summary']['rationale']
        else:
            n_cl = len(set(labels) - {-1})
            n_noise = int((labels == -1).sum())
            rationale = (
                f'Пересчёт DBSCAN (eps = {eps:.2f}, M = {ms}): '
                f'кластеров — {n_cl}, шум — {n_noise} из {len(X)} ({n_noise / len(X):.0%}).'
            )
    else:
        raise ValueError(f'Неизвестный метод: {method}')

    cities = X.copy()
    cities['imputed'] = X.isna().any(axis=1)
    for f in FEATURES:
        cities[f'z_{f}'] = z_df[f]
    cities['PC1'] = pc_df['PC1']
    cities['PC2'] = pc_df['PC2']
    cities['cluster'] = labels
    cities['group'] = cities['cluster'].map(cluster_name)
    cities['silhouette'] = silhouette_by_city(z_df, labels)

    n_clusters = int(len(set(labels) - {-1}))
    n_noise = int((labels == -1).sum())
    noise_cities = sorted(cities.index[cities.cluster == -1].tolist())
    metrics = quality(z_df, labels)

    summary = {
        'n': int(len(X)),
        'params': summary_params,
        'rationale': rationale,
        'clusters': n_clusters,
        'noise': n_noise,
        'noise_cities': noise_cities,
        'metrics': metrics,
        'metrics_n': int(len(X) - n_noise),
    }
    if merge_heights is not None:
        summary['merge_heights'] = merge_heights

    if method == 'DBSCAN':
        valid_mask = labels != -1
        if n_clusters >= 2 and int(valid_mask.sum()) > n_clusters:
            # Как в ноутбуке ЛР3: берём разбиение k-means по всем городам и оцениваем на той же не-шумовой подвыборке
            km_full = relabel_by_salary(_kmeans(n_clusters).fit_predict(z_df.to_numpy()) + 1, X)
            km_sub_labels = km_full[valid_mask]
            if len(set(km_sub_labels)) >= 2:
                summary['kmeans_same_subset_metrics'] = quality(z_df.iloc[np.where(valid_mask)[0]], km_sub_labels)
            else:
                summary['kmeans_same_subset_metrics'] = None
        else:
            summary['kmeans_same_subset_metrics'] = None

    return {'summary': summary, 'cities': cities, 'selection': selection}


def comparison_table(results):
    rows = []
    for method in SOURCES:
        if method not in results:
            continue
        r = results[method]
        s, cities = r['summary'], r['cities']
        sizes = cities[cities.cluster != -1].cluster.value_counts().sort_index()
        metrics = s.get('metrics') or {}
        rows.append({
            'Алгоритм': method,
            'Параметры': _format_short_params(method, s['params']),
            'Кластеров': s['clusters'],
            'Размеры кластеров': ' / '.join(map(str, sizes)) if len(sizes) else '—',
            'Доля шума': s['noise'] / s['n'],
            'Silhouette': metrics.get('silhouette', np.nan),
            'Davies-Bouldin': metrics.get('DB', np.nan),
            'Calinski-Harabasz': metrics.get('CH', np.nan),
            'Метрики по объектам': f"{s['metrics_n']} из {s['n']}" if metrics else 'не опред.',
        })
    return pd.DataFrame(rows).set_index('Алгоритм')


def _format_short_params(method, params):
    if method == 'Ward':
        return f"связь={params.get('linkage', 'ward')}, k={params.get('k', 2)}"
    if method == 'k-means':
        return f"k={params.get('k', 2)}, k-means++, n_init={params.get('n_init', 50)}"
    if method == 'DBSCAN':
        return f"eps={params.get('eps', 0.9):.2f}, M={params.get('min_samples', 6)}"
    return ', '.join(f'{k}={v}' for k, v in params.items())


def crosstab(results, a, b):
    """Состав кластеров двух алгоритмов, ARI по не-шумовым городам и число общих городов."""
    groups = pd.concat([results[a]['cities'].group.rename('A'), results[b]['cities'].group.rename('B')], axis=1)
    row_order = sort_groups(groups.A.unique())
    col_order = sort_groups(groups.B.unique())
    table = pd.crosstab(groups.A.rename(a), groups.B.rename(b)).reindex(index=row_order, columns=col_order, fill_value=0)
    both = groups[(groups != NOISE).all(axis=1)]
    if len(both) < 2:
        return table, None, int(len(both))
    return table, float(adjusted_rand_score(both.A, both.B)), int(len(both))


def ari_summary(results):
    """Сводка по ARI среди всех трёх пар алгоритмов (учитывает неопределённые пары)."""
    pairs = [('Ward', 'k-means'), ('k-means', 'DBSCAN'), ('Ward', 'DBSCAN')]
    defined = []
    missing_pairs = []
    for a, b in pairs:
        _, val, _ = crosstab(results, a, b)
        if val is None:
            missing_pairs.append((a, b))
        else:
            defined.append(val)
    min_val = min(defined) if defined else None
    return min_val, len(defined), len(pairs), missing_pairs


def disagreeing_cities(cities_a, cities_b, label_a='A', label_b='B'):
    """Города, у которых различается сопоставленный кластер или статус шума между двумя разбиениями."""
    joined = pd.concat([
        cities_a['group'].rename(label_a),
        cities_b['group'].rename(label_b),
        cities_a[FEATURES],
    ], axis=1)
    both = joined[(joined[label_a] != NOISE) & (joined[label_b] != NOISE)]
    mapping = {}
    if not both.empty:
        ct = pd.crosstab(both[label_a], both[label_b])
        for grp in ct.index:
            mapping[grp] = ct.loc[grp].idxmax()

    def _differs(row):
        ga, gb = row[label_a], row[label_b]
        if ga == NOISE and gb == NOISE:
            return False
        if ga == NOISE or gb == NOISE:
            return True
        return mapping.get(ga) != gb

    return joined[joined.apply(_differs, axis=1)].sort_index()


def filter_cities(cities, groups, ranges, include_imputed=True):
    """Отбор городов для отображения без пересчёта моделей."""
    mask = cities.group.isin(groups)
    for feature, (low, high) in ranges.items():
        values = cities[feature]
        full_range = low <= values.min() and high >= values.max()
        mask &= values.between(low, high) | (values.isna() & full_range)
    if not include_imputed:
        mask &= ~cities.imputed
    return cities[mask]


def hidden_imputed_reason(cities, groups, ranges, include_imputed=True):
    """Пояснение, если города с заполненными пропусками скрылись из-за сужения диапазона."""
    if not include_imputed:
        return ''
    imp = cities[cities.imputed & cities.group.isin(groups)]
    if imp.empty:
        return ''
    hidden_by = []
    for feature, (low, high) in ranges.items():
        values = cities[feature]
        if (low > values.min() or high < values.max()) and imp[feature].isna().any():
            hidden_by.append(feature)
    if not hidden_by:
        return ''
    names = ', '.join(imp.index.tolist())
    feats = ', '.join(hidden_by)
    return f'{names} скрыты: у них нет наблюдённого значения {feats}, а диапазон сужен.'


def profile(cities):
    """Средние и медианы наблюдённых значений, средние z-оценки (без заполненных пропусков)."""
    if cities.empty:
        return pd.DataFrame()
    z = pd.DataFrame({f'{f} (z)': cities[f'z_{f}'].where(cities[f].notna()) for f in FEATURES})
    table = pd.concat([
        cities.groupby('group').size().rename('Городов'),
        cities.groupby('group')[FEATURES].mean().add_prefix('Среднее '),
        cities.groupby('group')[FEATURES].median().add_prefix('Медиана '),
        z.groupby(cities.group).mean(),
    ], axis=1)
    table = table.reindex(sort_groups(table.index))
    table.index.name = 'Группа'
    return table


def typical_cities(cities, n=5):
    """Для каждого кластера n городов, ближайших к его центру в z-оценках."""
    non_noise = cities[cities.group != NOISE]
    out = {}
    for g in sort_groups(non_noise.group.unique()):
        part = non_noise[(non_noise.group == g) & (~non_noise.imputed)]
        if part.empty:
            part = non_noise[non_noise.group == g]
        z = part[[f'z_{f}' for f in FEATURES]]
        out[g] = ((z - z.mean()) ** 2).sum(axis=1).nsmallest(n).index.tolist()
    return out


def describe_group(z_row):
    """Содержательное описание профиля кластера по его средним z-оценкам."""
    def _phrase(val, high_strong, high_mild, mid, low_mild, low_strong):
        if pd.isna(val):
            return 'нет данных'
        s = _fmt_signed(val, 2)
        if val >= 0.55:
            return f'{high_strong} ({s}σ)'
        if val >= 0.20:
            return f'{high_mild} ({s}σ)'
        if val <= -0.55:
            return f'{low_strong} ({s}σ)'
        if val <= -0.20:
            return f'{low_mild} ({s}σ)'
        return f'{mid} ({s}σ)'

    sal = _phrase(
        z_row.get('Salary (z)', np.nan),
        'высокие зарплаты', 'зарплаты выше среднего', 'зарплаты около среднего',
        'зарплаты ниже среднего', 'низкие зарплаты',
    )
    prc = _phrase(
        z_row.get('Price (z)', np.nan),
        'высокие цены', 'цены выше среднего', 'цены около среднего',
        'цены ниже среднего', 'низкие цены',
    )
    wrk = _phrase(
        z_row.get('Work (z)', np.nan),
        'длинный рабочий год', 'часов больше среднего', 'рабочих часов около среднего',
        'часов меньше среднего', 'короткий рабочий год',
    )
    return f'{sal.capitalize()}; {prc}; {wrk}.'


def describe_noise_cities(cities):
    """Таблица шумовых городов с их главным отклонением по z-оценкам."""
    noise = cities[cities.group == NOISE]
    if noise.empty:
        return pd.DataFrame(columns=['City', 'Главное отклонение', 'Work', 'Price', 'Salary'])
    ru = {'Work': 'Рабочие часы', 'Price': 'Цены', 'Salary': 'Зарплаты'}
    rows = []
    for city, row in noise.sort_index().iterrows():
        z_vals = {f: row[f'z_{f}'] for f in FEATURES if pd.notna(row[f])}
        if z_vals:
            top_f = max(z_vals, key=lambda f: abs(z_vals[f]))
            reason = f'{ru[top_f]} {_fmt_signed(z_vals[top_f], 2)}σ'
        else:
            reason = 'нет наблюдённых признаков'
        rows.append({
            'City': city,
            'Главное отклонение': reason,
            'Work': row['Work'],
            'Price': row['Price'],
            'Salary': row['Salary'],
        })
    return pd.DataFrame(rows)


def variance_shares(X):
    """Доля дисперсии каждого признака до и после стандартизации."""
    z_df, filled = standardize(X)
    raw_var = filled.var()
    z_var = z_df.var()
    rows = []
    for f in FEATURES:
        rows.append({'Признак': f, 'Этап': 'До стандартизации', 'Доля дисперсии': float(raw_var[f] / raw_var.sum())})
        rows.append({'Признак': f, 'Этап': 'После z-стандартизации', 'Доля дисперсии': float(z_var[f] / z_var.sum())})
    return pd.DataFrame(rows)
