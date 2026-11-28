"""Тесты расчётной модели, графиков и интерактивных сценариев дашборда ЛР4."""
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from streamlit.testing.v1 import AppTest

LAB04_DIR = Path(__file__).resolve().parent
if str(LAB04_DIR) not in sys.path:
    sys.path.insert(0, str(LAB04_DIR))

import charts  # noqa: E402
import model  # noqa: E402

APP_PATH = str(LAB04_DIR / 'app.py')


class ModelAndReproducibilityTest(unittest.TestCase):
    def setUp(self):
        self.X = model.load_raw()
        self.bench = model.load_results()
        self.Z, _ = model.standardize(self.X)
        self.bparams = model.benchmark_params(self.bench)

    def test_benchmark_reproducibility(self):
        """partition() на эталонных параметрах в точности воспроизводит ЛР1–3."""
        for method, params in self.bparams.items():
            live = model.partition(self.X, method, params, benchmark=self.bench)
            saved = self.bench[method]
            self.assertTrue((live['cities']['cluster'] == saved['cities']['cluster']).all(), method)
            for metric_name, val in saved['summary']['metrics'].items():
                self.assertAlmostEqual(live['summary']['metrics'][metric_name], val, places=12)
            for col in ('z_Work', 'z_Price', 'z_Salary', 'PC1', 'PC2'):
                np.testing.assert_allclose(live['cities'][col].to_numpy(), saved['cities'][col].to_numpy(), atol=1e-12)

    def test_data_expectations(self):
        table = model.comparison_table(self.bench)
        self.assertEqual(table['Кластеров'].tolist(), [2, 2, 2])
        self.assertEqual(self.bench['DBSCAN']['summary']['noise'], 10)
        self.assertEqual(int(self.X.isna().sum().sum()), 4)
        for a, b in [('Ward', 'k-means'), ('k-means', 'DBSCAN')]:
            self.assertAlmostEqual(model.crosstab(self.bench, a, b)[1], 1.0)

    def test_ari_ignores_row_order(self):
        reordered = {m: dict(r) for m, r in self.bench.items()}
        reordered['k-means']['cities'] = reordered['k-means']['cities'].iloc[::-1]
        self.assertAlmostEqual(model.crosstab(reordered, 'Ward', 'k-means')[1], 1.0)

    def test_ari_summary_detects_undefined_pairs(self):
        live = dict(self.bench)
        live['DBSCAN'] = model.partition(self.X, 'DBSCAN', {'eps': 0.30, 'min_samples': 6}, benchmark=self.bench)
        min_val, n_def, n_tot, missing = model.ari_summary(live)
        self.assertAlmostEqual(min_val, 1.0)
        self.assertEqual(n_def, 1)
        self.assertEqual(n_tot, 3)
        self.assertEqual(len(missing), 2)

    def test_dynamic_palette_and_numeric_sorting(self):
        groups = [f'Кластер {i}' for i in range(1, 11)] + [model.NOISE]
        ordered = model.sort_groups(groups)
        self.assertEqual(ordered[1], 'Кластер 2')
        self.assertEqual(ordered[9], 'Кластер 10')
        self.assertEqual(ordered[-1], model.NOISE)

        scale = charts.color_scale([f'Кластер {i}' for i in range(1, 9)] + [model.NOISE])
        self.assertEqual(len(scale.domain), 9)
        self.assertEqual(len(set(scale.range[:8])), 8)
        self.assertEqual(scale.range[-1], model.NOISE_COLOR)

    def test_sandbox_edge_cases(self):
        p0 = model.partition(self.X, 'DBSCAN', {'eps': 0.30, 'min_samples': 6}, benchmark=self.bench)
        self.assertEqual(p0['summary']['clusters'], 0)
        self.assertEqual(p0['summary']['noise'], 48)
        self.assertIsNone(p0['summary']['metrics'])

        p1 = model.partition(self.X, 'DBSCAN', {'eps': 1.00, 'min_samples': 6}, benchmark=self.bench)
        self.assertEqual(p1['summary']['clusters'], 1)
        self.assertIsNone(p1['summary']['metrics'])

        p3db = model.partition(self.X, 'DBSCAN', {'eps': 0.75, 'min_samples': 6}, benchmark=self.bench)
        self.assertEqual(p3db['summary']['clusters'], 3)
        self.assertEqual(p3db['summary']['noise'], 15)

        p_single = model.partition(self.X, 'Ward', {'k': 2, 'linkage': 'single'}, benchmark=self.bench)
        self.assertEqual(p_single['summary']['clusters'], 2)
        self.assertEqual(len(model.typical_cities(p_single['cities'])), 2)

        p8 = model.partition(self.X, 'k-means', {'k': 8}, benchmark=self.bench)
        self.assertEqual(p8['summary']['clusters'], 8)
        prof8 = model.profile(p8['cities'])
        self.assertEqual(len(prof8), 8)
        for grp in prof8.index:
            txt = model.describe_group(prof8.loc[grp])
            self.assertTrue(len(txt) > 10)
            self.assertNotIn('-0.', txt)

    def test_noise_descriptions_match_actual_noise(self):
        p3db = model.partition(self.X, 'DBSCAN', {'eps': 0.75, 'min_samples': 6}, benchmark=self.bench)
        ndf = model.describe_noise_cities(p3db['cities'])
        self.assertEqual(sorted(ndf['City'].tolist()), sorted(p3db['summary']['noise_cities']))

    def test_profile_masks_imputed_z_scores(self):
        cities = self.bench['k-means']['cities']
        prof = model.profile(cities)
        c2 = cities[cities['group'] == 'Кластер 2']
        expected_z_work = c2.loc[c2['Work'].notna(), 'z_Work'].mean()
        self.assertAlmostEqual(prof.loc['Кластер 2', 'Work (z)'], expected_z_work, places=10)

    def test_filter_and_hidden_imputed_reason(self):
        cities = self.bench['DBSCAN']['cities']
        full = {f: (float(self.X[f].min()), float(self.X[f].max())) for f in model.FEATURES}
        groups = cities.group.unique()
        self.assertEqual(len(model.filter_cities(cities, [model.NOISE], full)), 10)
        self.assertTrue(model.filter_cities(cities, [], full).empty)
        self.assertEqual(len(model.filter_cities(cities, groups, full, include_imputed=False)), 46)

        narrowed = full | {'Work': (float(self.X.Work.min()) + 1, float(self.X.Work.max()))}
        reason = model.hidden_imputed_reason(cities, groups, narrowed, include_imputed=True)
        self.assertIn('Cairo', reason)
        self.assertIn('Jakarta', reason)
        self.assertIn('Work', reason)

    def test_chart_specs_encoding(self):
        prof = model.profile(self.bench['k-means']['cities'])
        z_spec = charts.z_profile_chart(prof).to_dict()
        bar_layer = z_spec['layer'][0]
        self.assertEqual(bar_layer['encoding']['x']['sort'], model.FEATURES)
        self.assertEqual(bar_layer['encoding']['x']['axis']['labelAngle'], 0)

        dist_spec = charts.method_distribution_chart(self.bench).to_dict()
        self.assertEqual(dist_spec['encoding']['y']['sort'], list(model.SOURCES))
        self.assertEqual(dist_spec['encoding']['y']['axis']['labelAngle'], 0)

        dendro = model.dendrogram_data(self.Z, linkage_name='ward', k=2, labels_by_city=self.bench['Ward']['cities']['cluster'])
        d_spec = charts.dendrogram_chart(dendro).to_dict()
        self.assertGreaterEqual(len(d_spec['layer']), 3)


class AppUITest(unittest.TestCase):
    def run_app(self):
        at = AppTest.from_file(APP_PATH, default_timeout=60).run()
        self.assertFalse(at.exception, at.exception)
        return at

    def test_benchmark_hides_sandbox_sliders_and_renders_three_columns(self):
        """E2: в Эталоне слайдеры песочницы не показываются, все 3 столбца видны."""
        at = self.run_app()
        slider_keys = [s.key for s in at.slider]
        self.assertNotIn('ward_k', slider_keys)
        self.assertNotIn('kmeans_k', slider_keys)
        self.assertNotIn('dbscan_eps', slider_keys)
        md = ' '.join(m.value for m in at.markdown)
        for m_name in model.SOURCES:
            self.assertIn(f'method-col-name">{m_name}<', md)

    def test_sandbox_persists_across_views_and_resets_explicitly(self):
        """E1 + реальный тест сброса: k=3 сохраняется при переходах и последующих кликах, сброс возвращает 2."""
        at = self.run_app()
        at.segmented_control(key='mode').set_value('Песочница').run()
        at.slider(key='kmeans_k').set_value(3).run()
        self.assertEqual(at.slider(key='ward_k').value, 2)
        self.assertEqual(at.slider(key='kmeans_k').value, 3)

        # Переходим в «Выводы» и выполняем ещё один прогон (подсветка города)
        at.segmented_control(key='nav_view').set_value('Выводы').run()
        at.multiselect(key='highlight_cities').set_value(['Tokyo']).run()
        self.assertFalse(at.exception, at.exception)
        diff_df = at.dataframe[-1].value
        self.assertEqual(diff_df.loc['k-means', 'Кластеров (эталон → сейчас)'], '2 → 3')

        # Возвращаемся на главный экран — слайдер k-means всё ещё 3!
        at.segmented_control(key='nav_view').set_value('Сравнение 3 методов').run()
        self.assertEqual(at.slider(key='kmeans_k').value, 3)

        # Нажимаем «Сбросить» — k-means возвращается к 2
        at.button(key='reset_btn').click().run()
        self.assertEqual(at.slider(key='kmeans_k').value, 2)
        at.segmented_control(key='nav_view').set_value('Выводы').run()
        self.assertEqual(at.dataframe[-1].value.loc['k-means', 'Кластеров (эталон → сейчас)'], '2 → 2')

    def test_benchmark_return_does_not_show_sandbox_values(self):
        """E2: после изменения в песочнице возврат в Эталон показывает k=2 и скрывает слайдеры."""
        at = self.run_app()
        at.segmented_control(key='mode').set_value('Песочница').run()
        at.slider(key='kmeans_k').set_value(3).run()
        at.segmented_control(key='mode').set_value('Эталон ЛР1–3').run()
        self.assertNotIn('kmeans_k', [s.key for s in at.slider])
        md = ' '.join(m.value for m in at.markdown)
        self.assertIn('k=2 · k-means++', md)

    def test_sandbox_dbscan_edge_cases_and_honest_ari(self):
        """E5: при DBSCAN eps=0.30 (0 кластеров) шапка не пишет 'Мин. ARI = 1.000'."""
        at = self.run_app()
        at.segmented_control(key='mode').set_value('Песочница').run()

        at.slider(key='dbscan_eps').set_value(0.30).run()
        self.assertFalse(at.exception, at.exception)
        md_0 = ' '.join(m.value for m in at.markdown)
        self.assertIn('с DBSCAN не опред.', md_0)
        self.assertNotIn('Мин. ARI = 1.000', md_0)

        for eps in (1.00, 0.75):
            at.slider(key='dbscan_eps').set_value(eps).run()
            self.assertFalse(at.exception, at.exception)
        self.assertAlmostEqual(at.slider(key='dbscan_eps').value, 0.75)

    def test_interpretation_persists_after_view_roundtrip(self):
        """E4: возврат «Интерпретация → Данные → Интерпретация» не сбрасывает выбранные группы в пустой список."""
        at = self.run_app()
        at.segmented_control(key='nav_view').set_value('Интерпретация').run()
        self.assertTrue(len(at.multiselect(key='interp_groups').value) >= 2)
        at.segmented_control(key='nav_view').set_value('Данные').run()
        at.segmented_control(key='nav_view').set_value('Интерпретация').run()
        self.assertFalse(at.exception, at.exception)
        self.assertTrue(len(at.multiselect(key='interp_groups').value) >= 2)

    def test_all_views_and_city_inspector_values(self):
        at = self.run_app()
        at.multiselect(key='highlight_cities').set_value(['Tokyo', 'Cairo']).run()
        self.assertFalse(at.exception, at.exception)
        insp_df = at.dataframe[0].value
        self.assertEqual(insp_df.loc['Tokyo', 'DBSCAN'], model.NOISE)
        self.assertTrue(any('Пропуски в исходных данных' in c.value for c in at.caption))

        for sec in ('Данные', 'Интерпретация', 'Выводы', 'Сравнение 3 методов'):
            at.segmented_control(key='nav_view').set_value(sec).run()
            self.assertFalse(at.exception, f'Error on view {sec}: {at.exception}')

        # Пустой выбор и только шум в «Интерпретации»
        at.segmented_control(key='nav_view').set_value('Интерпретация').run()
        at.segmented_control(key='interp_method').set_value('DBSCAN').run()
        at.multiselect(key='interp_groups').set_value([model.NOISE]).run()
        self.assertTrue(any('у шума общего профиля нет' in i.value for i in at.info))
        at.multiselect(key='interp_groups').set_value([]).run()
        self.assertTrue(any('Нет городов для отображения' in i.value for i in at.info))

    def test_load_error_renders_clean_st_error(self):
        with patch('model.results_mtime', return_value=(-1,)), \
             patch('model.load_results', side_effect=ValueError('Симуляция рассогласования CSV')):
            at = AppTest.from_file(APP_PATH, default_timeout=60).run()
            self.assertFalse(at.exception, at.exception)
            self.assertTrue(any('Симуляция рассогласования CSV' in e.value for e in at.error))


if __name__ == '__main__':
    unittest.main()
