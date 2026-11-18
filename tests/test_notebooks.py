import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from IPython.core.inputtransformer2 import TransformerManager

import notebook_support


def source(pattern, original_cell):
    path = next(Path('.').rglob(pattern))
    nb = json.loads(path.read_text(encoding='utf8'))
    return ''.join(nb['cells'][original_cell + 2]['source'])


def test_local_dataset_no_implicit_network(tmp_path, monkeypatch):
    monkeypatch.setattr(notebook_support, 'DATA_DIR', tmp_path)
    monkeypatch.delenv('ALLOW_DATA_DOWNLOAD', raising=False)
    with pytest.raises(FileNotFoundError):
        notebook_support.read_dataset('https://example.invalid/data.csv')
    (tmp_path / 'data.csv').write_text('a\n1\n')
    assert notebook_support.read_dataset('https://example.invalid/data.csv').a.tolist() == [1]


@pytest.mark.parametrize('text,lemma', [('покупка жилья', 'жильё'), ('на образование', 'образование'),
                                      ('свадьбы', 'свадьба'), ('автомобиля', 'автомобиль'), ('недвижимости', 'недвижимость')])
def test_russian_purpose_categories(text, lemma):
    assert lemma in notebook_support.RussianLemmatizer().lemmatize(text)


def test_series_fill_updates_parent_under_copy_on_write():
    code = source('*borrower*.ipynb', 8)
    data = pd.DataFrame({'days_employed': [1., np.nan], 'total_income': [2., np.nan]})
    namespace = {'data': data, 'average_days_employed': 3., 'average_total_income': 4.}
    # Execute the actual assignment lines from the notebook.
    for line in code.splitlines():
        if '.fillna(' in line:
            exec(line, namespace)
    assert data.iloc[1].tolist() == [3., 4.]


def test_month_duration_is_explicit_and_finite():
    frame = pd.DataFrame({'BeginDate': pd.to_datetime(['2020-01-01']), 'EndDate': pd.to_datetime(['2020-02-01'])})
    namespace = {'df': frame, 'pd': pd, 'np': np}
    code = source('*telecom.ipynb', 73)
    exec(code, namespace)
    assert namespace['df'].duration_M.iloc[0] == pytest.approx(31 / (365.2425 / 12))


def test_time_series_features_are_strictly_past():
    namespace = {}
    exec(source('*time-series.ipynb', 23), namespace)
    frame = pd.DataFrame({'num_orders': np.arange(10)}, index=pd.date_range('2020-01-01', periods=10, freq='h'))
    before = namespace['make_features'](frame, 2, 3)
    frame.iloc[5:, 0] = 10000
    after = namespace['make_features'](frame, 2, 3)
    assert before.loc[before.index[5], 'rolling_mean'] == after.loc[after.index[5], 'rolling_mean']
    assert before.loc[before.index[5], 'lag_1'] == 4


def test_all_cells_compile():
    for path in Path('.').glob('Module*/*.ipynb'):
        for cell in json.loads(path.read_text(encoding='utf8'))['cells']:
            if cell['cell_type'] == 'code':
                ast.parse(TransformerManager().transform_cell(''.join(cell['source'])))


def test_smape_zero_denominator_and_shape():
    from sklearn.metrics import make_scorer
    namespace = {'np': np, 'make_scorer': make_scorer}
    exec(source('*gold-recovery.ipynb', 61), namespace)
    smape = namespace['sMAPE_count']
    assert smape([0, 1], [0, 1]) == 0
    assert smape([0, 1], [1, 0]) == 200
    with pytest.raises(ValueError):
        smape([], [])


def test_supervised_scaler_fits_training_rows_only():
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    frame = pd.DataFrame({'EstimatedSalary': np.arange(80) ** 2,
                          'Balance': np.arange(80) ** 3, 'Exited': np.tile([0, 1], 40)})
    namespace = {'df_OHE_scal': frame, 'train_test_split': train_test_split, 'StandardScaler': StandardScaler}
    exec(source('*supervised-lerning.ipynb', 16), namespace)
    indices = namespace['features_train'].index
    np.testing.assert_allclose(namespace['scaler'].mean_, frame.loc[indices, ['EstimatedSalary', 'Balance']].mean())
    assert not np.allclose(namespace['scaler'].mean_, frame[['EstimatedSalary', 'Balance']].mean())


def test_numerical_encoder_accepts_unseen_test_categories():
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import OrdinalEncoder, StandardScaler
    frame = pd.DataFrame({'Price': np.arange(40) + 100, 'Model': [f'model{i}' for i in range(40)],
                          'RegistrationYear': np.arange(40) + 1980, 'Power': np.arange(40) + 20,
                          'Kilometer': np.arange(40) * 100, 'PostalCode': np.arange(40) + 1000})
    namespace = {'df': frame, 'cat_columns': ['Model'], 'train_test_split': train_test_split,
                 'OrdinalEncoder': OrdinalEncoder, 'StandardScaler': StandardScaler}
    exec(source('*numerical-methods.ipynb', 53), namespace)
    assert (namespace['features_test'].Model == -1).all()
    assert namespace['features_train'].shape[1] == 5


def test_bootstrap_and_revenue_preserve_index_alignment():
    namespace = {'pd': pd, 'np': np, 'rnd_state': 42,
                 'BARREL_INCOME': 1., 'REGION_BUDGET': 0., 'WELL_FOR_RESEARCH': 10}
    exec(source('*machine-learning-in-business.ipynb', 42), namespace)
    exec(source('*machine-learning-in-business.ipynb', 44), namespace)
    y = pd.Series(np.arange(300), index=np.arange(300) + 1000)
    prediction = pd.Series(np.arange(300), index=y.index)
    assert namespace['revenue_count'](y, prediction) == np.arange(100, 300).sum()
    first = namespace['bootstrap'](y, prediction)
    assert first == namespace['bootstrap'](y, prediction)
    assert first[1] <= first[0] <= first[2]
    assert first[3] == 0


def test_current_model_families():
    from catboost import CatBoostRegressor
    from lightgbm import LGBMRegressor
    from sklearn.linear_model import LogisticRegression, SGDClassifier
    from sklearn.model_selection import GridSearchCV
    from statsmodels.tsa.seasonal import seasonal_decompose
    import phik  # noqa: F401 (registers the pandas extension)
    rng = np.random.default_rng(42)
    X = rng.normal(size=(60, 3))
    y = (X[:, 0] > 0).astype(int)
    for model in (CatBoostRegressor(iterations=3, verbose=False, allow_writing_files=False), LGBMRegressor(n_estimators=3, verbosity=-1)):
        assert np.isfinite(model.fit(X, y).predict(X)).all()
    namespace = {'LogisticRegression': LogisticRegression, 'rand_stat': 42}
    exec(source('*telecom.ipynb', 119), namespace)
    grid = GridSearchCV(LogisticRegression(max_iter=1000), namespace['params_lr'], cv=2, error_score='raise')
    grid.fit(X, y)
    hinge = SGDClassifier(loss='hinge', random_state=42).fit(X, y)
    assert np.isfinite(hinge.decision_function(X)).all()
    series = pd.Series(np.tile(np.arange(12), 3), index=pd.date_range('2020-01-01', periods=36, freq='h'))
    assert len(seasonal_decompose(series, period=12).trend) == 36
    correlation = pd.DataFrame({'a': X[:, 0], 'b': X[:, 1]}).phik_matrix(interval_cols=['a', 'b'], njobs=1)
    assert correlation.shape == (2, 2)
