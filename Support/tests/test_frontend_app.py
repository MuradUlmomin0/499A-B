"""Browser-view checks use saved artifacts and only the 20-record demo."""
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from Support.frontend import service

ROOT=Path(__file__).resolve().parents[2]
APP=ROOT/'Support/frontend/app.py'


def boot():
    app=AppTest.from_file(str(APP),default_timeout=30).run()
    assert not app.exception
    return app


def page(app,name):
    app.sidebar.radio(key='page').set_value(name).run()
    assert not app.exception
    return app


def test_overview_reports_actual_demo_contract():
    app=boot()
    assert [m.value for m in app.metric]==['20','46','34']
    text=' '.join(str(x.value) for x in app.markdown)
    assert 'simulated' in text and 'MLP' in text


def test_real_prediction_csv_and_model_change_clear_previous_result():
    app=page(boot(),'Predict traffic')
    app.button(key='predict').click().run()
    assert not app.exception
    assert len(app.dataframe[0].value)==20
    assert app.session_state['_prediction'].model_key=='fedavg'
    assert len(app.get('download_button'))==1
    assert 'softmax' in ' '.join(c.value for c in app.caption)
    app.sidebar.selectbox(key='model').set_value('fusion').run()
    assert not app.exception
    assert app.session_state['_prediction'] is None
    assert not app.get('download_button')
    assert not app.dataframe
    app.button(key='predict').click().run()
    assert not app.exception
    view=app.session_state['_prediction']
    assert view.model_key=='fusion' and len(view.frame)==20
    assert len([c for c in view.frame if c.startswith('probability_')])==34
    assert len(app.get('download_button'))==1


def test_saved_results_keep_test_role_units_and_both_partitions():
    app=page(boot(),'Saved research results')
    actual=service.load_research_results().drop(columns='method')
    assert app.dataframe[0].value.equals(actual)
    assert 'three seeds' in ' '.join(c.value for c in app.caption)
    app.radio(key='condition').set_value('dirichlet').run()
    assert not app.exception
    assert app.dataframe[0].value.equals(service.load_research_results(condition='dirichlet').drop(columns='method'))


def test_history_shows_twenty_saved_validation_rounds():
    app=page(boot(),'Training history')
    charts=app.get('vega_lite_chart')
    assert len(charts)==2
    assert 'Validation accuracy (%)' in charts[0].proto.spec
    assert 'Validation macro F1' in charts[1].proto.spec
    assert len(app.dataframe[0].value)==20
    assert app.dataframe[0].value.equals(service.load_training_history('fedavg'))
    assert 'validation' in ' '.join(c.value for c in app.caption).lower()


@pytest.mark.parametrize(('view','function'),[
    ('Saved research results','load_research_results'),
    ('Training history','load_training_history'),
])
def test_absent_optional_assets_have_a_readable_message(monkeypatch,view,function):
    def missing(*a,**k):raise service.UnavailableError('Not included in this package.')
    monkeypatch.setattr(service,function,missing)
    app=page(boot(),view)
    assert not app.exception and app.info


def test_partial_bundle_can_still_predict_fedavg(monkeypatch):
    original=service.PredictionService.catalog
    def partial(self):
        return [c if c.key=='fedavg' else service.ModelChoice(c.key,c.label,None,False,'Not included.') for c in original(self)]
    monkeypatch.setattr(service.PredictionService,'catalog',partial)
    app=page(boot(),'Predict traffic')
    assert len(app.sidebar.selectbox(key='model').options)==1
    app.button(key='predict').click().run()
    assert not app.exception and len(app.dataframe[0].value)==20


def test_inference_failure_discards_old_download(monkeypatch):
    app=page(boot(),'Predict traffic')
    app.button(key='predict').click().run()
    assert app.get('download_button')
    def invalid(*a,**k):raise service.FrontendError('Invalid model contract.')
    monkeypatch.setattr(service.PredictionService,'predict',invalid)
    app.button(key='predict').click().run()
    assert not app.exception and app.error
    assert app.session_state['_prediction'] is None
    assert not app.get('download_button')
