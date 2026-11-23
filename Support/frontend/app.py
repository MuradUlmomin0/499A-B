"""Local demonstration of saved PRIVATE SHIELD models and research records.

Run from the project: python -m streamlit run Support/frontend/app.py
    --server.address 127.0.0.1 --server.port 8501
"""
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))

import streamlit as st

from Support.frontend import service


@st.cache_resource(max_entries=1)
def prediction_service():
    """Share one CPU model cache, with inference locked inside the service."""
    return service.PredictionService(ROOT)


def overview(api):
    st.header('Intrusion detection, explained simply')
    st.write('Give a saved model measurements from network traffic. It predicts a traffic category, '
             'such as benign traffic or an attack type.')
    try:
        summary=api.demo_summary()
    except service.FrontendError as exc:
        st.error(str(exc));return
    columns=st.columns(3)
    for column,label,key in zip(columns,['Demo records','Input features','Traffic categories'],['records','features','classes']):
        column.metric(label,summary[key])
    sizes=' → '.join(map(str,[summary['features'],*summary['hidden_dims'],summary['classes']]))
    st.subheader('One network, four ways to combine client learning')
    st.write(f'The classifier is an MLP (a small neural network): **{sizes}**. '
             'The research uses ten simulated clients. Their local models are combined into a global model.')
    st.markdown('''| Method | How it combines client learning |
| :--- | :--- |
| FedAvg | Gives clients weight according to their training sample counts. |
| FedMADE (adapted) | Uses class prediction patterns to estimate client usefulness. |
| Temporal trust | Uses changes over rounds to adjust client trust. |
| Fusion | Blends temporal weights and class-utility weights equally. |''')
    st.write('Use **Predict traffic** for saved-model inference, **Saved research results** for the '
             'completed comparisons, and **Training history** for recorded validation curves.')
    st.info('This is a local demonstration using prepared traffic features. The clients are simulated; '
            'the CSV data cannot establish independence between physical devices or captures.')
    if summary['testing_only']:st.warning('This package contains testing-only synthetic demo inputs.')


def predict(api,key):
    st.header('Predict traffic')
    st.caption('Input: the bundled 20 prepared, validation-derived traffic records. '
               'Output: a predicted traffic category for each record.')
    if key is None:
        st.info('No saved prediction model is included in this package.');return
    if st.button('Run prediction',key='predict',type='primary'):
        st.session_state['_prediction']=None
        try:
            with st.spinner('Classifying the bundled records on CPU…'):
                st.session_state['_prediction']=api.predict(key)
        except service.FrontendError as exc:st.error(str(exc))
    view=st.session_state.get('_prediction')
    if view is None:return
    if view.testing_only:st.warning('Testing-only model or synthetic input: these predictions are framework checks.')
    st.success(f'{view.model_label}: {len(view.frame)} records classified.')
    st.caption('Model score is the predicted category’s softmax score, not calibrated confidence. '
               'These unlabeled demo records do not provide a detection accuracy.')
    display=view.frame[['row_index','class_id','class_name','model_score']].rename(columns={
        'row_index':'Record','class_id':'Class ID','class_name':'Predicted category','model_score':'Model score'})
    st.dataframe(display,hide_index=True,width='stretch',column_config={
        'Model score':st.column_config.NumberColumn(format='%.4f')})
    st.download_button('Download predictions CSV',data=view.csv_bytes(),
        file_name=f'private_shield_{view.model_key}_predictions.csv',mime='text/csv',key='download_predictions')
    st.caption('The CSV includes every category score. probability_0, probability_1, … follow the class-ID mapping below.')
    if st.checkbox('Show prepared input features',key='show_features'):
        st.dataframe(view.feature_preview,hide_index=True,width='stretch')
    with st.expander('Class ID mapping and saved model'):
        st.dataframe({'Class ID':list(range(len(view.class_names))),'Category':list(view.class_names)},hide_index=True)
        st.text(f'Saved experiment: {view.model_name}\nModel SHA-256: {view.artifact_sha256}')


def results():
    st.header('Saved research results')
    st.caption('Previously completed final-test evaluation; means across three seeds (42, 43, 44). '
               'These results are separate from the 20-record prediction demo.')
    condition=st.radio('Client partition',['iid','dirichlet'],key='condition',horizontal=True,
        format_func=lambda c:'IID: similar client distributions' if c=='iid' else 'Non-IID: uneven label distributions')
    try:frame=service.load_research_results(ROOT,condition)
    except service.UnavailableError as exc:st.info(str(exc));return
    except service.FrontendError as exc:st.error(str(exc));return
    st.dataframe(frame.drop(columns='method'),hide_index=True,width='stretch',column_config={
        col:st.column_config.NumberColumn(format='%.2f' if '(%)' in col or '(pp)' in col else '%.4f')
        for col in frame if col not in ('method','Model')})
    st.caption('Accuracy is a percentage. Macro F1 gives each class equal importance (0–1). '
               'Target recall drop is in percentage points (pp); lower is better, and a negative value means recall increased.')
    st.bar_chart(frame.set_index('Model')[['Target recall drop (pp)']],y_label='Target recall drop (pp)')
    st.write('The attack changed DoS-HTTP_Flood training labels to BenignTraffic on two of ten clients '
             'in rounds 4–20, after three honest rounds. Holdout labels were unchanged.')
    st.info('In these recorded experiments, fusion reduced mean target-recall damage compared with FedAvg '
            'and temporal trust. Adapted FedMADE had less damage than fusion. Some rare classes still '
            'had zero recall, so these results do not establish reliable detection of every attack.')


def history(key):
    st.header('Training history')
    st.caption('Saved validation metrics from 20 completed training rounds. The dashboard is reading '
               'an existing history; no training is running here.')
    if key is None:st.info('No saved model history is available.');return
    try:frame=service.load_training_history(key,ROOT)
    except service.UnavailableError as exc:st.info(str(exc));return
    except service.FrontendError as exc:st.error(str(exc));return
    accuracy=frame.set_index('round_id')[['accuracy']].mul(100).rename(columns={'accuracy':'Validation accuracy (%)'})
    f1=frame.set_index('round_id')[['macro_f1']].rename(columns={'macro_f1':'Validation macro F1'})
    left,right=st.columns(2)
    with left:st.line_chart(accuracy,x_label='Communication round',y_label='Validation accuracy (%)')
    with right:st.line_chart(f1,x_label='Communication round',y_label='Validation macro F1 (0–1)')
    with st.expander('View original validation metrics'):
        st.dataframe(frame,hide_index=True,width='stretch')


def main():
    st.set_page_config(page_title='PRIVATE SHIELD',page_icon='🛡️',layout='wide')
    st.title('PRIVATE SHIELD')
    st.caption('Local federated intrusion-detection demonstration')
    api=prediction_service();catalog=api.catalog();available=[c.key for c in catalog if c.available]
    with st.sidebar:
        st.header('PRIVATE SHIELD')
        page=st.radio('Navigate',['Overview','Predict traffic','Saved research results','Training history'],key='page')
        if st.session_state.get('model') not in available:st.session_state['model']=available[0] if available else None
        key=st.selectbox('Saved model',available,key='model',format_func=lambda k:service.MODELS[k][0],disabled=not available)
        st.caption('Clean IID · seed 42 · final-round model')
        absent=[c.label for c in catalog if not c.available]
        if absent:st.caption('Not included: '+', '.join(absent))
        st.divider()
        st.caption('CPU inference · up to 20 records · local browser')
    if st.session_state.get('_prediction_model')!=key:
        st.session_state['_prediction']=None;st.session_state['_prediction_model']=key
    if page=='Overview':overview(api)
    elif page=='Predict traffic':predict(api,key)
    elif page=='Saved research results':results()
    else:history(key)


if __name__=='__main__':main()
