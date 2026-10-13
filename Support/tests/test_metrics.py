import pytest
from support.evaluation.metrics import classification_metrics


def test_known_metrics():
    m=classification_metrics([0,0,1,1],[0,1,1,1],2)
    assert m['accuracy']==0.75
    assert m['balanced_accuracy']==0.75
    assert m['macro_precision']==pytest.approx(5/6)
    assert m['macro_recall']==0.75
    assert m['macro_f1']==pytest.approx((2/3+0.8)/2)
    assert m['weighted_f1']==m['macro_f1']
    assert m['confusion_matrix']==[[1,1],[0,2]]


def test_missing_class_policy():
    m=classification_metrics([0,0],[0,0],3)
    assert m['balanced_accuracy']==1
    assert m['macro_recall']==pytest.approx(1/3)
    assert m['per_class'][2]['support']==0
    assert len(m['confusion_matrix'])==3


def test_empty():
    with pytest.raises(ValueError):
        classification_metrics([],[],3)
