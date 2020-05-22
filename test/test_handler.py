import pytest
from handler import predict_sentiment
import json

test_events = {
    "body": '{"text": "Der Aktienkurs für Puma ist sehr gut."}'
}


def test_handler():
    res = predict_sentiment(test_events, '')
    assert json.loads(res['body']) == {'sentiment': 'positive'}
