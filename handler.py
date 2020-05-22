try:
    import unzip_requirements
except ImportError:
    pass
from model import Model
import json

model = Model('./model', 'philschmid-models',
              'sentiment_classifier/german-bert-sentiment.tar.gz')


def predict_sentiment(event, context):
    try:
        print(event['body'])
        body = json.loads(event['body'])
        prediction = model.predict_sentiment(body['text'])

        return {
            "statusCode": 200,
            "headers": {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*',
                "Access-Control-Allow-Credentials": True

            },
            "body": json.dumps({'sentiment': prediction})
        }
    except Exception as e:
        print(repr(e))
        return {
            "statusCode": 500,
            "headers": {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*',
                "Access-Control-Allow-Credentials": True
            },
            "body": json.dumps({"error": repr(e)})
        }
