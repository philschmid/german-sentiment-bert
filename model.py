import boto3
import os
import tarfile
import io
import base64
import json
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig
from typing import List, Union
import re
import torch
s3 = boto3.client('s3')


class Model():
    def __init__(self, model_path: str, s3_bucket=None, file_prefix=None):
        # load model
        self.model, self.tokenizer = self.from_pretrained(
            model_path, s3_bucket, file_prefix)
        # helper functions
        self.clean_chars = re.compile(r'[^A-Za-züöäÖÜÄß ]', re.MULTILINE)
        self.clean_http_urls = re.compile(r'https*\S+', re.MULTILINE)
        self.clean_at_mentions = re.compile(r'@\S+', re.MULTILINE)

    def replace_numbers(self, text: str) -> str:
        # replace numbers 0-9 to real strings
        return text.replace("0", " null").replace("1", " eins").replace("2", " zwei").replace("3", " drei").replace("4", " vier").replace("5", " fünf").replace("6", " sechs").replace("7", " sieben").replace("8", " acht").replace("9", " neun")

    def clean_text(self, text: str) -> str:
        text = text.replace("\n", " ")
        text = self.clean_http_urls.sub('', text)
        text = self.clean_at_mentions.sub('', text)
        text = self.replace_numbers(text)
        text = self.clean_chars.sub('', text)
        text = ' '.join(text.split())
        text = text.strip().lower()
        return text

    def save_model(self, out_path: str, model_name='model'):
        self.model.save_pretrained(out_path)
        self.tokenizer.save_pretrained(out_path)
        pack_model(out_path, model_name)

    def load_model(self, model_path: str):
        if os.path.isfile(f'{model_path}/pytorch_model.bin'):
            model = AutoModelForSequenceClassification.from_pretrained(
                model_path)
            config = AutoConfig.from_pretrained(f'{model_path}/config.json')
        return model

    def load_model_from_s3(self, model_path: str, s3_bucket: str, file_prefix: str):
        if model_path and s3_bucket and file_prefix:
            obj = s3.get_object(Bucket=s3_bucket, Key=file_prefix)
            bytestream = io.BytesIO(obj['Body'].read())
            tar = tarfile.open(fileobj=bytestream, mode="r:gz")
            config = AutoConfig.from_pretrained(f'{model_path}/config.json')
            for member in tar.getmembers():
                if member.name.endswith(".bin"):
                    f = tar.extractfile(member)
                    state = torch.load(io.BytesIO(f.read()))
                    model = AutoModelForSequenceClassification.from_pretrained(
                        pretrained_model_name_or_path=None, state_dict=state, config=config)
            return model
        else:
            raise KeyError('No S3 Bucket and Key Prefix provided')

    def load_tokenizer(self, model_path: str):
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        return tokenizer

    def from_pretrained(self, model_path: str, s3_bucket: str, file_prefix: str):
        if os.path.isfile(f'{model_path}/pytorch_model.bin'):
            model = self.load_model(model_path)
        else:
            model = self.load_model_from_s3(model_path, s3_bucket, file_prefix)
        tokenizer = self.load_tokenizer(model_path)
        return model, tokenizer

    def predict_sentiment(self, texts: Union[List[str], str]) -> List[str]:
        try:
            if isinstance(texts, str):
                texts = [texts]
            texts = [self.clean_text(text) for text in texts]
          # Add special tokens takes care of adding [CLS], [SEP], <s>... tokens in the right way for each model.
            input_ids = self.tokenizer.batch_encode_plus(
                texts, pad_to_max_length=True, add_special_tokens=True)
            input_ids = torch.tensor(input_ids["input_ids"])

            with torch.no_grad():
                logits = self.model(input_ids)
            print(logits[0])
            label_ids = torch.argmax(logits[0], axis=1)

            labels = [self.model.config.id2label[label_id]
                      for label_id in label_ids.tolist()]
            if len(labels) == 1:
                return labels[0]
            return labels
        except Exception as e:
            raise(e)
