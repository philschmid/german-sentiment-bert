import os
import tarfile


def unpack_model(model_name=''):
    tar = tarfile.open(f"{model_name}.tar.gz", "r:gz")
    tar.extractall()
    tar.close()


unpack_model('german-bert-sentiment')
