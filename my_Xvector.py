import os

import torchaudio
from speechbrain.pretrained import EncoderClassifier

classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-xvect-voxceleb",
                                            savedir="pretrained_models/spkrec-xvect-voxceleb")
matches = ['M', 'manipulated', 'cut']
valid_check = ['test', 'validation']

def one_to_xvec(path):
    signal, fs = torchaudio.load(path)
    embeddings = classifier.encode_batch(signal)
    embeddings = embeddings.detach().cpu().numpy()
    embedding = embeddings[0][0]
    return embedding