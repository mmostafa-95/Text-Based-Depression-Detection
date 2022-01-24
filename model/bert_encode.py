import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from model.tokenization import FullTokenizer
module_url = 'https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/2'
bert_layer = hub.KerasLayer(module_url, trainable=False)



vocab_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
do_lower_case = bert_layer.resolved_object.do_lower_case.numpy()
tokenizer = FullTokenizer(vocab_file, do_lower_case)


def bert_encode_predict(text, tokenizer=tokenizer, max_len=128):
    all_tokens = []
    all_masks = []
    all_segments = []
    
    
    text = tokenizer.tokenize(text)

    text = text[:max_len-2]
    input_sequence = ["[CLS]"] + text + ["[SEP]"]
    pad_len = max_len - len(input_sequence)

    tokens = tokenizer.convert_tokens_to_ids(input_sequence) + [0] * pad_len
    pad_masks = [1] * len(input_sequence) + [0] * pad_len
    segment_ids = [0] * max_len

    all_tokens.append(tokens)
    all_masks.append(pad_masks)
    all_segments.append(segment_ids)

    return np.array(all_tokens), np.array(all_masks), np.array(all_segments)