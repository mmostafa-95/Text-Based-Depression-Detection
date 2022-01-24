import tensorflow as tf
from model.preprocess import *
from model.bert_encode import *

class Model:
    def __init__(self) :
        self.model = tf.keras.models.load_model('saved_model')
        return 


    def predict_results(self, data):
        preprocessed_data = preprocess_text(data)
        embedded_data = bert_encode_predict(preprocessed_data)
        return self.model.predict(embedded_data)


    