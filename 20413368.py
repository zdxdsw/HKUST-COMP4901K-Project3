from __future__ import print_function
import argparse
import json
import os
import time
import keras
from keras.models import *
from keras.layers import Input, Dense, Embedding, Dropout, TimeDistributed
from keras.layers import *
from keras.optimizers import Adam
import numpy as np
from data_helper import load_data, build_input_data
from scorer import scoring
from utils import TestCallback, make_submission
from keras.layers import Bidirectional
from keras.layers import GRU
from keras_self_attention import SeqSelfAttention
from keras.layers import dot, concatenate
from keras.callbacks import EarlyStopping



'''
def attention_3d_block(inputs):
    #input_dim = int(inputs.shape[2])
    a = Permute((2, 1))(inputs)
    a = Dense(10, activation='softmax')(a)
    a_probs = Permute((2, 1), name='attention_vec')(a)
    #output_attention_mul = merge([inputs, a_probs], name='attention_mul', mode='mul')
    output_attention_mul = multiply([inputs, a_probs], name='attention_mul')
    return output_attention_mul

    # build RNN model with attention
    
    inputs = Input(shape=(TIME_STEPS, INPUT_DIM))
    drop1 = Dropout(0.3)(inputs)
    lstm_out = Bidirectional(LSTM(lstm_units, return_sequences=True), name='bilstm')(drop1)
    attention_mul = attention_3d_block(lstm_out)
    attention_flatten = Flatten()(attention_mul)
    drop2 = Dropout(0.3)(attention_flatten)
    output = Dense(10, activation='sigmoid')(drop2)
    model = Model(inputs=inputs, outputs=output)
   
'''
    
def build_model(embedding_dim, hidden_size, drop, sequence_length, vocabulary_size):
    inputs = Input(shape=(sequence_length,), dtype='int32')
    # inputs -> [batch_size, sequence_length]

    emb_layer = Embedding(input_dim=vocabulary_size, output_dim=embedding_dim, input_length=sequence_length)
    # emb_layer.trainable = False
    # if you uncomment this line, the embeddings will be untrainable

    embedding = emb_layer(inputs)
    # embedding -> [batch_size, sequence_length, embedding_dim]
    dense_1 = Dense(units = hidden_size, activation = 'tanh')(embedding)
    drop_1 = Dropout(drop)(dense_1)
    # dropout at embedding layer

    # add a LSTM here, set units=hidden_size, dropout=drop, recurrent_dropout = drop, return_sequences=True
    # please read https://keras.io/layers/recurrent/
    lstm_out_1 = LSTM(units=hidden_size, dropout = drop, recurrent_dropout = drop, kernel_regularizer=keras.regularizers.l2(8e-6),
                      return_sequences=True)(drop_1)
    # lstm_out_1 -> [batch_size, sequence_length, hidden_size]
    # drop_1 = Dropout(drop)(lstm_out_1)
    #att = SeqSelfAttention(attention_width=10, attention_type=SeqSelfAttention.ATTENTION_TYPE_MUL, attention_activation='sigmoid',
     #                      kernel_regularizer=keras.regularizers.l2(12e-6), use_attention_bias=False, name='Attention')(lstm_out_1)

    lstm_out_2 = LSTM(units=hidden_size, dropout = drop, recurrent_dropout = drop, kernel_regularizer=keras.regularizers.l2(8e-6),
                     return_sequences=True)(concatenate([embedding, lstm_out_1], axis=2))
    #lstm_out_3 = LSTM(units=hidden_size, dropout = drop, recurrent_dropout = drop, kernel_regularizer=keras.regularizers.l2(10e-6),
     #                 return_sequences=True)(concatenate([embedding, lstm_out_2], axis=2))
    # lstm_out_1 -> [batch_size, sequence_length, hidden_size]
    # lstm_out_3 = LSTM(units=hidden_size, dropout = drop, recurrent_dropout = drop, return_sequences = True)(lstm_out_2)
    # add a TimeDistributed here, set units=hidden_size, dropout=drop, recurrent_dropout = drop, return_sequences=True
    # please read  https://keras.io/layers/wrappers/
    # output: outputs -> [batch_size, sequence_length, vocabulary_size]
    outputs = TimeDistributed(Dense(units=vocabulary_size, activation='softmax'))(concatenate([embedding, lstm_out_1, lstm_out_2], axis=2))

    # End of Model Architecture
    # ----------------------------------------#

    model = Model(inputs=inputs, outputs=outputs)

    adam = Adam()
    model.compile(loss='sparse_categorical_crossentropy', optimizer=adam)

    print(model.summary())
    return model


def predict_final_word(model, vocabulary, filename):
    id_list = []
    prev_tokens_list = []
    prev_tokens_lens = []
    with open(filename, "r") as fin:
        fin.readline()
        for line in fin:
            id_, prev_sent, grt_last_token = line.strip().split(",")
            id_list.append(id_)
            prev_tokens = prev_sent.split()
            prev_tokens_list.append(prev_tokens)
            prev_tokens_lens.append(len(prev_tokens))
    X = np.array([build_input_data(t, vocabulary)[0][0].tolist()
                  for t in prev_tokens_list])
    y_prob = model.predict(X, batch_size=32)
    last_token_probs = np.array([y_prob[b, prev_tokens_lens[b] - 1, :]
                                 for b in range(y_prob.shape[0])])

    return dict(zip(id_list, last_token_probs))


def main(opt):
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu
    if opt.mode == "train":
        st = time.time()
        print('Loading data')
        x_train, y_train, x_valid, y_valid, vocabulary_size = load_data(
            "data", opt.debug)
        print('\nx_train.shape = ', x_train.shape)
        print('\ny_train.shape = ', y_train.shape)
        

        num_training_data = x_train.shape[0]
        sequence_length = x_train.shape[1]
        print(num_training_data)

        print('Vocab Size', vocabulary_size)

        model = build_model(opt.embedding_dim, opt.hidden_size, opt.drop, sequence_length, vocabulary_size)
        print("Traning Model...")
        

        callback = [EarlyStopping(monitor='val_loss', patience=20, verbose=1), TestCallback((x_valid,y_valid), model=model)]
        history = model.fit(x_train, y_train, validation_data=(x_valid, y_valid), batch_size=opt.batch_size,
                            epochs=opt.epochs, verbose=1,
                            callbacks=callback)
        model.save(opt.saved_model)
        #callbacks=[early_stopping]
        print("Training cost time: ", time.time() - st)

    else:
        model = keras.models.load_model(opt.saved_model, custom_objects=SeqSelfAttention.get_custom_objects())
        #model = load_model(opt.saved_model)
        vocabulary = json.load(open(os.path.join("data", "vocab.json")))
        predict_dict = predict_final_word(model, vocabulary, opt.input)
        sub_file = make_submission(predict_dict, opt.student_id, opt.input)
        if opt.score:
            scoring(sub_file, os.path.join("data"), type="valid")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-mode", default="train", choices=["train", "test"],
                        help="Train or test mode")
    parser.add_argument("-saved_model", type=str, default="model.h5",
                        help="saved model path")
    parser.add_argument("-input", type=str, default=os.path.join("data", "valid.csv"),
                        help="Input path for generating submission")
    parser.add_argument("-debug", action="store_true",
                        help="Use validation data as training data if it is true")
    parser.add_argument("-score", action="store_true",
                        help="Report score if it is")
    parser.add_argument("-student_id", default=None, required=True,
                        help="Student id number is compulsory!")

    parser.add_argument("-epochs", type=int, default=1,
                        help="training epoch num")
    parser.add_argument("-batch_size", type=int, default=32,
                        help="training batch size")
    parser.add_argument("-embedding_dim", type=int, default=100,
                        help="word embedding dimension")
    parser.add_argument("-hidden_size", type=int, default=500,
                        help="rnn hidden size")
    parser.add_argument("-drop", type=float, default=0.5,
                        help="dropout")
    parser.add_argument("-gpu", type=str, default="",
                        help="dropout")
    opt = parser.parse_args()
    main(opt)
