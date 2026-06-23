from keras._tf_keras.keras.preprocessing.text import Tokenizer
from keras._tf_keras.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from keras._tf_keras.keras.layers import Embedding, LSTM, Dense, Input  
from keras._tf_keras.keras.models import Model


with open(r'C:\Users\kenco\OneDrive\Desktop\AI Stuff\translation_ai\data\train.en', 'r', encoding='utf-8') as en_file:
    english_sentences = en_file.readlines()
with open(r'C:\Users\kenco\OneDrive\Desktop\AI Stuff\translation_ai\data\train.fr', 'r', encoding='utf-8') as fr_file:
    french_sentences = fr_file.readlines()

english_sentences = [line.strip() for line in english_sentences]
french_sentences = [line.strip() for line in french_sentences]

max_length = 50
eng_vocab_size = 50000
french_vocab_size = 50000
lstm_units = 256

en_tokenizer= Tokenizer(num_words=eng_vocab_size, filters='')
en_tokenizer.fit_on_texts(english_sentences)
en_sequences = en_tokenizer.texts_to_sequences(english_sentences)
en_padded = pad_sequences(en_sequences, maxlen=max_length, padding='post')

fr_tokenizer = Tokenizer(num_words=french_vocab_size, filters= '')
fr_tokenizer.fit_on_texts(french_sentences)
fr_sequences = fr_tokenizer.texts_to_sequences(french_sentences)
fr_padded = pad_sequences(fr_sequences, maxlen=max_length, padding='post')

x_train, x_test, y_train, y_test = train_test_split(en_padded, fr_padded, test_size=0.2, random_state=42)

embedding_dim = 256

encoder_inputs = Input(shape=(max_length,), name='encoder_inputs')
encoder_embedding = Embedding(eng_vocab_size, embedding_dim)(encoder_inputs)
encoder_outputs, state_h, state_c = LSTM(lstm_units, return_state=True)(encoder_embedding)

decoder_inputs = Input(shape=(max_length - 1,), name='decoder_inputs')
decoder_embedding = Embedding(french_vocab_size, embedding_dim)(decoder_inputs)
decoder_lstm = LSTM(lstm_units, return_sequences = True, return_state=False )
decoder_outputs = decoder_lstm(decoder_embedding, initial_state=[state_h,state_c])
decoder_dense = Dense(french_vocab_size,activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

model = Model([encoder_inputs, decoder_inputs,], decoder_outputs)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(
    [x_train, y_train[:, :-1]],
    y_train[:,1:],
    validation_data=([x_test, y_test[:,:-1]], y_test[:, 1:]),
    batch_size = 64,
    epochs=10,
)

model.save('Translation.keras')