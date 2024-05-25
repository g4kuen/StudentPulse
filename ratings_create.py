import pandas as pd
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense, Dropout
import pickle
from tqdm import tqdm

file_path = 'datasets/Reviews.csv'
df = pd.read_csv(file_path, delimiter=',')

df['текст'] = df['текст'].astype(str).fillna('')

X_columns = ['текст', 'понимание материала', 'организация занятия', 'полезность материала', 'интересность материала']
y_columns = ['понимание материала', 'организация занятия', 'полезность материала', 'интересность материала']

# Правильная предобработка текста
df['текст'] = df['текст'].apply(lambda x: x.lower())  # Приведение к нижнему регистру

# Разделение данных на обучающий и тестовый наборы
X_train, X_test, y_train, y_test = train_test_split(df[X_columns[0]], df[y_columns], test_size=0.2, random_state=42)
for aspect in y_columns:
    class_counts = y_train[aspect].value_counts()
    print(f"Class counts for aspect '{aspect}':\n{class_counts}\n")
max_words = 100000
max_sequence_length = 100
embedding_dim = 100  # Увеличим размер эмбеддинга
lstm_units = 50  # Увеличим количество нейронов в слое LSTM
dropout_rate = 0.2  # Добавим dropout для предотвращения переобучения
epochs = 20  # Увеличим количество эпох обучения

# Обновленная часть кода для обучения модели
for aspect in y_columns:
    tokenizer = Tokenizer(num_words=max_words)
    tokenizer.fit_on_texts(X_train)

    with open(f'tokenizers/tokenizer_{aspect}.pickle', 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

    X_train_seq = tokenizer.texts_to_sequences(X_train)
    X_test_seq = tokenizer.texts_to_sequences(X_test)

    X_train_pad = pad_sequences(X_train_seq, maxlen=max_sequence_length)
    X_test_pad = pad_sequences(X_test_seq, maxlen=max_sequence_length)

    model = Sequential()
    model.add(Embedding(input_dim=max_words, output_dim=embedding_dim, input_length=max_sequence_length))
    model.add(LSTM(units=lstm_units, dropout=dropout_rate))
    model.add(Dense(1, activation='linear'))

    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_squared_error'])

    class_weights = {0: 1, 1: 70, 2: 53, 3: 9, 4: 2.25, 5: 1}

    # Добавлен параметр validation_data для отслеживания метрик на валидационной выборке
    model.fit(X_train_pad, y_train[aspect], epochs=epochs, batch_size=128, validation_split=0.2,
              class_weight=class_weights, validation_data=(X_test_pad, y_test[aspect]))

    model.save(f'models/model_{aspect}.keras')
