from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional, BatchNormalization
from keras.optimizers import SGD
from keras.regularizers import l1, l2
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.callbacks import LearningRateScheduler, EarlyStopping
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
from sklearn.model_selection import GridSearchCV
# Загрузка данных
df = pd.read_csv('datasets/distortion_train.tsv', sep='\t')

class_counts = df['target'].value_counts()
print("Количество объектов каждого класса:")
print(class_counts)
total_samples = len(df)
class_ratios = class_counts / total_samples
print("Точное соотношение классов:")
print(class_ratios)

# Преобразование целевой переменной
label_encoder = LabelEncoder()
df['target'] = label_encoder.fit_transform(df['target'])

# Удаление индексов из текстовых данных
df['text'] = df['text'].replace(r'\b\d+\b', '', regex=True)

# Разделение данных
train_data, test_data, train_labels, test_labels = train_test_split(
    df['text'], df['target'], test_size=0.2, stratify=df['target'], random_state=42
)
# Replace NaN values with an empty string
train_data = train_data.fillna('')
test_data = test_data.fillna('')

# Токенизация текста
max_words = 10000
tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(train_data)

# Преобразование в последовательности и дополнение нулями
train_sequences = tokenizer.texts_to_sequences(train_data)
test_sequences = tokenizer.texts_to_sequences(test_data)

max_sequence_length = 100
train_data_pad = pad_sequences(train_sequences, maxlen=max_sequence_length)
test_data_pad = pad_sequences(test_sequences, maxlen=max_sequence_length)

# Создание модели
embedding_dim =50
model = Sequential()
model.add(Embedding(input_dim=max_words, output_dim=embedding_dim, input_length=max_sequence_length))
model.add(Bidirectional(LSTM(units=50, return_sequences=True, kernel_regularizer=l1(0.001), recurrent_regularizer=l2(0.001))))
model.add(Bidirectional(LSTM(units=50, kernel_regularizer=l2(0.001))))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(units=1, activation='sigmoid'))

# Компиляция модели с использованием оптимизатора SGD
model.compile(optimizer='sgd', loss='binary_crossentropy', metrics=['accuracy'])

# Learning Rate Schedule
def lr_scheduler(epoch):
    if epoch < 4:
        return 0.01
    else:
        if epoch < 10:
            return 0.0001
        else:
            return 0.000001

lr_schedule = LearningRateScheduler(lr_scheduler)

# Обучение модели с использованием весов классов
class_weights = {0: 1, 1: 1}

early_stopping = EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True)

model.fit(train_data_pad, train_labels, epochs=25, batch_size=64, validation_split=0.2, class_weight=class_weights, callbacks=[lr_schedule, early_stopping])

# Оценка модели на тестовых данных
test_loss, test_accuracy = model.evaluate(test_data_pad)
print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}')

# Предсказания модели
predictions = model.predict(test_data_pad)
binary_predictions = (predictions > 0.5).astype(int)

# Матрица ошибок и отчет о классификации
print(confusion_matrix(test_labels, binary_predictions))
print(classification_report(test_labels, binary_predictions))

# Сохранение модели
model.save('models/updated_model_bidirectional_lstm_sgd.keras')
