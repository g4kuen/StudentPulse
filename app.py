from ratings_test import get_ratings
from mat_test import get_mat
from spam_test import get_spam
input_text = 'Im delighted that this course gave me the confidence to learn more about python. Now python seems less intimidating'
#	A bit slow to the real subject at first with the two first chapter. But this is a great course to begin python with.,
from keras.models import load_model



print('Cпам:')
print(get_spam(input_text))
print('Нецензурная лексика:')
print(get_mat(input_text))
print('Оценки:')
print(get_ratings(input_text))