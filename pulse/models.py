from django.db import models
from django.contrib.auth.models import User
from django.db.models import Avg
from django.urls import reverse


class Lesson(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE,verbose_name="Пользователь")
    title = models.CharField(max_length=100,verbose_name="Название")
    description = models.TextField(verbose_name="Описание занятия")

    def __str__(self):
        return self.title

    def get_absolute_url(self):
        return reverse('lesson_detail', args=[str(self.id)])

    def avg_rating_criterion1(self):
        avg_rating = self.review_set.aggregate(avg_rating_criterion1=Avg('rating_criterion1'))['avg_rating_criterion1']
        return round(avg_rating, 3) if avg_rating is not None else None

    def avg_rating_criterion2(self):
        avg_rating = self.review_set.aggregate(avg_rating_criterion2=Avg('rating_criterion2'))['avg_rating_criterion2']
        return round(avg_rating, 3) if avg_rating is not None else None

    def avg_rating_criterion3(self):
        avg_rating = self.review_set.aggregate(avg_rating_criterion3=Avg('rating_criterion3'))['avg_rating_criterion3']
        return round(avg_rating, 3) if avg_rating is not None else None

    def avg_rating_criterion4(self):
        avg_rating = self.review_set.aggregate(avg_rating_criterion4=Avg('rating_criterion4'))['avg_rating_criterion4']
        return round(avg_rating, 3) if avg_rating is not None else None


class Review(models.Model):
    lesson = models.ForeignKey(Lesson, on_delete=models.CASCADE,verbose_name="Занятие")
    user = models.ForeignKey(User, on_delete=models.CASCADE,verbose_name="Пользователь")
    content = models.TextField(verbose_name="текст отзыва")

    rating_criterion1 = models.FloatField(verbose_name='понимание материала')
    rating_criterion2 = models.FloatField(verbose_name='организация занятия')
    rating_criterion3 = models.FloatField(verbose_name='полезность материала')
    rating_criterion4 = models.FloatField(verbose_name='интересность материала')

    def __str__(self):
        return f"Отзыв от пользователя с логином: {self.user.username}. для занятия с названием: {self.lesson.title}"
