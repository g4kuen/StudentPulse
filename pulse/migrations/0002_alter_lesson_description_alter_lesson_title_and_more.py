# Generated by Django 4.2.9 on 2024-01-13 01:16

from django.conf import settings
from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
        ('pulse', '0001_initial'),
    ]

    operations = [
        migrations.AlterField(
            model_name='lesson',
            name='description',
            field=models.TextField(verbose_name='Описание занятия'),
        ),
        migrations.AlterField(
            model_name='lesson',
            name='title',
            field=models.CharField(max_length=100, verbose_name='Название'),
        ),
        migrations.AlterField(
            model_name='lesson',
            name='user',
            field=models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to=settings.AUTH_USER_MODEL, verbose_name='Пользователь'),
        ),
        migrations.AlterField(
            model_name='review',
            name='content',
            field=models.TextField(verbose_name='текст отзыва'),
        ),
        migrations.AlterField(
            model_name='review',
            name='lesson',
            field=models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='pulse.lesson', verbose_name='Занятие'),
        ),
        migrations.AlterField(
            model_name='review',
            name='rating_criterion1',
            field=models.FloatField(verbose_name='понимание материала'),
        ),
        migrations.AlterField(
            model_name='review',
            name='rating_criterion2',
            field=models.FloatField(verbose_name='организация занятия'),
        ),
        migrations.AlterField(
            model_name='review',
            name='rating_criterion3',
            field=models.FloatField(verbose_name='полезность материала'),
        ),
        migrations.AlterField(
            model_name='review',
            name='rating_criterion4',
            field=models.FloatField(verbose_name='интересность материала'),
        ),
        migrations.AlterField(
            model_name='review',
            name='user',
            field=models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to=settings.AUTH_USER_MODEL, verbose_name='Пользователь'),
        ),
    ]
