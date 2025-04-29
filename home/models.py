import uuid
from django.db import models

from home.utils import get_clip_data_set, get_uniform_checker_image
from image_compare.storage_backends import MediaStorage, PrivateMediaStorage



class uniformChecker(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False, unique=True)
    image = models.ImageField(upload_to=get_uniform_checker_image,storage=PrivateMediaStorage)
    result = models.BooleanField()
    confidence = models.FloatField()
    note = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Uniform Checker History - {self.created_at}"
    


class DataSet(models.Model):
    id = models.AutoField(primary_key=True)
    image = models.ImageField(upload_to=get_clip_data_set,storage=MediaStorage)
    prompt = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)
    class Meta:
        indexes = [
            models.Index(fields=['created_at'])
        ]


class TrainingState(models.Model):
    id = models.AutoField(primary_key=True)
    last_trained_id = models.ForeignKey(DataSet, on_delete=models.SET_NULL, null=True, blank=True)
    last_trained_time = models.DateTimeField(null=True, blank=True)
