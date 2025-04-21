import uuid
from django.db import models

from home.utils import get_uniform_checker_image
from image_compare.storage_backends import PrivateMediaStorage

# Create your models here.


class uniformChecker(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False, unique=True)
    image = models.ImageField(upload_to=get_uniform_checker_image,storage=PrivateMediaStorage)
    result = models.BooleanField()
    confidence = models.FloatField()
    note = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Uniform Checker History - {self.created_at}"