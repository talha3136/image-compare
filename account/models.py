
from datetime import time, timedelta
from django.db import models
from django.utils import timezone
from django.utils.translation import gettext_lazy as _
from django.contrib.auth.models import AbstractBaseUser, PermissionsMixin, BaseUserManager
from rest_framework_simplejwt.tokens import RefreshToken
import uuid
from home.utils import get_team_member_profile
from image_compare.storage_backends import MediaStorage


# Create your models here.

class CustomAccountManager(BaseUserManager):
    def create_superuser(self, email, password, first_name, **other_fields):
        other_fields.setdefault('is_staff', True)
        other_fields.setdefault('is_superuser', True)
        other_fields.setdefault('is_active', True)
        if other_fields.get('is_staff') is not True:
            raise ValueError('Superuser must be assigned to is_staff=True.')
        if other_fields.get('is_superuser') is not True:
            raise ValueError('Superuser must be assigned to is_superuser=True.')
        return self.create_user(email, password, first_name, **other_fields)

    def create_user(self, email, password, first_name, **other_fields):
        if not email:
            raise ValueError(_('You must provide an email address'))

        email = self.normalize_email(email)
        user = self.model(email=email, first_name=first_name, **other_fields)
        user.set_password(password)
        user.save()
        return user


class User(AbstractBaseUser, PermissionsMixin ):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False, unique=True)
    email = models.EmailField(_('email address'), unique=True,  error_messages={'unique': "User with this Email already registered."})
    first_name = models.CharField(max_length=150, blank=True)
    last_name = models.CharField(max_length=150, blank=True, null =True)
    is_staff = models.BooleanField(default=False)
    is_active = models.BooleanField(default=True)
    jobPosition= models.CharField(max_length=150, null=True, blank=True)
    department= models.CharField(max_length=150, null=True, blank=True)
    mobileNo = models.CharField(max_length=18, null=True, blank=True)
    phoneNo = models.CharField(max_length=18, null=True, blank=True)
    profileImage =models.FileField(
        upload_to=get_team_member_profile, 
        storage=MediaStorage(),
        null=True,
        blank=True,max_length=255
    )
    objects = CustomAccountManager()
    USERNAME_FIELD = 'email'
    REQUIRED_FIELDS = ['first_name']

    def __str__(self):
        return self.first_name
    def has_group(self,  groupName):
        return self.groups.filter(name=groupName).exists()
    @property
    def token(self):
        refresh = RefreshToken.for_user(self)
        return {
            'refresh': str(refresh),
            'access': str(refresh.access_token),
        }