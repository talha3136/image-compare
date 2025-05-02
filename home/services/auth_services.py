from datetime import  timedelta
from hashlib import sha256
import os
from django.utils import timezone
from rest_framework.exceptions import AuthenticationFailed, PermissionDenied
import requests
from account.models import User
from rest_framework_simplejwt.tokens import AccessToken, RefreshToken


class AuthenticationService:
    # @staticmethod
    # def verify_recaptcha(recaptcha_token):

    #     response = requests.post(
    #         'https://www.google.com/recaptcha/api/siteverify', 
    #         {
    #             'secret': settings.RECPCHA_PRIVATE_KEY,
    #             'response': recaptcha_token,
    #         }
    #     )
    #     print(response.text)
    #     data = response.json()
        
    #     if data['success']:
    #         return True
    #     else:
    #         return False
    
    @staticmethod
    def login(email, password, is_superuser):
        user: User = User.objects.filter(email=email).first()

        if not user:
            raise AuthenticationFailed('User not found')

        if is_superuser and not user.is_superuser:
            raise AuthenticationFailed('Only superusers can login')
        if not is_superuser and user.is_superuser:
            raise AuthenticationFailed('Superusers cannot login from this portal')

        if not user.is_active:
            raise AuthenticationFailed('User not active')

        if not user.check_password(password):
            raise AuthenticationFailed('Invalid email or password')

        access_token = str(AccessToken.for_user(user))
        refresh_token = str(RefreshToken.for_user(user))

        return access_token, refresh_token, user