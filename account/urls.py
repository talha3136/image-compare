from . import views
from django.urls import path, include
from rest_framework_simplejwt.views import TokenRefreshView
from rest_framework import routers

router = routers.DefaultRouter()
router.register('auth', views.AuthViewSet, basename='auth')
router.register('user', views.UserViewSet, basename='user')
router.register('get_user_from_token',views.UserTokenViewSet, basename='user_token')


urlpatterns=router.urls
urlpatterns = [
    path('', include(router.urls)),

    path('token/refresh', TokenRefreshView.as_view(), name='token_obtain_pair'),


]