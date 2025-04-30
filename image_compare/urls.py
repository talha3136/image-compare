"""
URL configuration for image_compare project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.1/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from django.urls import include
from drf_yasg import openapi
from drf_yasg.views import get_schema_view
from rest_framework import permissions

from home.utils import server_running

sw='swagger/'
SchemaView = get_schema_view(
    openapi.Info(
        title="Image compare API",
        default_version='3.0.0',
    ),
    public=False,
    permission_classes=([permissions.IsAuthenticated])
)

urlpatterns = [
    path('', server_running),
    path('admin/', admin.site.urls),
    path('api/', include('home.urls')),
    path('api/', include('account.urls')),

    path(sw, SchemaView.with_ui('swagger', cache_timeout=0), name='schema-swagger-ui'),

]
