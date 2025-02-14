from django.urls import path, include
from rest_framework.routers import DefaultRouter
from .views import CompareImagesViewSet  # Import the ViewSet from views.py

# Create a router and register the ViewSet
router = DefaultRouter()
router.register(r'compare-images', CompareImagesViewSet, basename='compare-images')

urlpatterns = [
    path('', include(router.urls)), 
]
