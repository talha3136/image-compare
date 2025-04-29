from django.urls import path, include
from rest_framework.routers import DefaultRouter
from .views import CompareImagesViewSet, CustomUniformCheckerViewSet , uniformCheckerViewset 

# Create a router and register the ViewSet
router = DefaultRouter()
router.register(r'compare-images', CompareImagesViewSet, basename='compare-images')
router.register(r'uniform-checker', uniformCheckerViewset, basename='history')
router.register(r'custom-uniform-checker', CustomUniformCheckerViewSet, basename='uniform-checker')

urlpatterns = [
    path('', include(router.urls)), 
]
