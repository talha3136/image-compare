from django.urls import path, include
from rest_framework.routers import DefaultRouter
from .views import CompareImagesViewSet , uniformCheckerViewset 

# Create a router and register the ViewSet
router = DefaultRouter()
router.register(r'compare-images', CompareImagesViewSet, basename='compare-images')
router.register(r'uniform-checker', uniformCheckerViewset, basename='history')

urlpatterns = [
    path('', include(router.urls)), 
]
