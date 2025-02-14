from rest_framework import serializers



class ImageSerializer(serializers.Serializer):
    image1 = serializers.FileField()
    image2 = serializers.FileField()
