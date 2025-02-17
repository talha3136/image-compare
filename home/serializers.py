from rest_framework import serializers



class ImageSerializer(serializers.Serializer):
    profileImageURL = serializers.URLField(required=True)
    targetImageURL = serializers.URLField(required=True)
