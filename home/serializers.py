from rest_framework import serializers



class ImageUrlSerializer(serializers.Serializer):
    profileImageURL = serializers.URLField(required=True)
    targetImageURL = serializers.URLField(required=True)



class ImageSerializer(serializers.Serializer):
    profileImage = serializers.FileField()
    targetImage = serializers.FileField()

