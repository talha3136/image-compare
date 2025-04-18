from rest_framework import serializers



class ImageUrlSerializer(serializers.Serializer):
    profileImageURL = serializers.URLField(required=True)
    targetImageURL = serializers.URLField(required=True)



class ImageSerializer(serializers.Serializer):
    profileImage = serializers.FileField()
    targetImage = serializers.FileField()


class uniformVerifyImageSerializer(serializers.Serializer):
    image = serializers.ImageField(required=True)
    text_prompt = serializers.CharField(required=False)


class uniformVerifyImageUrlSerializer(serializers.Serializer):
    imageURL = serializers.URLField(required=True)
    text_prompt = serializers.CharField(required=False)

