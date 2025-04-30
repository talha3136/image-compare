from rest_framework import serializers
from .models import DataSet, uniformChecker


class EmptySerializer(serializers.Serializer):
    pass

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

class uniformCheckerSerializer(serializers.ModelSerializer):
    class Meta:
        model = uniformChecker
        fields = '__all__'

        
class CustomUniformVerifyImageSerializer(serializers.Serializer):
    image = serializers.ImageField(required=True)
    threshold = serializers.FloatField(required=False)
    prompt = serializers.CharField(required=False)


class CustomUniformVerifyImageUrlSerializer(serializers.Serializer):
    imageURL = serializers.URLField(required=True)
    threshold = serializers.FloatField(required=False)
    prompt = serializers.CharField(required=False)


class DataSettSerializer(serializers.ModelSerializer):
    class Meta:
        model = DataSet
        fields = '__all__'


class GenrateDataSetSerializer(serializers.Serializer):
    image = serializers.ImageField(required=True)
    prompt = serializers.CharField(required=False)

    def create(self, validated_data):
        return DataSet.objects.create(**validated_data)
    
class GenrateDataSetFromUrlSerializer(serializers.Serializer):
    imageURL = serializers.URLField(required=True)
    prompt = serializers.CharField(required=False)