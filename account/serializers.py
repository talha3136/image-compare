from uuid import UUID
from django.shortcuts import get_object_or_404
from rest_framework.serializers import ModelSerializer, ValidationError, Serializer
from rest_framework import serializers
from .models import  User




class LoginSerializer(serializers.Serializer):
    email = serializers.EmailField()
    password = serializers.CharField()
    # recaptchaToke = serializers.CharField(required = False)

class logoutSerializer(serializers.Serializer):
    refresh = serializers.CharField(max_length = 400)

class ChangeTeamMemberPasswordSerializer(serializers.Serializer):
    password = serializers.CharField(required=True, min_length=6)
    confirm_password  = serializers.CharField(required=True, min_length=6)
    user = serializers.CharField(max_length = 36, required=True)
    def validate(self, attrs):
        if attrs['password'] != attrs['confirm_password']:
            raise ValidationError('Passwords do not match')
        return attrs
    
class ChangeSelfPasswordSerializer(serializers.Serializer):
    old_password = serializers.CharField(required=False, min_length=6)
    password = serializers.CharField(required=True, min_length=6)
    confirm_password  = serializers.CharField(required=True, min_length=6)
    user = serializers.CharField(max_length = 36, required=True)
    def validate(self, attrs):
        if attrs['password'] != attrs['confirm_password']:
            raise ValidationError('Passwords do not match')
        user = get_object_or_404(User, pk=attrs['user'])
        old_password = attrs['old_password']
        if not user.check_password(old_password):
            raise ValidationError('Old password is incorrect')
        return attrs

class SuperUserSerializer(ModelSerializer):
    class Meta:
        model = User
        fields = ['first_name', 'last_name', 'email', 'phoneNo','password', 'profileImage','is_superuser','is_staff','is_active']

    def create(self, validated_data):
        password = validated_data.pop('password', None)
        if password and len(password) < 6:
            raise ValidationError({"status":False,"data": "Password must be at least 6 characters long."})
        
        instance = self.Meta.model(**validated_data)
        if password is not None:
            instance.set_password(password)
        instance.save()
        return instance

    def to_representation(self, instance):  
        # Exclude fields from the serialized data
        excluded_fields = ['password', 'groups', 'user_permissions']
        data = super().to_representation(instance)
        for field in excluded_fields:
            data.pop(field, None)
        return data
    


class UserSerializer(ModelSerializer):
    class Meta:
        model = User
        fields = '__all__'
        read_only_fields = (
            'groups',
            'user_permissions'
        )

    def create(self, validated_data):
        password = validated_data.pop('password', None)
        if password and len(password) < 8:
            raise ValidationError({"status":False,"data": "Password must be at least 8 characters long."})

        instance = self.Meta.model(**validated_data)
        if password is not None:
            instance.set_password(password)
        instance.save()
        return instance

    def to_representation(self, instance):  
        # Exclude fields from the serialized data
        excluded_fields = ['password', 'groups', 'user_permissions']
        data = super().to_representation(instance)
        for field in excluded_fields:
            data.pop(field, None)
        return data
    


class USerShortInfoSerializer(serializers.ModelSerializer):
    class Meta:
        model = User
        fields = ('id', 'first_name', 'last_name', 'profileImage', 'role', 'email', 'phoneNo')