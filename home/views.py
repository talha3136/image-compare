import os
from home.serializers import ImageSerializer, ImageUrlSerializer, uniformVerifyImageSerializer, uniformVerifyImageUrlSerializer
from rest_framework import viewsets
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework.response import Response
from rest_framework.decorators import action
from home.services import FileRelatedService
import requests
from home.support_services import allowed_file, compare_faces, preprocess_and_save_image,verify_uniform,save_uploaded_image



class CompareImagesViewSet(viewsets.GenericViewSet):
    serializer_class = ImageSerializer
    parser_classes = (MultiPartParser, FormParser)

    @action(
        detail=False, 
        methods=['POST'],
        url_path='verify', 
        serializer_class=uniformVerifyImageSerializer
    )
    def verify_uniform_image(self, request):
        """Verify security uniform from uploaded image file"""
        serializer = self.get_serializer(data=request.data)
        if not serializer.is_valid():
            return Response({'error': serializer.errors}, status=400)
        
        image = request.FILES.get('image')
        threshold =  0.65
        text_prompt = request.data.get('text_prompt')
        
        if not image:
            return Response({'error': 'No image file provided'}, status=400)
        
        if not allowed_file(image.name):
            return Response({'error': f'Invalid file type. Allowed: {", ".join(ALLOWED_EXTENSIONS)}'}, status=400)
        
        try:
            # Save and process the image
            image_bytes = image.read()
            image_path = save_uploaded_image(image_bytes, image.name)
            
            # Verify uniform
            result, confidence, note, required, optional, missing = verify_uniform(image_path, threshold,text_prompt)
            
            return Response({
                'result': result,
                'confidence_percentage': confidence,
                'note': note,
                # 'required_elements': required,
                # 'optional_elements': optional,
                # 'missing_elements': missing
            }, status=200)
            
        except Exception as e:
            return Response({'error': f'Image processing error: {str(e)}'}, status=500)
    
    @action(
        detail=False, 
        methods=['POST'],
        url_path='verify-url', 
        serializer_class=uniformVerifyImageUrlSerializer
    )
    def verify_uniform_url(self, request):
        """Verify security uniform from image URL"""
        serializer = self.get_serializer(data=request.data)
        if not serializer.is_valid():
            return Response({'error': serializer.errors}, status=400)
        
        image_url = serializer.validated_data.get('imageURL')
        threshold = 0.65
        text_prompt = request.data.get('text_prompt')

        
        if not image_url:
            return Response({'error': 'No image URL provided'}, status=400)
        
        try:

            response = requests.get(image_url, timeout=10)
            if response.status_code != 200:
                return Response({'error': 'Failed to download image from URL'}, status=400)
            
            # Extract filename from URL
            url_path = image_url.split('?')[0]  
            filename = url_path.split('/')[-1]
            if not allowed_file(filename):
                filename = "downloaded_image.jpg" 
            
            # Save and process the image
            image_path = save_uploaded_image(response.content, filename)
            
            # Verify uniform
            result, confidence, note, required, optional, missing = verify_uniform(image_path, threshold,text_prompt)
            
            return Response({
                'result': result,
                'confidence_percentage': confidence,
                'note': note,
                # 'required_elements': required,
                # 'optional_elements': optional,
                # 'missing_elements': missing
            }, status=200)
            
        except Exception as e:
            return Response({'error': f'Image processing error: {str(e)}'}, status=500)
    
    @action(
        detail=False, 
        methods=['POST'],
        url_path='compare', 
        serializer_class=ImageUrlSerializer
    )
    def compare(self, request):
        profileImageURL = request.data.get('profileImageURL')
        targetImageURL = request.data.get('targetImageURL')
        profileImagePath = FileRelatedService.convert_url_to_file(profileImageURL)
        targetImagePath = FileRelatedService.convert_url_to_file(targetImageURL)
        if profileImagePath and targetImagePath:
            try:
                with open(profileImagePath, 'rb') as profileImage:
                    image1_bytes = profileImage.read()
                with open(targetImagePath, 'rb') as targetImage:
                    image2_bytes = targetImage.read()
                image1_face_path = preprocess_and_save_image(image1_bytes, os.path.basename(profileImagePath))
                image2_face_path = preprocess_and_save_image(image2_bytes, os.path.basename(targetImagePath))
                if not image1_face_path or not image2_face_path:
                    return Response({'result': False, 'confidence_percentage': 0.0, 'note': 'Face detection failed.'}, status=200)
                result, confidence, note = compare_faces(image1_face_path, image2_face_path)
                return Response({'result': result, 'confidence_percentage': confidence, 'note': note}, status=200)
            finally:
                # Ensure uploaded images are removed after processing
                if os.path.exists(profileImagePath):
                    os.remove(profileImagePath)
                if os.path.exists(targetImagePath):
                    os.remove(targetImagePath)
        return Response({'error': 'Unable to retrieve images from URLs'}, status=400)
    
    @action(
        detail=False, 
        methods=['POST'], 
        url_path='compare-image', 
        serializer_class=ImageSerializer
    )
    def compare_image_file(self, request):
        if 'profileImage' not in request.FILES or 'targetImage' not in request.FILES:
            return Response({'error': 'Please upload two images'}, status=400)
        profileImage = request.FILES['profileImage']
        targetImage = request.FILES['targetImage']
        if not (allowed_file(targetImage.name) and allowed_file(profileImage.name)):
            return Response({'error': 'Invalid file type. Allowed: png, jpg, jpeg, gif'}, status=400)
        image1_bytes = profileImage.read()
        image2_bytes = targetImage.read()
        image1_face_path = preprocess_and_save_image(image1_bytes, profileImage.name)
        image2_face_path = preprocess_and_save_image(image2_bytes, targetImage.name)
        try:
            if not image1_face_path or not image2_face_path:
                return Response({'result': False, 'confidence_percentage': 0.0, 'note': 'Face detection failed.'}, status=200)
            
            result, confidence, note = compare_faces(image1_face_path, image2_face_path)
            return Response({'result': result, 'confidence_percentage': confidence, 'note': note}, status=200)
        finally:
            # Ensure uploaded images are removed after processing
            if os.path.exists(image1_face_path):
                os.remove(image1_face_path)
            if os.path.exists(image2_face_path):
                os.remove(image2_face_path)