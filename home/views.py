import os
from home.services.uniform_verification_service import UniformVerificationService
from .models import DataSet, uniformChecker
from .serializers import CustomUniformVerifyImageSerializer, CustomUniformVerifyImageUrlSerializer, DataSettSerializer, GenrateDataSetFromUrlSerializer, GenrateDataSetSerializer, ImageSerializer, ImageUrlSerializer, uniformCheckerSerializer, uniformVerifyImageSerializer, uniformVerifyImageUrlSerializer
from rest_framework import viewsets, mixins, status
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework.response import Response
from rest_framework.decorators import action
from .services.services import FileRelatedService
import requests
from .services.support_services import allowed_file, compare_faces, preprocess_and_save_image, verify_uniform, save_uploaded_image, ALLOWED_EXTENSIONS
from .utils import DefaultPagination
from django.core.files.base import ContentFile
from PIL import Image



class CustomUniformCheckerViewSet(viewsets.GenericViewSet):
    parser_classes = (MultiPartParser, FormParser)

    @action(
        detail=False, 
        methods=['POST'],
        url_path='verify-uniform-with-image_v2', 
        serializer_class=CustomUniformVerifyImageSerializer
    )
    def verify_uniform_image_v2(self, request):
        file = request.FILES.get('image')
        threshold = float(request.data.get('threshold', 0.65))
        user_prompt = request.data.get('prompt', '')

        if not file:
            return Response({'error': 'No image uploaded'}, status=400)

        try:
            image = Image.open(file).convert('RGB')
        except Exception:
            return Response({'error': 'Invalid image format'}, status=400)

        result, confidence, summary = UniformVerificationService.verify_image(image, user_prompt, threshold)

        if summary == 'Failed to load model':
            return Response({'error': summary}, status=500)
        if summary == 'Error during model inference':
            return Response({'error': summary}, status=500)

        return Response({
            "result": result,
            "confidence_pct": f"{confidence * 100:.2f}",
            "summary": summary
        })
    
    @action(
        detail=False,
        methods=['POST'],
        url_path='verify-uniform-with-image-url-v2',
        serializer_class=CustomUniformVerifyImageUrlSerializer
    )
    def verify_uniform_image_url_v2(self, request):
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        image_url = serializer.validated_data['imageURL']
        threshold = serializer.validated_data.get('threshold', 0.65)
        user_prompt = serializer.validated_data.get('prompt', '')

        success, image = UniformVerificationService.open_image_from_url(image_url)
        if not success:
            return Response({'error': 'Invalid or inaccessible image URL'}, status=400)

        result, confidence, summary = UniformVerificationService.verify_image(image, user_prompt, threshold)

        if summary == 'Failed to load model':
            return Response({'error': summary}, status=500)
        if summary == 'Error during model inference':
            return Response({'error': summary}, status=500)

        return Response({
            "result": result,
            "confidence_pct": f"{confidence * 100:.2f}",
            "summary": summary
        })


    
    @action(
        detail=False, 
        methods=['post'],
        url_path='genrate-dataset',
        serializer_class=GenrateDataSetSerializer
    )
    def generate_dataset(self, request):
        serializer = GenrateDataSetSerializer(data=request.data)
        if serializer.is_valid():
            serializer.save()
            return Response({'message': 'dataset saved'}, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
    
    @action(
        detail=False, 
        methods=['POST'],
        url_path='generate-dataset-from-url',
        serializer_class=GenrateDataSetFromUrlSerializer
    )
    def generate_dataset_from_url(self, request):
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        image_url = serializer.validated_data['imageURL']
        prompt = serializer.validated_data.get('prompt', '')

        success, result = UniformVerificationService.create_dataset_from_url(image_url, prompt)

        if success:
            return Response(result, status=201)
        else:
            return Response({'error': result}, status=400 if 'image' in result.lower() else 500)

    @action(
        detail=False,
        methods=['get'],
        url_path='get-dataset',
        serializer_class=DataSettSerializer
    )
    def get_dataset(self, request):
        queryset = DataSet.objects.all()
        serializer = DataSettSerializer(queryset, many=True)
        return Response(serializer.data, status=status.HTTP_200_OK)
    
    @action(
        detail=False, 
        methods=['post'],
        url_path='train-model',
    )
    def train_model(self, request):
        success, result = UniformVerificationService.train_clip_model()

        if success:
            return Response(result, status=200)
        else:
            return Response({'message': result}, status=500)
    


class uniformCheckerViewset(viewsets.GenericViewSet,mixins.ListModelMixin,mixins.DestroyModelMixin):
    queryset = uniformChecker.objects.all()
    serializer_class = uniformCheckerSerializer
    parser_classes = (MultiPartParser, FormParser)
    pagination_class = DefaultPagination

    @action(
        detail=False, 
        methods=['POST'],
        url_path='verify-uniform-with-image', 
        serializer_class=uniformVerifyImageSerializer
    )
    def verify_uniform_image(self, request):
        """Verify security uniform from uploaded image file"""
        serializer = self.get_serializer(data=request.data)
        if not serializer.is_valid():
            return Response({'error': serializer.errors}, status=400)
        
        image = request.FILES.get('image')
        threshold = 0.65
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
            result, confidence, note, missing = verify_uniform(image_path, threshold, text_prompt)

            # Save the image to model-compatible File object
            django_file = ContentFile(image_bytes, name=image.name)

            # Save result in database
            uniform_instance = uniformChecker.objects.create(
                image=django_file,
                result=result,
                confidence=confidence,
                note=note
            )

            return Response({
                'id': str(uniform_instance.id),
                'result': result,
                'confidence_percentage': confidence,
                'note': note,
            }, status=200)

        except Exception as e:
            return Response({'error': f'Image processing error: {str(e)}'}, status=500)

    
    @action(
        detail=False, 
        methods=['POST'],
        url_path='verify-uniform-with-url', 
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

            # Extract filename
            url_path = image_url.split('?')[0]
            filename = url_path.split('/')[-1]
            if not allowed_file(filename):
                filename = "downloaded_image.jpg"

            image_bytes = response.content
            image_path = save_uploaded_image(image_bytes, filename)

            # Verify uniform
            result, confidence, note, missing = verify_uniform(image_path, threshold, text_prompt)

            # Save to model
            django_file = ContentFile(image_bytes, name=filename)

            uniform_instance = uniformChecker.objects.create(
                image=django_file,
                result=result,
                confidence=confidence,
                note=note
            )

            return Response({
                'id': str(uniform_instance.id),
                'result': result,
                'confidence_percentage': confidence,
                'note': note,
            }, status=200)

        except Exception as e:
            return Response({'error': f'Image processing error: {str(e)}'}, status=500)

        

class CompareImagesViewSet(viewsets.GenericViewSet):
    serializer_class = ImageSerializer
    parser_classes = (MultiPartParser, FormParser)

    
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