import shutil
import tempfile
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import io
import cv2
import dlib
from home.serializers import ImageSerializer, ImageUrlSerializer
import numpy as np
from PIL import Image
from datetime import datetime
from deepface import DeepFace
from django.http import JsonResponse
from rest_framework import viewsets
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework.response import Response
from rest_framework.decorators import action
from drf_yasg.utils import swagger_auto_schema
from drf_yasg import openapi

from home.services import FileRelatedService

# Load dlib face detector and shape predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'webp'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def is_blurry(image):
    """
    Check if the image is blurry by calculating the variance of the Laplacian.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    return variance < 100

def enhance_image(image):
    """
    Enhance blurry and low-quality images.
    """
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    contrast_enhanced = cv2.cvtColor(lab, cv2.COLOR_Lab2BGR)
    sharpen_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened = cv2.filter2D(contrast_enhanced, -1, sharpen_kernel)
    denoised = cv2.bilateralFilter(sharpened, d=9, sigmaColor=50, sigmaSpace=50)
    return denoised

def preprocess_and_save_image(image_bytes, filename):
    try:
        image = Image.open(io.BytesIO(image_bytes))
        image_np = np.array(image)
        image_cv = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

        if is_blurry(image_cv):
            image_cv = enhance_image(image_cv)

        temp_dir = tempfile.mkdtemp()
        output_filename = f"processed_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_{filename}"
        output_filepath = os.path.join(temp_dir, output_filename)
        cv2.imwrite(output_filepath, image_cv)
        return output_filepath, temp_dir
    except Exception as e:
        return None, None

def compare_faces(image1_path, image2_path):
    try:
        result = DeepFace.verify(img1_path=image1_path, img2_path=image2_path, model_name="ArcFace", detector_backend="retinaface")
        return result['verified'], f"{(1 - result['distance']) * 100:.2f}"
    except Exception as e:
        return False, "0.0"

class CompareImagesViewSet(viewsets.GenericViewSet):
    serializer_class = ImageSerializer
    parser_classes = (MultiPartParser, FormParser)

    @action(detail=False, methods=['POST'], url_path='compare', serializer_class=ImageUrlSerializer)
    def compare(self, request):
        profileImageURL = request.data.get('profileImageURL')
        targetImageURL = request.data.get('targetImageURL')

        profileImage = FileRelatedService.convert_url_to_file(profileImageURL)
        targetImage = FileRelatedService.convert_url_to_file(targetImageURL)
        
        image1_bytes = profileImage.read()
        image2_bytes = targetImage.read()
        
        image1_face_path, temp_dir1 = preprocess_and_save_image(image1_bytes, profileImage.name)
        image2_face_path, temp_dir2 = preprocess_and_save_image(image2_bytes, targetImage.name)
        
        if not image1_face_path or not image2_face_path:
            return Response({'result': False, 'confidence_percentage': 0.0, 'note': 'Face detection failed.'}, status=200)
        
        result, confidence = compare_faces(image1_face_path, image2_face_path)
        shutil.rmtree(temp_dir1)
        shutil.rmtree(temp_dir2)
        
        note = 'Comparison successful but faces do not match closely.' if not result else 'Comparison successful and faces match closely.'
        return Response({'result': result, 'confidence_percentage': confidence, 'note': note}, status=200)

    @action(detail=False, methods=['POST'], url_path='compare-image', serializer_class=ImageSerializer)
    def compare_image_file(self, request):
        if 'profileImage' not in request.FILES or 'targetImage' not in request.FILES:
            return Response({'error': 'Please upload two images'}, status=400)
        
        profileImage = request.FILES['profileImage']
        targetImage = request.FILES['targetImage']
        
        if not (allowed_file(targetImage.name) and allowed_file(profileImage.name)):
            return Response({'error': 'Invalid file type. Allowed: png, jpg, jpeg, gif'}, status=400)
        
        image1_bytes = profileImage.read()
        image2_bytes = targetImage.read()
        
        image1_face_path, temp_dir1 = preprocess_and_save_image(image1_bytes, profileImage.name)
        image2_face_path, temp_dir2 = preprocess_and_save_image(image2_bytes, targetImage.name)
        
        if not image1_face_path or not image2_face_path:
            return Response({'result': False, 'confidence_percentage': 0.0, 'note': 'Face detection failed.'}, status=200)
        
        result, confidence = compare_faces(image1_face_path, image2_face_path)
        shutil.rmtree(temp_dir1)
        shutil.rmtree(temp_dir2)
        
        note = 'Comparison successful but faces do not match closely.' if not result else 'Comparison successful and faces match closely.'
        return Response({'result': result, 'confidence_percentage': confidence, 'note': note}, status=200)
