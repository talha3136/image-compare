import os
import io
import cv2
import dlib
import numpy as np
from PIL import Image, ImageEnhance, ExifTags
from datetime import datetime
from deepface import DeepFace
from rest_framework import viewsets
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework.response import Response
from rest_framework.decorators import action
from home.services import FileRelatedService
from home.serializers import ImageSerializer, ImageUrlSerializer

# Load dlib face detector and shape predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'webp'}
UPLOADS_DIR = "uploads"
os.makedirs(UPLOADS_DIR, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def is_blurry(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    return variance < 100

def enhance_image(image):
    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    enhancer = ImageEnhance.Sharpness(pil_image)
    sharp_image = enhancer.enhance(2.0)
    enhancer = ImageEnhance.Contrast(sharp_image)
    contrast_image = enhancer.enhance(1.5)
    return cv2.cvtColor(np.array(contrast_image), cv2.COLOR_RGB2BGR)

def auto_rotate_image(image):
    """ Auto-rotate image using EXIF metadata """
    try:
        for orientation in ExifTags.TAGS.keys():
            if ExifTags.TAGS[orientation] == 'Orientation':
                break
        exif = image._getexif()
        if exif is not None:
            orientation = exif.get(orientation, None)
            if orientation == 3:
                image = image.rotate(180, expand=True)
            elif orientation == 6:
                image = image.rotate(270, expand=True)
            elif orientation == 8:
                image = image.rotate(90, expand=True)
    except Exception:
        pass  # Ignore if EXIF metadata is missing
    return image

def is_flipped(image):
    """ Detect if an image is flipped using face landmarks symmetry """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    if len(faces) == 0:
        return False  # No face detected, assume not flipped

    for face in faces:
        landmarks = predictor(gray, face)
        left_eye_x = (landmarks.part(36).x + landmarks.part(39).x) / 2
        right_eye_x = (landmarks.part(42).x + landmarks.part(45).x) / 2
        return left_eye_x > right_eye_x  # True if flipped

    return False

def preprocess_and_save_image(image_bytes, filename):
    try:
        image = Image.open(io.BytesIO(image_bytes))
        image = auto_rotate_image(image)  # Fix rotation
        image_np = np.array(image)
        image_cv = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

        if is_flipped(image_cv):
            image_cv = cv2.flip(image_cv, 1)  # Fix flipped image

        if is_blurry(image_cv):
            image_cv = enhance_image(image_cv)

        output_filename = f"processed_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_{filename}"
        output_filepath = os.path.join(UPLOADS_DIR, output_filename)
        cv2.imwrite(output_filepath, image_cv)
        return output_filepath
    except Exception:
        return None

def compare_faces(image1_path, image2_path):
    models = ["Facenet512", "OpenFace", "ArcFace"]
    results = []

    try:
        for model_name in models:
            try:
                result = DeepFace.verify(img1_path=image1_path, img2_path=image2_path, model_name=model_name, detector_backend="retinaface")
                verified = result.get('verified', False)
                distance = result.get('distance', None)
                results.append((model_name, verified, distance))
            except Exception:
                results.append((model_name, False, None))

        verified_count = sum(1 for _, verified, _ in results if verified)
        total_models = len(models)
        result = verified_count >= total_models / 2
        confidence = sum([(1 - distance) * 100 if distance is not None else 0 for _, _, distance in results]) / len(results)
        note = "Faces match closely." if result else "Faces do not match closely."

        return result, f"{confidence:.2f}", note
    finally:
        if os.path.exists(image1_path):
            os.remove(image1_path)
        if os.path.exists(image2_path):
            os.remove(image2_path)

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