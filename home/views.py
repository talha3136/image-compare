# import shutil
# import tempfile
# import os
# import io
# import cv2
# import dlib
# from home.serializers import ImageSerializer
# import numpy as np
# from PIL import Image
# from datetime import datetime
# from deepface import DeepFace
# from django.http import JsonResponse
# from rest_framework import viewsets
# from rest_framework.parsers import MultiPartParser, FormParser
# from rest_framework.response import Response
# from rest_framework.decorators import action

# # Load dlib face detector and shape predictor
# detector = dlib.get_frontal_face_detector()
# predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

# def allowed_file(filename):
#     return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# def draw_face_landmarks(image):
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     faces = detector(gray)
#     cropped_faces = []  
    
#     for face in faces:
#         x1, y1, x2, y2 = (face.left(), face.top(), face.right(), face.bottom())
#         cropped_face = image[y1:y2, x1:x2]
#         cropped_faces.append(cropped_face)
        
#         landmarks = predictor(gray, face)
#         for n in range(68):
#             x, y = landmarks.part(n).x, landmarks.part(n).y
#             cv2.circle(cropped_face, (x - x1, y - y1), 1, (0, 255, 0), -1)
    
#     if not cropped_faces:
#         return image
    
#     return cropped_faces[0]

# def preprocess_and_save_image_with_landmarks(image_bytes, filename):
#     try:
#         image = Image.open(io.BytesIO(image_bytes))
#         image_np = np.array(image)
#         image_cv = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        
#         image_with_landmarks = draw_face_landmarks(image_cv)
#         if image_with_landmarks is image_cv:
#             return None, None
        
#         temp_dir = tempfile.mkdtemp()
#         timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
#         output_filename = f"landmarks_{timestamp}_{filename}"
#         output_filepath = os.path.join(temp_dir, output_filename)
        
#         cv2.imwrite(output_filepath, image_with_landmarks)
#         return output_filepath, temp_dir
#     except Exception as e:
#         print(f"Error processing and saving the image with landmarks {filename}: {e}")
#         return None, None

# def compare_faces_multiple_models(image1_path, image2_path):
#     models = ['VGG-Face', 'Facenet', 'Facenet512', 'OpenFace', 'DeepFace']
#     results = []
    
#     for model in models:
#         try:
#             result = DeepFace.verify(img1_path=image1_path, img2_path=image2_path, model_name=model, detector_backend='mtcnn')
#             results.append(result)
#         except Exception as e:
#             print(f"Error with {model} model: {e}")
    
#     if not results:
#         return False, "No valid comparisons"
    
#     verified_count = sum(1 for result in results if result['verified'])
#     verification_rate = verified_count / len(results)
#     distances = [result['distance'] for result in results if 'distance' in result]
#     avg_distance = np.mean(distances) if distances else 1.0
    
#     is_match = verification_rate > 0.5 and avg_distance < 0.5
#     confidence = max(0, min(100, (1 - avg_distance) * 100 * verification_rate))
    
#     return bool(is_match), f"{confidence:.2f}%"


# from rest_framework.decorators import action
# from drf_yasg.utils import swagger_auto_schema
# from drf_yasg import openapi
# class CompareImagesViewSet(viewsets.ViewSet):
#     serializer_class = ImageSerializer
#     parser_classes = (MultiPartParser, FormParser)
    
#     @swagger_auto_schema(
#         operation_description="Compare two images",
#         request_body=ImageSerializer,
#         responses={200: openapi.Response('Comparison result', ImageSerializer)}
#     )
#     @action(
#         detail=False, 
#         methods=['post'],
#         url_path='compare',
#     )
#     def compare(self, request):
#         if 'image1' not in request.FILES or 'image2' not in request.FILES:
#             return Response({'error': 'Please upload two images'}, status=400)
        
#         image1 = request.FILES['image1']
#         image2 = request.FILES['image2']
        
#         if not (allowed_file(image1.name) and allowed_file(image2.name)):
#             return Response({'error': 'Invalid file type. Allowed: png, jpg, jpeg, gif'}, status=400)
        
#         image1_bytes = image1.read()
#         image2_bytes = image2.read()
        
#         image1_face_path, temp_dir1 = preprocess_and_save_image_with_landmarks(image1_bytes, image1.name)
#         image2_face_path, temp_dir2 = preprocess_and_save_image_with_landmarks(image2_bytes, image2.name)
        
#         if image1_face_path is None or image2_face_path is None:
#             return Response({'error': 'Face detection failed for one or both images'}, status=400)
        
#         result, confidence = compare_faces_multiple_models(image1_face_path, image2_face_path)
        
#         shutil.rmtree(temp_dir1)
#         shutil.rmtree(temp_dir2)
        
#         return Response({'result': result, 'confidence': confidence})



import shutil
import tempfile
import os
import io
import cv2
from home.serializers import ImageSerializer
import numpy as np
from PIL import Image
from datetime import datetime
from deepface import DeepFace
from django.http import JsonResponse
from rest_framework import viewsets
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework.response import Response
from rest_framework.decorators import action
from rest_framework.decorators import action
from drf_yasg.utils import swagger_auto_schema
from drf_yasg import openapi

# Load OpenCV face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def detect_face(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    cropped_faces = []
    for (x, y, w, h) in faces:
        cropped_face = image[y:y + h, x:x + w]
        cropped_faces.append(cropped_face)
    
    return cropped_faces

def preprocess_and_save_image(image_bytes, filename):
    try:
        image = Image.open(io.BytesIO(image_bytes))
        image_np = np.array(image)
        image_cv = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        
        faces = detect_face(image_cv)
        if not faces:
            return None, None
        
        # Save the first detected face
        temp_dir = tempfile.mkdtemp()
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_filename = f"face_{timestamp}_{filename}"
        output_filepath = os.path.join(temp_dir, output_filename)
        
        cv2.imwrite(output_filepath, faces[0])  # Save the first detected face
        return output_filepath, temp_dir
    except Exception as e:
        print(f"Error processing and saving the image {filename}: {e}")
        return None, None

def compare_faces_multiple_models(image1_path, image2_path):
    models = ['VGG-Face', 'Facenet', 'Facenet512', 'OpenFace', 'DeepFace']
    results = []
    
    for model in models:
        try:
            result = DeepFace.verify(img1_path=image1_path, img2_path=image2_path, model_name=model, detector_backend='mtcnn')
            results.append(result)
        except Exception as e:
            print(f"Error with {model} model: {e}")
    
    if not results:
        return False, "No valid comparisons"
    
    verified_count = sum(1 for result in results if result['verified'])
    verification_rate = verified_count / len(results)
    distances = [result['distance'] for result in results if 'distance' in result]
    avg_distance = np.mean(distances) if distances else 1.0
    
    is_match = verification_rate > 0.5 and avg_distance < 0.5
    confidence = max(0, min(100, (1 - avg_distance) * 100 * verification_rate))
    
    return bool(is_match), f"{confidence:.2f}%"

class CompareImagesViewSet(viewsets.ViewSet):
    serializer_class = ImageSerializer
    parser_classes = (MultiPartParser, FormParser)
    @swagger_auto_schema(
        operation_description="Compare two images",
        request_body=ImageSerializer,
        responses={200: openapi.Response('Comparison result', ImageSerializer)}
    )
    
    @action(
        detail=False, 
        methods=['post'],
        url_path='compare',
    )
    def compare(self, request):
        if 'image1' not in request.FILES or 'image2' not in request.FILES:
            return Response({'error': 'Please upload two images'}, status=400)
        
        image1 = request.FILES['image1']
        image2 = request.FILES['image2']
        
        if not (allowed_file(image1.name) and allowed_file(image2.name)):
            return Response({'error': 'Invalid file type. Allowed: png, jpg, jpeg, gif'}, status=400)
        
        image1_bytes = image1.read()
        image2_bytes = image2.read()
        
        image1_face_path, temp_dir1 = preprocess_and_save_image(image1_bytes, image1.name)
        image2_face_path, temp_dir2 = preprocess_and_save_image(image2_bytes, image2.name)
        
        if image1_face_path is None or image2_face_path is None:
            return Response({'error': 'Face detection failed for one or both images'}, status=400)
        
        result, confidence = compare_faces_multiple_models(image1_face_path, image2_face_path)
        
        shutil.rmtree(temp_dir1)
        shutil.rmtree(temp_dir2)
        
        return Response({'result': result, 'confidence': confidence})
