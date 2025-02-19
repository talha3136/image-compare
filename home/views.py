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
# from drf_yasg.utils import swagger_auto_schema
# from drf_yasg import openapi

# from home.services import FileRelatedService

# # Load dlib face detector and shape predictor
# detector = dlib.get_frontal_face_detector()
# predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')




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

#         if not filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff','webp')):
#             filename = f"{filename}.jpg"

#         timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
#         output_filename = f"landmarks_{timestamp}_{filename}"
#         output_filepath = os.path.join(temp_dir, output_filename)
#         os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
#         cv2.imwrite(output_filepath, image_with_landmarks)
#         return output_filepath, temp_dir
#     except Exception as e:
#         print(f"Error processing and saving the image with landmarks {filename}: {e}")
#         return None, None

# def compare_faces_multiple_models(image1_path, image2_path):
#     # models = ['VGG-Face', 'Facenet', 'Facenet512', 'OpenFace', 'DeepFace','InsightFace'] 

#     models = ['ArcFace', 'Facenet', 'DeepFace']


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



# class CompareImagesViewSet(viewsets.GenericViewSet):
#     serializer_class = ImageSerializer
#     parser_classes = (MultiPartParser, FormParser)

#     @action(
#         detail=False, 
#         methods=['POST'],
#         url_path='compare',
#         serializer_class = ImageSerializer
#     )
#     def compare_image(self, request):        
#         profileImageURL = request.data.get('profileImageURL')
#         targetImageURL = request.data.get('targetImageURL')

#         profileImage = FileRelatedService.convert_url_to_file(profileImageURL)
#         targetImage = FileRelatedService.convert_url_to_file(targetImageURL)
        
#         image1_bytes = profileImage.read()
#         image2_bytes = targetImage.read()
        
#         image1_face_path, temp_dir1 = preprocess_and_save_image_with_landmarks(image1_bytes, profileImage.name)
#         image2_face_path, temp_dir2 = preprocess_and_save_image_with_landmarks(image2_bytes, targetImage.name)
        
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
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif','webp'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def is_blurry(image):
    """
    Check if the image is blurry by calculating the variance of the Laplacian.
    A lower variance indicates a blurrier image.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    return variance < 100  # Threshold value for blurriness, can be adjusted


def enhance_image(image):
    """
    Enhance blurry images by applying sharpening filter, contrast enhancement, and unsharp masking.
    """

    # Sharpening using stronger kernel
    kernel = np.array([[-1, -1, -1],
                       [-1,  9, -1],
                       [-1, -1, -1]])
    sharpened_image = cv2.filter2D(image, -1, kernel)

    # Increase the contrast of the image
    lab = cv2.cvtColor(sharpened_image, cv2.COLOR_BGR2Lab)
    l, a, b = cv2.split(lab)
    l = cv2.equalizeHist(l)
    enhanced_image = cv2.merge((l, a, b))
    enhanced_image = cv2.cvtColor(enhanced_image, cv2.COLOR_Lab2BGR)

    # Unsharp Masking for additional enhancement
    blurred = cv2.GaussianBlur(enhanced_image, (9, 9), 10.0)
    unsharp_image = cv2.addWeighted(enhanced_image, 1.5, blurred, -0.5, 0)

    return unsharp_image


def draw_face_landmarks(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    cropped_faces = []  
    
    for face in faces:
        x1, y1, x2, y2 = (face.left(), face.top(), face.right(), face.bottom())
        cropped_face = image[y1:y2, x1:x2]
        cropped_faces.append(cropped_face)
        
        landmarks = predictor(gray, face)
        for n in range(68):
            x, y = landmarks.part(n).x, landmarks.part(n).y
            cv2.circle(cropped_face, (x - x1, y - y1), 1, (0, 255, 0), -1)
    
    if not cropped_faces:
        return image, "No face detected"
    
    return cropped_faces[0], None  # Return the cropped face


def preprocess_and_save_image_with_landmarks(image_bytes, filename):
    try:
        image = Image.open(io.BytesIO(image_bytes))
        image_np = np.array(image)
        image_cv = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

        # Check if the image is blurry and enhance if needed
        if is_blurry(image_cv):
            # print(f"Image {filename} is blurry. Enhancing...")
            image_cv = enhance_image(image_cv)

        image_with_landmarks, detection_error = draw_face_landmarks(image_cv)
        
        if image_with_landmarks is image_cv:
            return None, None, detection_error
        
        temp_dir = tempfile.mkdtemp()

        if not filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff', 'webp')):
            filename = f"{filename}.jpg"

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_filename = f"landmarks_{timestamp}_{filename}"
        output_filepath = os.path.join(temp_dir, output_filename)
        os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
        cv2.imwrite(output_filepath, image_with_landmarks)
        return output_filepath, temp_dir, None
    except Exception as e:
        print(f"Error processing and saving the image with landmarks {filename}: {e}")
        return None, None, f"Error processing image: {e}"


def compare_faces_multiple_models(image1_path, image2_path):
    models = ['ArcFace', 'Facenet', 'DeepFace']
    results = []
    
    for model in models:
        try:
            result = DeepFace.verify(img1_path=image1_path, img2_path=image2_path, model_name=model, detector_backend='mtcnn')
            results.append(result)
        except Exception as e:
            print(f"Error with {model} model: {e}")
    
    if not results:
        return False, "No valid comparisons were made"
    
    verified_count = sum(1 for result in results if result['verified'])
    verification_rate = verified_count / len(results)
    distances = [result['distance'] for result in results if 'distance' in result]
    avg_distance = np.mean(distances) if distances else 1.0
    
    if verification_rate <= 0.5 or avg_distance >= 0.5:
        return False, "Low verification rate or high distance between faces"
    
    confidence = max(0, min(100, (1 - avg_distance) * 100 * verification_rate))
    return True, f"{confidence:.2f}%"


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

        profileImage = FileRelatedService.convert_url_to_file(profileImageURL)
        targetImage = FileRelatedService.convert_url_to_file(targetImageURL)
        
        image1_bytes = profileImage.read()
        image2_bytes = targetImage.read()
        
        image1_face_path, temp_dir1, error1 = preprocess_and_save_image_with_landmarks(image1_bytes, profileImage.name)
        image2_face_path, temp_dir2, error2 = preprocess_and_save_image_with_landmarks(image2_bytes, targetImage.name)
        
        if image1_face_path is None or image2_face_path is None:
            error_msg = error1 if error1 else error2
            return Response({'error': error_msg}, status=400)
        
        result, confidence = compare_faces_multiple_models(image1_face_path, image2_face_path)
        
        shutil.rmtree(temp_dir1)
        shutil.rmtree(temp_dir2)
        
        if not result:
            return Response({'error': confidence}, status=400)
        
        return Response({'result': result, 'confidence': confidence})


    @action(
        detail=False, 
        methods=['post'],
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
        
        # Adjusted to unpack three values: image face path, temporary directory, and error message
        image1_face_path, temp_dir1, detection_error1 = preprocess_and_save_image_with_landmarks(image1_bytes, profileImage.name)
        image2_face_path, temp_dir2, detection_error2 = preprocess_and_save_image_with_landmarks(image2_bytes, targetImage.name)
        
        if image1_face_path is None or image2_face_path is None:
            # If either image fails, return the corresponding error message
            detection_error = detection_error1 or detection_error2
            return Response({'error': detection_error}, status=400)
        
        result, confidence = compare_faces_multiple_models(image1_face_path, image2_face_path)
        
        # Clean up temporary directories
        shutil.rmtree(temp_dir1)
        shutil.rmtree(temp_dir2)
        
        return Response({'result': result, 'confidence': confidence})
