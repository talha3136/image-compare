
# import cv2
# import numpy as np
# from PIL import Image

# # Load the uploaded image
# image_path = "/mnt/data/2025-02-17_1525410000-arezCapturedImage.jpg"
# image = cv2.imread(image_path)

# # Check if the image is loaded correctly
# if image is None:
#     result = "Error: Unable to load the image."
# else:
#     # Convert to grayscale to check for blurriness
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    
#     # Check image dimensions and channels
#     height, width, channels = image.shape if len(image.shape) == 3 else (*image.shape, 1)

#     result = {
#         "image_shape": (height, width, channels),
#         "blurriness_variance": variance,
#         "is_blurry": variance < 100
#     }

# result





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

def adjust_gamma(image, gamma=1.0):
    invGamma = 1.0 / gamma
    table = np.array([(i / 255.0) ** invGamma * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

def enhance_image(image):
    """
    Enhance blurry and low-quality images by applying sharpening filter, 
    contrast enhancement, noise reduction, and unsharp masking.
    """
    
    # Convert to LAB color space to adjust contrast
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
    l, a, b = cv2.split(lab)

    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) for better contrast
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l = clahe.apply(l)

    # Merge and convert back to BGR
    lab = cv2.merge((l, a, b))
    contrast_enhanced = cv2.cvtColor(lab, cv2.COLOR_Lab2BGR)
    
    # Adjust brightness using Gamma correction
    mean_brightness = np.mean(cv2.cvtColor(contrast_enhanced, cv2.COLOR_BGR2GRAY))
    gamma = 1.5 if mean_brightness < 100 else 0.8 if mean_brightness > 180 else 1.0
    contrast_enhanced = adjust_gamma(contrast_enhanced, gamma)
    
    # Sharpening using a stronger kernel
    sharpen_kernel = np.array([[0, -1, 0],
                               [-1, 5, -1],
                               [0, -1, 0]])
    sharpened = cv2.filter2D(contrast_enhanced, -1, sharpen_kernel)
    
    # Noise reduction using bilateral filter
    denoised = cv2.bilateralFilter(sharpened, d=9, sigmaColor=50, sigmaSpace=50)
    
    # Unsharp Masking for additional clarity
    blurred = cv2.GaussianBlur(denoised, (5, 5), 10.0)
    unsharp_image = cv2.addWeighted(denoised, 1.7, blurred, -0.7, 0)
    
    return unsharp_image


cnn_model_path = "mmod_human_face_detector.dat"
dlib_model_path = "shape_predictor_68_face_landmarks.dat"

if os.path.exists(cnn_model_path):
    detector = dlib.cnn_face_detection_model_v1(cnn_model_path)
    use_cnn = True
else:
    detector = dlib.get_frontal_face_detector()
    use_cnn = False

if os.path.exists(dlib_model_path):
    predictor = dlib.shape_predictor(dlib_model_path)
else:
    raise FileNotFoundError("Shape predictor model not found!")

def draw_face_landmarks(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    cropped_faces = []
    
    if use_cnn:
        faces = [d.rect for d in faces]  # Extract rectangle from CNN detection
    
    if isinstance(faces, list) and not faces:
        # Use OpenCV Haar Cascade as a fallback
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        detected_faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        for (x, y, w, h) in detected_faces:
            faces.append(dlib.rectangle(x, y, x+w, y+h))
    
    for face in faces:
        x1, y1, x2, y2 = max(0, face.left()-20), max(0, face.top()-20), min(image.shape[1], face.right()+20), min(image.shape[0], face.bottom()+20)
        cropped_face = image[y1:y2, x1:x2]
        cropped_faces.append(cropped_face)
        
        landmarks = predictor(gray, face)
        for n in range(68):
            x, y = landmarks.part(n).x - x1, landmarks.part(n).y - y1
            cv2.circle(cropped_face, (x, y), 1, (0, 255, 0), -1)
    
    if not cropped_faces:
        return image, "No face detected"
    
    return cropped_faces[0], None




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
        return False, 0.0
    
    verified_count = sum(1 for result in results if result['verified'])
    verification_rate = verified_count / len(results)
    distances = [result['distance'] for result in results if 'distance' in result]
    avg_distance = np.mean(distances) if distances else 1.0
    
    if verification_rate <= 0.5 or avg_distance >= 0.5:
        return False, 0.0
    
    confidence = max(0, min(100, (1 - avg_distance) * 100 * verification_rate))
    return True, f"{confidence:.2f}"


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
            image_with_error = "profileImage" if image1_face_path is None else "targetImage"
            # Return status code 200 even when there's an error
            return Response({
                'result': False, 
                'confidence_percentage': 0.0, 
                'note': f'{error_msg} ',
                'image_with_error': image_with_error
            }, status=200)
        
        
        result, confidence = compare_faces_multiple_models(image1_face_path, image2_face_path)
        
        shutil.rmtree(temp_dir1)
        shutil.rmtree(temp_dir2)
        
        note = 'Comparison successful but faces do not match closely.' if not result else 'Comparison successful and faces match closely.'
        return Response({
            'result': result, 
            'confidence_percentage': confidence,
            'note': note,
            'image_with_error': 'None' 
        }, status=200)


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
            # If either image fails, return the corresponding error message with the image that caused the error
            detection_error = detection_error1 or detection_error2
            image_with_error = "profileImage" if image1_face_path is None else "targetImage"
            return Response({
                'result': False, 
                'confidence_percentage': 0.0, 
                'note': detection_error,
                'image_with_error': image_with_error
            }, status=200)
        
        
        result, confidence = compare_faces_multiple_models(image1_face_path, image2_face_path)
        
        # Clean up temporary directories
        shutil.rmtree(temp_dir1)
        shutil.rmtree(temp_dir2)
        
        note = 'Comparison successful but faces do not match closely.' if not result else 'Comparison successful and faces match closely.'
        return Response({
            'result': result, 
            'confidence_percentage': confidence,
            'note': note,
            'image_with_error': 'None'
        }, status=200)

