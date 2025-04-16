import os

from home.serializers import ImageSerializer, ImageUrlSerializer, uniformVerifyImageSerializer, uniformVerifyImageUrlSerializer
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import io
import cv2
import dlib
import numpy as np
from PIL import Image, ImageEnhance
from datetime import datetime
from deepface import DeepFace
from rest_framework import viewsets
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework.response import Response
from rest_framework.decorators import action
from home.services import FileRelatedService
import requests
import torch
import clip

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load CLIP model
model, preprocess = clip.load("ViT-B/32", device=device)

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

def preprocess_and_save_image(image_bytes, filename):
    try:
        image = Image.open(io.BytesIO(image_bytes))
        image_np = np.array(image)
        image_cv = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        if is_blurry(image_cv):
            image_cv = enhance_image(image_cv)
        output_filename = f"processed_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_{filename}"
        output_filepath = os.path.join(UPLOADS_DIR, output_filename)
        cv2.imwrite(output_filepath, image_cv)
        return output_filepath
    except Exception as e:
        return None

def compare_faces(image1_path, image2_path):
    models = ["VGG-Face", "Facenet", "Facenet512", "DeepFace", "OpenFace", "ArcFace"]
    # models = [ "Facenet512",  "OpenFace", "ArcFace"]

    results = []
    try:
        for model_name in models:
            try:
                result = DeepFace.verify(img1_path=image1_path, img2_path=image2_path, model_name=model_name, detector_backend="retinaface")
                verified = result.get('verified', False)
                distance = result.get('distance', None)
                results.append((model_name, verified, distance))
            except Exception as e:
                results.append((model_name, False, None))
        verified_count = sum(1 for _, verified, _ in results if verified)
        total_models = len(models)
        result = verified_count >= total_models / 2
        confidence = sum([(1 - distance) * 100 if distance is not None else 0 for _, _, distance in results]) / len(results)
        note = "Comparison successful and faces match closely." if result else "Comparison successful but faces do not match closely."
        return result, f"{confidence:.2f}", note
    finally:
        if os.path.exists(image1_path):
            os.remove(image1_path)
        if os.path.exists(image2_path):
            os.remove(image2_path)


def save_uploaded_image(image_bytes, filename):
    """Save uploaded image to disk and return filepath"""
    output_filename = f"uniform_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_{filename}"
    output_filepath = os.path.join(UPLOADS_DIR, output_filename)
    
    with open(output_filepath, "wb") as f:
        f.write(image_bytes)
    
    return output_filepath


            
def verify_uniform(image_path, threshold=0.65):
    """Verify if image shows security uniform and return results"""
    try:
        # Load and preprocess image
        image = Image.open(image_path).convert("RGB")
        image_input = preprocess(image).unsqueeze(0).to(device)
        
        # Text prompts for CLIP
        text_prompts = [
            # Security uniform description
            "a UK security officer wearing a crisp white dress shirt and black tie, with formal trousers, with or without jacket, may have security badges or epaulettes",
            
            # Negative case
            "a person in casual clothes without white shirt and black tie: t-shirts, sweaters, jeans, hoodies, or informal attire"
        ]
        
        text_tokens = clip.tokenize(text_prompts).to(device)
        
        # Get prediction
        with torch.no_grad():
            logits_per_image, _ = model(image_input, text_tokens)
            probs = logits_per_image.softmax(dim=-1).cpu().numpy()
        
        # Calculate result
        confidence = float(probs[0][0])
        result = confidence >= threshold
        confidence_percentage = f"{confidence*100:.2f}"
        
        # Define uniform elements
        required_elements = ["White dress shirt", "Black tie", "Formal trousers"]
        optional_elements = ["Security jacket", "Epaulettes", "Badge"]
        
        # Determine missing elements
        missing_elements = []
        if not result:
            missing_elements = ["White dress shirt and black tie"] if confidence < 0.5 else ["One or more key uniform elements"]
        
        # Create response message
        if result:
            # note = "Uniform verification successful. Security uniform detected with proper white shirt and black tie."
            note = "Uniform verification successful. Security uniform detected."

        else:
            note = "Verification completed but security uniform not detected. Missing required elements."
        
        return result, confidence_percentage, note, required_elements, optional_elements, missing_elements
    
    except Exception as e:
        return False, "0.00", f"Error during verification: {str(e)}", [], [], []
    finally:
        # Clean up the file
        if os.path.exists(image_path):
            os.remove(image_path)

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
        
        if not image:
            return Response({'error': 'No image file provided'}, status=400)
        
        if not allowed_file(image.name):
            return Response({'error': f'Invalid file type. Allowed: {", ".join(ALLOWED_EXTENSIONS)}'}, status=400)
        
        try:
            # Save and process the image
            image_bytes = image.read()
            image_path = save_uploaded_image(image_bytes, image.name)
            
            # Verify uniform
            result, confidence, note, required, optional, missing = verify_uniform(image_path, threshold)
            
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
            result, confidence, note, required, optional, missing = verify_uniform(image_path, threshold)
            
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