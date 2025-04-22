import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import io
import cv2
import dlib
import numpy as np
from PIL import Image, ImageEnhance,ExifTags
from datetime import datetime
from deepface import DeepFace
import torch
import clip


device = "cuda" if torch.cuda.is_available() else "cpu"

# Load CLIP model
# model, preprocess = clip.load("ViT-B/32", device=device)
model, preprocess = clip.load("ViT-B/32", device=device, jit=False)


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
    # models = ["VGG-Face", "Facenet", "Facenet512", "DeepFace", "OpenFace", "ArcFace"]
    models = ["Facenet512", "ArcFace"]
    results = []

    try:
        for model_name in models:
            try:
                # result = DeepFace.verify(img1_path=image1_path, img2_path=image2_path, model_name=model_name, detector_backend="retinaface")
                result = DeepFace.verify(img1_path=image1_path, img2_path=image2_path, model_name=model_name, detector_backend="mtcnn")

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


def save_uploaded_image(image_bytes, filename):
    """Save uploaded image to disk and return filepath"""
    output_filename = f"uniform_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_{filename}"
    output_filepath = os.path.join(UPLOADS_DIR, output_filename)
    
    with open(output_filepath, "wb") as f:
        f.write(image_bytes)
    
    return output_filepath


def verify_uniform(image_path, threshold=0.65, user_prompt=None):
    """Verify if image shows security uniform and return results"""
    try:
        # Load and preprocess image
        image = Image.open(image_path).convert("RGB")
        image_input = preprocess(image).unsqueeze(0).to(device)
        
        # Default text prompt
        default_prompt = "a UK security officer wearing a crisp white dress shirt and black tie, with formal trousers, with or without jacket, may have security badges or epaulettes"
        negative_prompt = "a person in casual clothes without white shirt and black tie: t-shirts, sweaters, jeans, hoodies, or informal attire"
        
        # Use user prompt if provided
        positive_prompt = user_prompt if user_prompt else default_prompt
        text_prompts = [positive_prompt, negative_prompt]
        
        text_tokens = clip.tokenize(text_prompts).to(device)
        
        # Get prediction
        with torch.no_grad():
            logits_per_image, _ = model(image_input, text_tokens)
            probs = logits_per_image.softmax(dim=-1).cpu().numpy()
        
        # Calculate result
        confidence = float(probs[0][0])
        result = confidence >= threshold
        confidence_percentage = f"{confidence*100:.2f}"
        
        # Dummy elements - would ideally be extracted based on prompt
        # required_elements = ["White dress shirt", "Black tie"]
        # optional_elements = ["Security jacket", "Epaulettes", "Badge"]
        
        # Determine missing elements
        missing_elements = []
        if not result:
            missing_elements = ["White dress shirt and black tie"] if confidence < 0.5 else ["One or more key uniform elements"]
        
        # Create response message
        note = "Uniform verification successful. Security uniform detected." if result else "Verification completed but security uniform not detected. Missing required elements."
        
        # return result, confidence_percentage, note, required_elements, optional_elements, missing_elements
        return result, confidence_percentage, note, missing_elements

    
    except Exception as e:
        return False, "0.00", f"Error during verification: {str(e)}", [], [], []
    finally:
        # Clean up the file
        if os.path.exists(image_path):
            os.remove(image_path)

            
# def verify_uniform(image_path, threshold=0.65):
#     """Verify if image shows security uniform and return results"""
#     try:
#         # Load and preprocess image
#         image = Image.open(image_path).convert("RGB")
#         image_input = preprocess(image).unsqueeze(0).to(device)
        
#         # Text prompts for CLIP
#         text_prompts = [
#             # Security uniform description
#             "a UK security officer wearing a crisp white dress shirt and black tie, with formal trousers, with or without jacket, may have security badges or epaulettes",
            
#             # Negative case
#             "a person in casual clothes without white shirt and black tie: t-shirts, sweaters, jeans, hoodies, or informal attire"
#         ]
        
#         text_tokens = clip.tokenize(text_prompts).to(device)
        
#         # Get prediction
#         with torch.no_grad():
#             logits_per_image, _ = model(image_input, text_tokens)
#             probs = logits_per_image.softmax(dim=-1).cpu().numpy()
        
#         # Calculate result
#         confidence = float(probs[0][0])
#         result = confidence >= threshold
#         confidence_percentage = f"{confidence*100:.2f}"
        
#         # Define uniform elements
#         required_elements = ["White dress shirt", "Black tie", "Formal trousers"]
#         optional_elements = ["Security jacket", "Epaulettes", "Badge"]
        
#         # Determine missing elements
#         missing_elements = []
#         if not result:
#             missing_elements = ["White dress shirt and black tie"] if confidence < 0.5 else ["One or more key uniform elements"]
        
#         # Create response message
#         if result:
#             # note = "Uniform verification successful. Security uniform detected with proper white shirt and black tie."
#             note = "Uniform verification successful. Security uniform detected."

#         else:
#             note = "Verification completed but security uniform not detected. Missing required elements."
        
#         return result, confidence_percentage, note, required_elements, optional_elements, missing_elements
    
#     except Exception as e:
#         return False, "0.00", f"Error during verification: {str(e)}", [], [], []
#     finally:
#         # Clean up the file
#         if os.path.exists(image_path):
#             os.remove(image_path)