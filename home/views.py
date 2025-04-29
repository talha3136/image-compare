from datetime import datetime, timezone
import os
from venv import logger
import torch
from home.dataSet_service import ClipDataset
from .models import DataSet, TrainingState, uniformChecker
from .serializers import CustomUniformVerifyImageSerializer, CustomUniformVerifyImageUrlSerializer, DataSettSerializer, GenrateDataSetFromUrlSerializer, GenrateDataSetSerializer, ImageSerializer, ImageUrlSerializer, uniformCheckerSerializer, uniformVerifyImageSerializer, uniformVerifyImageUrlSerializer
from rest_framework import viewsets, mixins, status
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework.response import Response
from rest_framework.decorators import action
from .services import FileRelatedService
import requests
from .support_services import allowed_file, compare_faces, preprocess_and_save_image, verify_uniform, save_uploaded_image, ALLOWED_EXTENSIONS
from .utils import DefaultPagination
from django.core.files.base import ContentFile
import open_clip
from PIL import Image
import io
from torch.utils.data import Dataset, DataLoader

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "ViT-B-32"
PRETRAINED_SOURCE = "openai"
MODEL_CHECKPOINT = "model_checkpoints/clip_uniform.pt"
BATCH_SIZE = 16
LEARNING_RATE = 1e-5
EPOCHS = 3

# Initialize model and tokenizer (loaded once at startup)
model, _, preprocess = open_clip.create_model_and_transforms(MODEL_NAME, pretrained=PRETRAINED_SOURCE)
tokenizer = open_clip.get_tokenizer(MODEL_NAME)

# Load checkpoint if it exists
if os.path.exists(MODEL_CHECKPOINT):
    model.load_state_dict(torch.load(MODEL_CHECKPOINT, map_location=DEVICE))
model.to(DEVICE).eval()

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

        # Reload model state from checkpoint to ensure latest version
        if os.path.exists(MODEL_CHECKPOINT):
            try:
                model.load_state_dict(torch.load(MODEL_CHECKPOINT, map_location=DEVICE))
                model.to(DEVICE).eval()
            except Exception as e:
                logger.error(f"Error loading model checkpoint: {str(e)}")
                return Response({'error': 'Failed to load model'}, status=500)

        image = Image.open(file).convert('RGB')
        image_tensor = preprocess(image).unsqueeze(0).to(DEVICE)

        default_prompt = (
            "a UK security officer wearing a crisp white dress shirt, black tie, hi-vis "
        )
        negative_prompt = (
            "a person in casual clothes without white shirt and black tie"
        )
        prompt = user_prompt if user_prompt else default_prompt

        text_inputs = tokenizer([prompt, negative_prompt]).to(DEVICE)

        with torch.no_grad():
            image_features = model.encode_image(image_tensor)
            text_features = model.encode_text(text_inputs)
            logits = image_features @ text_features.T
            probs = logits.softmax(dim=-1).cpu().numpy()[0]

        confidence = probs[0]
        result = confidence >= threshold
        note = "✅ Uniform detected." if result else "❌ Uniform not detected."

        return Response({
            "result": result,
            "confidence_pct": f"{confidence * 100:.2f}",
            "summary": note
        })
    
    @action(
        detail=False,
        methods=['POST'],
        url_path='verify-uniform-with-image-url-v2',
        serializer_class=CustomUniformVerifyImageUrlSerializer
    )
    def verify_uniform_image_url_v2(self, request):
        serializer = CustomUniformVerifyImageUrlSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        image_url = serializer.validated_data['imageURL']
        threshold = serializer.validated_data.get('threshold', 0.65)
        user_prompt = serializer.validated_data.get('prompt', '')

        # Download and load image
        try:
            image_data = requests.get(image_url, timeout=10).content
            image = Image.open(io.BytesIO(image_data)).convert('RGB')
        except:
            return Response({'error': 'Invalid or inaccessible image URL'}, status=400)

        # Reload model from checkpoint
        try:
            model.load_state_dict(torch.load(MODEL_CHECKPOINT, map_location=DEVICE))
            model.to(DEVICE).eval()
        except:
            return Response({'error': 'Failed to load model'}, status=500)

        image_tensor = preprocess(image).unsqueeze(0).to(DEVICE)

        prompt = user_prompt or "a UK security officer wearing a crisp white dress shirt, black tie, hi-vis"
        negative_prompt = "a person in casual clothes without white shirt and black tie"

        text_inputs = tokenizer([prompt, negative_prompt]).to(DEVICE)

        with torch.no_grad():
            image_features = model.encode_image(image_tensor)
            text_features = model.encode_text(text_inputs)
            probs = (image_features @ text_features.T).softmax(dim=-1).cpu().numpy()[0]

        confidence = probs[0]
        result = confidence >= threshold

        return Response({
            "result": result,
            "confidence_pct": f"{confidence * 100:.2f}",
            "summary": "✅ Uniform detected." if result else "❌ Uniform not detected."
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
        """Generate dataset entry from image URL"""
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        image_url = serializer.validated_data['imageURL']
        prompt = serializer.validated_data.get('prompt', '')

        try:
            response = requests.get(image_url, timeout=10)
            response.raise_for_status()

            if not response.headers.get('content-type', '').startswith('image/'):
                return Response({'error': 'URL does not point to an image'}, status=400)

            image_bytes = response.content
            Image.open(io.BytesIO(image_bytes)).verify()  # Validate image
        except Exception:
            return Response({'error': 'Invalid or inaccessible image URL'}, status=400)

        filename = os.path.basename(image_url.split('?')[0]) or f"dataset_{timezone.now().strftime('%Y%m%d_%H%M%S')}.jpg"

        try:
            django_file = ContentFile(image_bytes, name=filename)
            dataset_entry = DataSet.objects.create(image=django_file, prompt=prompt)
            return Response({'message': 'Dataset entry created', 'id': dataset_entry.id}, status=201)
        except Exception as e:
            return Response({'error': f'Failed to save dataset: {str(e)}'}, status=500)

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
        try:
            # Initialize fresh model for training
            train_model, _, train_preprocess = open_clip.create_model_and_transforms(MODEL_NAME, pretrained=PRETRAINED_SOURCE)
            train_tokenizer = open_clip.get_tokenizer(MODEL_NAME)
            train_model = train_model.to(DEVICE)
            train_model.train()

            # Get new data
            state, _ = TrainingState.objects.get_or_create(id=1)
            new_data = (
                DataSet.objects.filter(id__gt=state.last_trained_id.id).order_by('id')
                if state.last_trained_id
                else DataSet.objects.all().order_by('id')
            )

            if not new_data.exists():
                return Response({'message': 'No new data to train on.'}, status=200)

            # Validate image accessibility
            for item in new_data:
                try:
                    with item.image.open('rb') as f:
                        pass  # Check if file can be opened
                except Exception as e:
                    logger.error(f"Image not accessible for item {item.id}: {str(e)}")
                    return Response({'message': f'Image not accessible: {item.image.name}'}, status=400)

            # Create dataset and DataLoader
            dataset = ClipDataset(new_data, train_preprocess, train_tokenizer)
            dataloader = DataLoader(
                dataset,
                batch_size=BATCH_SIZE,
                shuffle=True,
                num_workers=4,
                pin_memory=True if DEVICE == "cuda" else False
            )

            # Train model
            optimizer = torch.optim.AdamW(train_model.parameters(), lr=LEARNING_RATE)
            loss_fn = torch.nn.CrossEntropyLoss()

            for epoch in range(EPOCHS):
                for images, text_tokens in dataloader:
                    images, text_tokens = images.to(DEVICE), text_tokens.to(DEVICE)
                    # Ensure text_tokens is 2D (batch_size, sequence_length)
                    if text_tokens.dim() > 2:
                        text_tokens = text_tokens.squeeze()  # Remove extra dimensions if any
                        if text_tokens.dim() == 1:  # Handle edge case of single sequence
                            text_tokens = text_tokens.unsqueeze(0)

                    image_features = train_model.encode_image(images)
                    text_features = train_model.encode_text(text_tokens)
                    logits = image_features @ text_features.T
                    labels = torch.arange(len(images), device=DEVICE)
                    loss = loss_fn(logits, labels)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

            # Save model checkpoint
            os.makedirs(os.path.dirname(MODEL_CHECKPOINT), exist_ok=True)
            torch.save(train_model.state_dict(), MODEL_CHECKPOINT)

            # Update global model with new state
            model.load_state_dict(torch.load(MODEL_CHECKPOINT, map_location=DEVICE))
            model.to(DEVICE).eval()

            # Update training state
            state.last_trained_id = new_data.last()
            state.last_trained_time = datetime.now(timezone.utc)
            state.save()

            return Response({
                'message': f'Trained on {new_data.count()} new samples.',
                'last_trained_time': state.last_trained_time
            }, status=200)

        except Exception as e:
            logger.error(f"Training error: {str(e)}")
            return Response({'message': f'Error training model: {str(e)}'}, status=500)

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