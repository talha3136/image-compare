import io
import os
from venv import logger
import requests
from PIL import Image
import torch
from torch.utils.data import Dataset,DataLoader
from home.models import DataSet, TrainingState

import open_clip
from django.core.files.base import ContentFile
from django.utils import timezone
from urllib.request import urlopen

import multiprocessing
num_workers = min(4,multiprocessing.cpu_count())



DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "ViT-B-32-quickgelu"
PRETRAINED_SOURCE = "openai"
MODEL_CHECKPOINT = "model_checkpoints/clip_uniform.pt"
BATCH_SIZE = 16
LEARNING_RATE = 1e-5
EPOCHS = 3


model, _, preprocess = open_clip.create_model_and_transforms(MODEL_NAME, pretrained=PRETRAINED_SOURCE)
tokenizer = open_clip.get_tokenizer(MODEL_NAME)

# Load checkpoint if it exists
if os.path.exists(MODEL_CHECKPOINT):
    model.load_state_dict(torch.load(MODEL_CHECKPOINT, map_location=DEVICE))
model.to(DEVICE).eval()

class ClipDataset(Dataset):
    def __init__(self, queryset, preprocess, tokenizer):
        self.data = list(queryset)  # Cache for efficiency
        self.preprocess = preprocess
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        try:
            # Load image using storage backend
            with item.image.open('rb') as f:
                image = Image.open(io.BytesIO(f.read())).convert("RGB")
            image = self.preprocess(image)
            # Tokenize text and ensure 2D tensor
            text = self.tokenizer([item.prompt])[0]  # Tokenize as single item, take first tensor
            return image, text
        except Exception as e:
            logger.error(f"Error loading item {item.id}: {str(e)}")
            raise





class UniformVerificationService:

    @staticmethod
    def train_clip_model():
        try:
            train_model, _, train_preprocess = open_clip.create_model_and_transforms(MODEL_NAME, pretrained=PRETRAINED_SOURCE)
            train_tokenizer = open_clip.get_tokenizer(MODEL_NAME)
            train_model = train_model.to(DEVICE)
            train_model.train()

            if os.path.exists(MODEL_CHECKPOINT):
                try:
                    train_model.load_state_dict(torch.load(MODEL_CHECKPOINT, map_location=DEVICE))
                    logger.info("Loaded existing pretrained model checkpoint.")
                except Exception as e:
                    return False, f"Error loading checkpoint: {str(e)}"

            state, _ = TrainingState.objects.get_or_create(id=1)
            new_data = (
                DataSet.objects.filter(id__gt=state.last_trained_id.id).order_by('id')[:30]
                if state.last_trained_id else DataSet.objects.all().order_by('id')[:30]
            )
            new_data_list = list(new_data)

            if not new_data_list:
                return True, {'message': 'No new data to train on.', 'last_trained_time': state.last_trained_time}

            dataset = ClipDataset(new_data_list, train_preprocess, train_tokenizer)
            dataloader = DataLoader(
                dataset,
                batch_size=BATCH_SIZE,
                shuffle=True,
                num_workers=num_workers,
                pin_memory=(DEVICE == "cuda")
            )

            optimizer = torch.optim.AdamW(train_model.parameters(), lr=LEARNING_RATE)
            loss_fn = torch.nn.CrossEntropyLoss()

            for epoch in range(EPOCHS):
                for images, text_tokens in dataloader:
                    images, text_tokens = images.to(DEVICE), text_tokens.to(DEVICE)
                    if text_tokens.dim() > 2:
                        text_tokens = text_tokens.squeeze()
                        if text_tokens.dim() == 1:
                            text_tokens = text_tokens.unsqueeze(0)

                    image_features = train_model.encode_image(images)
                    text_features = train_model.encode_text(text_tokens)
                    logits = image_features @ text_features.T
                    labels = torch.arange(len(images), device=DEVICE)
                    loss = loss_fn(logits, labels)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

            os.makedirs(os.path.dirname(MODEL_CHECKPOINT), exist_ok=True)
            torch.save(train_model.state_dict(), MODEL_CHECKPOINT)
            logger.info("Saved updated model checkpoint.")


            model.load_state_dict(torch.load(MODEL_CHECKPOINT, map_location=DEVICE))
            model.to(DEVICE).eval()


            state.last_trained_id = new_data_list[-1]
            state.last_trained_time = timezone.now()
            state.save()

            return True, {
                'message': f'Trained on {new_data.count()} new samples.',
                'last_trained_time': state.last_trained_time
            }

        except Exception as e:
            logger.error(f"Training error: {str(e)}")
            return False, f"Error training model: {str(e)}"
        


    @staticmethod
    def create_dataset_from_url(image_url: str, prompt: str = ''):
        try:
            response = requests.get(image_url, timeout=10)
            response.raise_for_status()

            if not response.headers.get('content-type', '').startswith('image/'):
                return False, 'URL does not point to an image'

            image_bytes = response.content
            Image.open(io.BytesIO(image_bytes)).verify()  # Validate it's an image
        except Exception as e:
            logger.warning(f"Image fetch/validate failed: {str(e)}")
            return False, 'Invalid or inaccessible image URL'

        filename = os.path.basename(image_url.split('?')[0]) or f"dataset_{timezone.now().strftime('%Y%m%d_%H%M%S')}.jpg"

        try:
            django_file = ContentFile(image_bytes, name=filename)
            dataset_entry = DataSet.objects.create(image=django_file, prompt=prompt)
            return True, {'message': 'Dataset entry created', 'id': dataset_entry.id}
        except Exception as e:
            logger.error(f"Failed to save dataset: {str(e)}")
            return False, f'Failed to save dataset: {str(e)}'
        

    @staticmethod
    def verify_image(image: Image.Image, prompt: str = '', threshold: float = 0.65):
        try:
            if os.path.exists(MODEL_CHECKPOINT):
                model.load_state_dict(torch.load(MODEL_CHECKPOINT, map_location=DEVICE))
                model.to(DEVICE).eval()
        except Exception as e:
            logger.error(f"Model load error: {str(e)}")
            return False, 0.0, 'Failed to load model'

        try:
            image_tensor = preprocess(image).unsqueeze(0).to(DEVICE)

            default_prompt = "a UK security officer wearing a crisp white dress shirt and black tie, with or without jacket, may have security badges and hi-vis"
            negative_prompt = "a person in casual clothes without white shirt and black tie: t-shirts, sweaters, jeans, hoodies, or informal attire"
        
            prompt = prompt or default_prompt

            text_inputs = tokenizer([prompt, negative_prompt]).to(DEVICE)

            with torch.no_grad():
                image_features = model.encode_image(image_tensor)
                text_features = model.encode_text(text_inputs)
                logits = image_features @ text_features.T
                probs = logits.softmax(dim=-1).cpu().numpy()[0]

            confidence = probs[0]
            is_valid = confidence >= threshold
            summary = "✅ Uniform detected." if is_valid else "❌ Uniform not detected."

            return is_valid, confidence, summary
        except Exception as e:
            logger.error(f"Inference error: {str(e)}")
            return False, 0.0, 'Error during model inference'
        
    @staticmethod
    def open_image_from_url(image_url: str):
        """Opens image from URL using a file-like object (no explicit download)"""
        try:
            with urlopen(image_url, timeout=10) as response:
                if 'image' not in response.headers.get('Content-Type', ''):
                    return False, None
                image = Image.open(response).convert('RGB')
                return True, image
        except Exception as e:
            logger.warning(f"Image open error from URL: {str(e)}")
            return False, None