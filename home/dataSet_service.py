from venv import logger
import requests
from PIL import Image
import io
import torch
from torch.utils.data import Dataset

# class PromptImageDataset(Dataset):
#     def __init__(self, queryset, preprocess, tokenizer):
#         self.data = list(queryset)
#         self.preprocess = preprocess
#         self.tokenizer = tokenizer

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, idx):
#         record = self.data[idx]

#         # Load and process image
#         response = requests.get(record.image.url, verify=False)
#         image = Image.open(io.BytesIO(response.content)).convert('RGB')
#         image_tensor = self.preprocess(image)

#         # Tokenize the prompt with SimpleTokenizer
#         MAX_LEN = 77
#         tokens = self.tokenizer.encode(record.prompt)
#         if len(tokens) < MAX_LEN:
#             tokens += [0] * (MAX_LEN - len(tokens))
#         else:
#             tokens = tokens[:MAX_LEN]

#         prompt_tensor = torch.tensor(tokens, dtype=torch.long)

#         return image_tensor, prompt_tensor

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