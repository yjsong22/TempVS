import torch
from torch.nn import CosineSimilarity
from transformers import CLIPTokenizer, CLIPModel, CLIPTextModel, CLIPVisionModel, CLIPProcessor
from PIL import Image,  UnidentifiedImageError
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from torchvision.transforms import transforms, Compose, Resize, ToTensor, Normalize, Lambda

torch_device = "cuda:0" if torch.cuda.is_available() else "cpu"

models = [
    'openai/clip-vit-base-patch16',
    'openai/clip-vit-base-patch32',
    'openai/clip-vit-large-patch14',
]

model_id = models[1]

tokenizer = CLIPTokenizer.from_pretrained(model_id)
model = CLIPModel.from_pretrained(model_id).to(torch_device)
processor = CLIPProcessor.from_pretrained(model_id)

transform = Compose([
    Resize((224, 224)),
    Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),  # Ensure the image has 3 channels
    ToTensor(),
    Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
])
text_encoder = CLIPTextModel.from_pretrained(model_id).to(torch_device)
image_encoder = CLIPVisionModel.from_pretrained(model_id).to(torch_device)

# def calculate_clip_sim_with_features(img_path, text_input):
#     image_input = transform(Image.open(img_path)).unsqueeze(0).to(torch_device)
#     image_features = model.get_image_features(image_input)
    
#     text_inputs = tokenizer(text_input, padding="max_length", return_tensors="pt").to(torch_device)
#     text_features = model.get_text_features(**text_inputs)
    
#     image_features = image_features / image_features.norm(dim=-1, keepdim=True)
#     text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    
#     similarity = torch.nn.functional.cosine_similarity(image_features, text_features)
#     return similarity.item()

_text_features_cache = {}
def calculate_clip_sim_img_text(img_path, text_input):
    # Use cache for text features
    if text_input not in _text_features_cache:
        with torch.no_grad():
            text_inputs = tokenizer(text_input, padding="max_length", return_tensors="pt").to(torch_device)
            text_features = model.get_text_features(**text_inputs)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            _text_features_cache[text_input] = text_features
    
    text_features = _text_features_cache[text_input]
    
    with torch.no_grad():
        image_input = transform(Image.open(img_path)).unsqueeze(0).to(torch_device)
        image_features = model.get_image_features(image_input)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        similarity = torch.nn.functional.cosine_similarity(image_features, text_features)
        return similarity.item()
    
    
def calculate_clip_sim_img_img(image1_path, image2_path):
    try:
        image1 = Image.open(image1_path)
        image2 = Image.open(image2_path)
    except UnidentifiedImageError:
        return 1.0

    try:
        # Process the images and calculate cosine similarity
        inputs = processor(images=[image1, image2], return_tensors="pt", padding=True).to(torch_device)
        with torch.no_grad():
            image_features = model.get_image_features(**inputs)
        image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
        cosine_similarity = torch.nn.functional.cosine_similarity(image_features[0], image_features[1], dim=0)
        return cosine_similarity.item()
    except Exception as e:
        print(f"Error occurred during similarity calculation: {e}")
        # Optional: handle any other unexpected errors
        return 1.0