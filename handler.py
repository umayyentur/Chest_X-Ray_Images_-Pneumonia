import io
import json 
import base64
from io import BytesIO

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

import runpod 

model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 2)

model.load_state_dict(torch.load('/Users/umayyentur/Desktop/C/Chest_X-Ray_Images_(Pneumonia)/pneumonia_model.pth' , map_location=torch.device('cpu'), weights_only=False))
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
]) 

class_name = ['NORMAL', 'PNEUMONIA']

def validate_input(job_input):
    if 'image' not in job_input:
        return None, 'please proivde an image'
    
    if isinstance(job_input, str):
        try:
            job_input = json.loads(job_input)
        except json.JSONDecodeError:
            return None, 'Invalid JSON format'
        
    image_data = job_input.get('image')
    
    if image_data is None:
        return None, 'please proivde an image'
    
    if not isinstance(image_data, str):
        return None, 'Image must be a base64 encoded string'
    
    return { 'image': image_data }, None
    
    
def handler(job):
    job_input = job['input']
    validate_data , error_massage = validate_input(job_input)
    
    if error_massage:
        return { 'error': error_massage }
    
    image_base64 = validate_data['image']
    
    try:
        image_bytes = base64.b64decode(image_base64)
        image = Image.open(BytesIO(image_bytes)).convert('RGB')
        image = transform(image).unsqueeze(0)
        
        with torch.no_grad():
            output = model(image)
            _, predicted = torch.max(output, 1)
            
            predicted_class = class_name[predicted.item()]
            
        return { 'predicted_class': predicted_class }
    
    except base64.binascii.Error:
        return { 'error': 'Invalid base64 string' }
    except io.IOError:
        return { 'error': 'Invalid image' }
    except Exception as e:
        return { 'error': str(e) }
        
        
if __name__ == '__main__':
    runpod.serverless.start({'handler': handler})