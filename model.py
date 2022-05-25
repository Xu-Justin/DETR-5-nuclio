import torch
from torchvision import transforms

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

CLASSES = [
    'N/A',
    'car',
    'bus',
    'truck',
    'motorcycle',
    'pedestrian'
]

def get_model():
    custom_model = torch.load('DETR.model', map_location=device)
    return custom_model.to(device)

def get_result(custom_model, image):
    transform = transforms.Compose([
        transforms.Resize(800),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    width, height = image.size
    image = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        raw_result = custom_model(image)
    
    encoded_result = []
    for logits, boxes in zip(raw_result['pred_logits'][0], raw_result['pred_boxes'][0]):
        logits = logits.cpu()
        boxes = boxes.cpu()
        
        index = int(logits.argmax(-1))
        if(index<=0 or index >= len(CLASSES)): 
            # encoded_result.append({'none'})
            continue
        
        label = CLASSES[index]
        confidence = float(logits.softmax(-1)[index])
        
        x, y, w, h = boxes * torch.tensor([width, height, width, height], dtype=torch.float32)
        xmin = float(x - torch.div(w, 2))
        xmax = float(x + torch.div(w, 2))
        ymin = float(y - torch.div(h, 2))
        ymax = float(y + torch.div(h, 2))
        
        encoded_result.append({
            "confidence" : confidence,
            "label" : label,
            "points" : [xmin, ymin, xmax, ymax],
            "type" : "rectangle"
        })
        
    return encoded_result