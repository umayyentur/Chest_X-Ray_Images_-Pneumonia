import torch
import torchvision.models as models
import torchvision.transforms as transforms
import base64
from flask import Flask, render_template, request
from PIL import Image
import io

app = Flask(__name__)

# Model tanımlama
model = models.resnet18(pretrained=False)
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, 2)  # 2 sınıf: Normal / Pnömoni

# Ağırlıkları yükle
model.load_state_dict(torch.load("pneumonia_model.pth", map_location=torch.device("cpu")))
model.eval()

# Görsel işleme adımları
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files.get("image")
        if not file:
            return "Lütfen bir resim dosyası yükleyin", 400
        
        # 1) Dosya içeriğini okuyup sakla
        file_data = file.read()

        # 2) Model için PIL Image dönüştür
        image = Image.open(io.BytesIO(file_data)).convert("RGB")
        image_tensor = transform(image).unsqueeze(0)

        # 3) Tahmin al
        with torch.no_grad():
            output = model(image_tensor)
            prediction = torch.argmax(output, dim=1).item()

        # 4) İnsan okunabilir metin
        result = "Pneumonia detected!" if prediction == 1 else "Normal Lung Image"

        # 5) Görseli Base64'e çevirip şablona gönder
        image_data = base64.b64encode(file_data).decode("utf-8")
        image_url = f"data:image/jpeg;base64,{image_data}"

        return render_template(
            "index.html",
            original_img=image_url,  # Şablonda 'original_img' olarak kullanacağız
            prediction=result
        )

    # GET ise sadece formu göster
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8000)
