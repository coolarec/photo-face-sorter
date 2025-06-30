import shutil
from flask import Flask, request, render_template, redirect
from PIL import Image
import cv2
import threading
import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model.network import FaceClassifier

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
RESULT_FOLDER = 'result'
DATASET_FOLDER = 'model/dataset'
MODEL_PATH = 'model/model.pth'

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)
os.makedirs(DATASET_FOLDER, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 载入分类类别
class_names = sorted([d for d in os.listdir(DATASET_FOLDER) if os.path.isdir(os.path.join(DATASET_FOLDER, d))])
model = FaceClassifier(num_classes=len(class_names))
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device).eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

# 载入 OpenCV DNN 人脸检测模型
net = cv2.dnn.readNetFromCaffe('deploy.prototxt', 'res10_300x300_ssd_iter_140000.caffemodel')

def detect_faces_opencv_dnn(img):
    h, w = img.shape[:2]
    blob = cv2.dnn.blobFromImage(img, 1.0, (300,300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()
    faces = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * [w, h, w, h]
            box = box.astype(int)
            # clip防止越界
            top, right, bottom, left = box[1], box[2], box[3], box[0]
            top = max(0, top)
            left = max(0, left)
            bottom = min(h, bottom)
            right = min(w, right)
            faces.append((top, right, bottom, left))
    return faces





transform_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

def incremental_train(new_name, new_img_path, epochs=30, batch_size=16):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 获取新的类别总数
    new_num_classes = len(os.listdir(DATASET_FOLDER))

    # 新建模型，类别数量更新
    model_inc = FaceClassifier(num_classes=new_num_classes).to(device)

    # 加载旧权重（排除最后一层）
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    filtered_dict = {
        k: v for k, v in checkpoint.items()
        if not k.startswith('backbone.fc.3.')  # 跳过最后一层参数
    }
    # 载入已有参数
    model_dict = model_inc.state_dict()
    model_dict.update(filtered_dict)
    model_inc.load_state_dict(model_dict, strict=False)

    # 冻结除最后一层以外的参数
    for name, param in model_inc.named_parameters():
        param.requires_grad = name.startswith('backbone.fc.3')

    # 定义优化器和损失函数
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model_inc.parameters()), lr=0.001
    )

    # 构建包含新类别的全量数据集
    full_dataset = datasets.ImageFolder(DATASET_FOLDER, transform=transform_train)
    train_loader = DataLoader(full_dataset, batch_size=batch_size, shuffle=True)

    # 增量训练
    model_inc.train()
    for epoch in range(epochs):
        total_loss = 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model_inc(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss / len(train_loader):.4f}")

    # 保存更新后的模型
    torch.save(model_inc.state_dict(), MODEL_PATH)
    print(f"✅ 增量训练完成，'{new_name}' 已添加并保存模型")


temp_face_path = None
temp_face_tensor = None  # 保存转换后的tensor，方便训练用




@app.route("/", methods=["GET", "POST"])
def index():
    global temp_face_path, temp_face_tensor

    result = None
    image_url = None
    need_name = False
    if request.method == "POST":
        if "file" in request.files:
            file = request.files["file"]
            if file.filename == "":
                return redirect(request.url)
            save_path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(save_path)
            image_url = save_path.replace("\\", "/")

            img_cv = cv2.imread(save_path)
            faces = detect_faces_opencv_dnn(img_cv)

            if not faces:
                result = "未检测到人脸"
            else:
                top, right, bottom, left = faces[0]
                face_img = img_cv[top:bottom, left:right]
                face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(face_rgb)
                input_tensor = transform(pil_img).unsqueeze(0).to(device)

                with torch.no_grad():
                    output = model(input_tensor)
                    probs = torch.softmax(output, dim=1)
                    conf, pred_idx = torch.max(probs, 1)
                    conf = conf.item()
                    pred_idx = pred_idx.item()

                if conf < 0.5:  # 置信度低，询问新名字
                    result = "识别置信度低，请输入新名字"
                    need_name = True
                    temp_face_path = save_path
                    temp_face_tensor = input_tensor.cpu()  # 这里保留tensor，方便后续训练
                else:
                    pred_name = class_names[pred_idx]
                    result = f"识别为：{pred_name}"
                    # 保存识别图片
                    person_dir = os.path.join(RESULT_FOLDER, pred_name)
                    os.makedirs(person_dir, exist_ok=True)
                    shutil.copy(save_path, os.path.join(person_dir, file.filename))

        elif "new_name" in request.form:
            new_name = request.form["new_name"].strip()
            if new_name and temp_face_path and temp_face_tensor is not None:
                # 保存图片到新类别文件夹
                person_dir = os.path.join(DATASET_FOLDER, new_name)
                os.makedirs(person_dir, exist_ok=True)
                new_img_path = os.path.join(person_dir, os.path.basename(temp_face_path))
                shutil.copy(temp_face_path, new_img_path)

                # 更新类别列表（简单追加，后续需重载模型）
                if new_name not in class_names:
                    class_names.append(new_name)

                # 将新数据增量训练（示例同步调用，建议改后台线程）
                def train_new():
                    incremental_train(new_name, new_img_path)
                    # 增量训练完成后重新加载模型
                    global model, class_names
                    class_names = sorted(
                        [d for d in os.listdir(DATASET_FOLDER) if os.path.isdir(os.path.join(DATASET_FOLDER, d))])
                    new_model = FaceClassifier(num_classes=len(class_names))
                    new_model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
                    new_model.to(device).eval()
                    model = new_model
                    print("✅ 模型已重新加载")

                threading.Thread(target=train_new).start()

                result = f"新用户 {new_name} 数据已保存，模型正在更新中..."
                temp_face_path = None
                temp_face_tensor = None

    return render_template("index.html", result=result, image=image_url, need_name=need_name)

if __name__ == "__main__":
    app.run(debug=True)
