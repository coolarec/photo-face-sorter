import os
import cv2
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from network import FaceClassifier

# OpenCV DNN 人脸检测模型路径
PROTO_PATH = '../deploy.prototxt'
MODEL_PATH = '../res10_300x300_ssd_iter_140000.caffemodel'

# 加载 DNN 人脸检测模型
net = cv2.dnn.readNetFromCaffe(PROTO_PATH, MODEL_PATH)

class FaceCropDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []  # [(图片路径, label), ...]
        self.class_to_idx = {}
        classes = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
        for i, cls in enumerate(classes):
            self.class_to_idx[cls] = i
            cls_dir = os.path.join(root_dir, cls)
            for fname in os.listdir(cls_dir):
                if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    self.samples.append((os.path.join(cls_dir, fname), i))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img_cv = cv2.imread(img_path)
        h, w = img_cv.shape[:2]

        # 人脸检测
        blob = cv2.dnn.blobFromImage(img_cv, 1.0, (300, 300), (104.0, 177.0, 123.0))
        net.setInput(blob)
        detections = net.forward()

        faces = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.5:
                box = detections[0, 0, i, 3:7] * [w, h, w, h]
                box = box.astype(int)
                top, right, bottom, left = box[1], box[2], box[3], box[0]
                # 限制边界
                top = max(0, top)
                left = max(0, left)
                bottom = min(h, bottom)
                right = min(w, right)
                faces.append((top, right, bottom, left))

        if faces:
            # 取第一个人脸裁剪
            top, right, bottom, left = faces[0]
            face_img_cv = img_cv[top:bottom, left:right]
            face_img = cv2.cvtColor(face_img_cv, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(face_img)
        else:
            # 没检测到人脸，退化处理，直接用原图转PIL
            pil_img = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))

        if self.transform:
            pil_img = self.transform(pil_img)

        return pil_img, label


transform_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

transform_val = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

def evaluate(model, dataloader, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for imgs, labels in dataloader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total if total else 0

def train():
    device = torch.device("cuda")

    full_dataset = FaceCropDataset("dataset", transform=transform_train)
    num_classes = len(full_dataset.class_to_idx)

    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # 验证集用val的transform
    val_dataset.dataset.transform = transform_val

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)

    model = FaceClassifier(num_classes=num_classes).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    epochs = 10
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        val_acc = evaluate(model, val_loader, device)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss / len(train_loader):.4f}, Val Acc: {val_acc:.4f}")

    torch.save(model.state_dict(), "model.pth")
    print("训练完成，模型已保存。")

if __name__ == "__main__":
    train()