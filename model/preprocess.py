import os
from PIL import Image
from facenet_pytorch import MTCNN
import torch

DATASET_DIR = 'dataset'
OUTPUT_DIR = 'dataset_cropped'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mtcnn = MTCNN(keep_all=True, device=device)

def preprocess():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for cls in os.listdir(DATASET_DIR):
        cls_path = os.path.join(DATASET_DIR, cls)
        if not os.path.isdir(cls_path):
            continue
        out_cls_path = os.path.join(OUTPUT_DIR, cls)
        os.makedirs(out_cls_path, exist_ok=True)

        for img_name in os.listdir(cls_path):
            if not img_name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                continue
            img_path = os.path.join(cls_path, img_name)
            img = Image.open(img_path).convert('RGB')

            # MTCNN检测人脸 bbox，返回PIL裁剪图列表
            faces = mtcnn(img)
            if faces is None:
                print(f"[WARN] {img_path} 未检测到人脸，保存原图")
                img.save(os.path.join(out_cls_path, img_name))
                continue

            if isinstance(faces, torch.Tensor):
                # faces可能是单张脸，形状(N, C, H, W) 但N=1
                if faces.dim() == 4 and faces.size(0) == 1:
                    faces = [faces[0]]  # 变成list，单张脸tensor
                else:
                    faces = list(faces)  # 多张脸，转成list

            for i, face in enumerate(faces):
                # face: (C, H, W), float tensor in [0, 1]
                face_img = (face * 255).permute(1, 2, 0).byte().cpu().numpy()
                face_pil = Image.fromarray(face_img)

                save_name = img_name if len(
                    faces) == 1 else f"{os.path.splitext(img_name)[0]}_{i}{os.path.splitext(img_name)[1]}"
                face_pil.save(os.path.join(out_cls_path, save_name))
                print(f"保存裁剪人脸: {os.path.join(out_cls_path, save_name)}")


if __name__ == '__main__':
    preprocess()
