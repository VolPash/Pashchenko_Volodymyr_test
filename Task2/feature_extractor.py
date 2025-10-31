# feature_extractor.py
import torch
import numpy as np
import cv2
import sys
import os

# --- 1. Импорт Исходного Кода Моделей (файлы .py) ---
PROJECT_ROOT = os.path.dirname(__file__) 
SUPERGLUE_REPO_PATH = os.path.join(PROJECT_ROOT, "superglue_repo")
if SUPERGLUE_REPO_PATH not in sys.path:
     sys.path.append(SUPERGLUE_REPO_PATH)

try:
    from superpoint import SuperPoint
    from superglue import SuperGlue
except ImportError:
    print("КРИТИЧЕСКАЯ ОШИБКА: Не удалось импортировать SuperPoint/SuperGlue.")
    print(f"Убедитесь, что папка 'superglue_repo' существует здесь: {SUPERGLUE_REPO_PATH}")
    print("(Эта папка должна содержать 'superpoint.py' и 'superglue.py')")
    class SuperPoint(torch.nn.Module):
        def __init__(self, config={}): pass
        def forward(self, data): raise ImportError("SuperPoint не загружен")
    class SuperGlue(torch.nn.Module):
        def __init__(self, config={}): pass
        def forward(self, data): raise ImportError("SuperGlue не загружен")

# --- 2. Пути к Весам Моделей (файлы .pth) ---
SP_WEIGHTS_PATH = r"C:\Users\User\Desktop\Task2\models\superpoint_v1.pth"
SG_WEIGHTS_PATH = r"C:\Users\User\Desktop\Task2\models\superglue_outdoor.pth"
# ------------------------------------

# Глобальные переменные
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SUPERPOINT = None
SUPERGLUE = None

def frame2tensor(img_mono, device):
    img = img_mono.astype('float32') / 255.0
    tensor = torch.from_numpy(img)[None, None, :, :]  
    return tensor.to(device)

def initialize_models():
    """Инициализирует и загружает веса для SuperPoint и SuperGlue."""
    global SUPERPOINT, SUPERGLUE
    if SUPERPOINT is not None and SUPERGLUE is not None:
        return SUPERPOINT, SUPERGLUE

    print(f"Инициализация моделей на устройстве: {DEVICE}")

    if not os.path.exists(SP_WEIGHTS_PATH):
        raise FileNotFoundError(f"Веса SuperPoint не найдены по пути: {SP_WEIGHTS_PATH}")
    if not os.path.exists(SG_WEIGHTS_PATH):
        raise FileNotFoundError(f"Веса SuperGlue не найдены по пути: {SG_WEIGHTS_PATH}")

    # SuperPoint
    SP_CONFIG = {'nms_radius': 4, 'keypoint_threshold': 0.005, 'max_keypoints': 1024}
    superpoint_model = SuperPoint(SP_CONFIG)
    superpoint_model.load_state_dict(torch.load(SP_WEIGHTS_PATH, map_location=DEVICE))
    SUPERPOINT = superpoint_model.to(DEVICE).eval()

    # SuperGlue
    SG_CONFIG = {'weights': 'outdoor'}
    superglue_model = SuperGlue(SG_CONFIG)
    state = torch.load(SG_WEIGHTS_PATH, map_location=DEVICE)
    if isinstance(state, dict) and 'state_dict' in state:
        state = state['state_dict']
    superglue_model.load_state_dict(state, strict=False) 
    SUPERGLUE = superglue_model.to(DEVICE).eval()

    print("Модели SuperPoint и SuperGlue успешно загружены.")
    return SUPERPOINT, SUPERGLUE

def run_superpoint_superglue(img1_mono, img2_mono):
    """Прогоняет изображения через SuperPoint и SuperGlue."""
    # Обратите внимание: 'initialize_models' вызывается без аргумента 'device'
    sp_model, sg_model = initialize_models() 

    tensor1 = frame2tensor(img1_mono, DEVICE)
    tensor2 = frame2tensor(img2_mono, DEVICE)

    with torch.no_grad():
        pred1 = sp_model({'image': tensor1})
        pred2 = sp_model({'image': tensor2})

        kpts0_tensor = pred1.get('keypoints', [torch.empty(0, 2, device=DEVICE)])[0]
        kpts1_tensor = pred2.get('keypoints', [torch.empty(0, 2, device=DEVICE)])[0]
        scores0_tensor = pred1.get('scores', [torch.empty(0, device=DEVICE)])[0]
        scores1_tensor = pred2.get('scores', [torch.empty(0, device=DEVICE)])[0]
        desc0_tensor = pred1.get('descriptors', [torch.empty(1, 256, 0, device=DEVICE)])[0]
        desc1_tensor = pred2.get('descriptors', [torch.empty(1, 256, 0, device=DEVICE)])[0]

        if kpts0_tensor.shape[0] == 0 or kpts1_tensor.shape[0] == 0:
            print("Предупреждение: SuperPoint не нашел ключевых точек. Пропускаем SuperGlue.")
            matches_tensor = torch.empty(1, kpts0_tensor.shape[0] if kpts0_tensor.shape[0] > 0 else 1, dtype=torch.int64, device=DEVICE).fill_(-1)
        else:
            superglue_pred = sg_model({
                'keypoints0': kpts0_tensor.unsqueeze(0),
                'keypoints1': kpts1_tensor.unsqueeze(0),
                'descriptors0': desc0_tensor,
                'descriptors1': desc1_tensor,
                'scores0': scores0_tensor.unsqueeze(0),
                'scores1': scores1_tensor.unsqueeze(0),
                'image0': tensor1,
                'image1': tensor2
            })
            matches_tensor = superglue_pred['matches0']

    kpts0 = kpts0_tensor.cpu().numpy()
    kpts1 = kpts1_tensor.cpu().numpy()
    matches = matches_tensor[0].cpu().numpy()
    return kpts0, kpts1, matches


