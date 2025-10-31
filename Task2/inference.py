# inference.py
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import warnings

# --- 1. Импорт ваших модулей ---
try:
    from data_loader import load_all_data, load_process_and_mask_image
    from feature_extractor import run_superpoint_superglue
except ModuleNotFoundError as e:
    print(f"КРИТИЧЕСКАЯ ОШИБКА: Не удалось импортировать модуль: {e}")
    print("Убедитесь, что файлы 'data_loader.py' и 'feature_extractor.py' находятся в той же папке, что и 'inference.py'")
    exit()

# --- 2. Конфигурация ---
OUTPUT_DIR = "results" 
os.makedirs(OUTPUT_DIR, exist_ok=True)
warnings.filterwarnings("ignore") 

# --- 3. Геометрическая проверка (RANSAC) ---
def run_ransac_and_assess_geometry(matched_kpts0, matched_kpts1):
    if len(matched_kpts0) >= 4:
        H, mask = cv2.findHomography(matched_kpts0, matched_kpts1, cv2.RANSAC, 5.0)
        inliers = mask.sum() if mask is not None else 0
        if inliers > 50: 
            match_assessment = f"ГЕОМЕТРИЧЕСКИ ОДИНАКОВЫ: Найдена стабильная Гомография с {inliers} inliers."
        else:
            match_assessment = f"ГЕОМЕТРИЧЕСКИ РАЗЛИЧНЫ: Найдено только {inliers} inliers."
    else:
        H, mask, inliers = None, None, 0
        match_assessment = "НЕДОСТАТОЧНО ТОЧЕК: Менее 4 соответствий для RANSAC."
    return H, mask, inliers, match_assessment

# --- 4. Визуализация ---
def visualize_matches(img1_vis, img2_vis, final_kpts0, final_kpts1, inliers, pair_index, date1, date2):
    H1, W1 = img1_vis.shape[:2]
    H2, W2 = img2_vis.shape[:2]
    H_max = max(H1, H2)
    W_total = W1 + W2

    combined_img = np.zeros((H_max, W_total, 3), dtype=np.uint8)
    combined_img[:H1, :W1] = img1_vis
    combined_img[:H2, W1:] = img2_vis

    plt.figure(figsize=(15, 8))
    plt.imshow(combined_img)
    plt.axis('off')
    plt.title(f"Пара {pair_index} ({date1} -> {date2}). Inliers RANSAC: {inliers}")

    for i in range(len(final_kpts0)):
        pt1 = final_kpts0[i].astype(int)
        pt2 = final_kpts1[i].astype(int)
        plt.plot([pt1[0], pt2[0] + W1], [pt1[1], pt2[1]], color='lime', alpha=0.5, linewidth=0.8)

    plt.scatter(final_kpts0[:, 0], final_kpts0[:, 1], s=5, c='red', marker='o')
    plt.scatter(final_kpts1[:, 0] + W1, final_kpts1[:, 1], s=5, c='red', marker='o')

    output_path = os.path.join(OUTPUT_DIR, f"superglue_matches_pair_{pair_index}_{date1}_{date2}.png")
    plt.savefig(output_path, bbox_inches='tight', dpi=200)
    plt.close() 
    print(f"Визуализация сохранена: {output_path}")

# --- 5. Главный Цикл Инференса ---
def main_inference_loop(df_pairs, df_labels):
    pairs_to_process = df_pairs.head(5) 
    print(f"Начинаем обработку {len(pairs_to_process)} пар.")
    
    for idx, row in pairs_to_process.iterrows():
        path1, path2 = row['path_src'], row['path_ref']
        date1, date2 = row['date_src'], row['date_ref']
        
        print(f"\n--- Обработка пары {idx+1}: {date1} -> {date2} ---")

        data1 = load_process_and_mask_image(path1, df_labels)
        data2 = load_process_and_mask_image(path2, df_labels)

        if data1 is None or data2 is None:
            print("Пропуск: Не удалось загрузить одно или оба изображения.")
            continue
            
        img1_mono, img1_vis = data1['img_mono'], data1['img_vis']
        img2_mono, img2_vis = data2['img_mono'], data2['img_vis']

        # Обратите внимание: 'run_superpoint_superglue' вызывается без 'device'
        kpts0, kpts1, matches = run_superpoint_superglue(img1_mono, img2_mono)
        
        if kpts0.shape[0] == 0 or kpts1.shape[0] == 0:
            continue

        valid_matches_mask = matches > -1
        matched_kpts0 = kpts0[valid_matches_mask]
        matched_kpts1 = kpts1[matches[valid_matches_mask]]

        H, mask, inliers, assessment = run_ransac_and_assess_geometry(matched_kpts0, matched_kpts1)
        print(f"Результат RANSAC: {assessment}")

        if inliers > 5:
            if mask is not None:
                inlier_mask = mask.flatten().astype(bool)
                final_kpts0 = matched_kpts0[inlier_mask]
                final_kpts1 = matched_kpts1[inlier_mask]
            else:
                final_kpts0, final_kpts1 = matched_kpts0, matched_kpts1
                
            visualize_matches(img1_vis, img2_vis, final_kpts0, final_kpts1, inliers, idx + 1, date1, date2)
        else:
             print("Пропуск визуализации: Слишком мало инлайеров.")

# --- 6. Точка входа ---
if __name__ == '__main__':
    print("=== ЗАПУСК КОНВЕЙЕРА SUPERGLUE ===")
    df_pairs, df_labels = load_all_data()
    
    if df_pairs.empty:
        print("КРИТИЧЕСКАЯ ОШИБКА: Не найдено пар изображений.")
        print(f"Проверьте, что BASE_DATASET_PATH в 'data_loader.py' указан верно.")
    else:
        main_inference_loop(df_pairs, df_labels)
        
    print("\n=== РАБОТА ЗАВЕРШЕНА ===")
    print(f"Проверьте папку '{OUTPUT_DIR}' для просмотра результатов.")
