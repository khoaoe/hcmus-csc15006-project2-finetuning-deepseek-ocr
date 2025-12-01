import os
import json
import pandas as pd
from tqdm import tqdm

# BASE_DIR = "/kaggle/input/uit-hwdb/UT_HWDB_line" 
BASE_DIR = "/kaggle/input/uit-hwdb/UIT_HWDB_paragraph" 
OUTPUT_DIR = "/kaggle/working"

def convert_uit_structure_to_jsonl(root_path, output_filename):
    """
    Hàm này quét đệ quy cấu trúc thư mục UIT-HWDB và tạo file JSONL chuẩn cần cho fine-tuning
    """
    dataset_rows = []
    
    if not os.path.exists(root_path):
        print(f"Không tìm thấy đường dẫn: {root_path}")
        return

    # Duyệt qua các folder con (1, 2, 3, ..., 250, 251, ...)
    subfolders = [f for f in os.listdir(root_path) if os.path.isdir(os.path.join(root_path, f))]
    
    print(f"Đang xử lý {len(subfolders)} thư mục trong {root_path}...")
    
    for folder in tqdm(subfolders):
        folder_path = os.path.join(root_path, folder)
        label_file = os.path.join(folder_path, "label.json")
        
        # Chỉ xử lý nếu tìm thấy file label.json
        if os.path.exists(label_file):
            try:
                with open(label_file, 'r', encoding='utf-8') as f:
                    labels = json.load(f)
                
                # Duyệt qua từng ảnh trong file json
                for img_name, text_content in labels.items():
                    # Tạo đường dẫn tuyệt đối tới ảnh
                    img_full_path = os.path.join(folder_path, img_name)
                    
                    # Kiểm tra xem ảnh có thực sự tồn tại không
                    if os.path.exists(img_full_path):
                        dataset_rows.append({
                            "image": img_full_path, # Đường dẫn ảnh
                            "text": text_content    # Nhãn (Ground Truth)
                        })
            except Exception as e:
                print(f"Lỗi đọc file {label_file}: {e}")

    # Lưu ra file JSONL
    output_path = os.path.join(OUTPUT_DIR, output_filename)
    with open(output_path, 'w', encoding='utf-8') as f:
        for entry in dataset_rows:
            json.dump(entry, f, ensure_ascii=False)
            f.write('\n')
            
    print(f"--> Đã lưu {len(dataset_rows)} mẫu dữ liệu vào {output_path}")

# --- THỰC THI ---
# TRAIN SET
# train_path_line = os.path.join(BASE_DIR, "UIT_HWDB_line/train_data")
# convert_uit_structure_to_jsonl(train_path_line, "train_line.jsonl")

train_path_para = os.path.join(BASE_DIR, "UIT_HWDB_paragraph/train_data")
convert_uit_structure_to_jsonl(train_path_para, "train_paragraph.jsonl")

# TEST_SET
# test_path_line = os.path.join(BASE_DIR, "UIT_HWDB_line/test_data")
# convert_uit_structure_to_jsonl(test_path_line, "test_line.jsonl")

test_path_line = os.path.join(BASE_DIR, "UIT_HWDB_paragraph/test_data")
convert_uit_structure_to_jsonl(test_path_line, "test_paragraph.jsonl")