import cv2
import numpy as np
import time
import os

# 嘗試導入 tflite_runtime，如果失敗則使用 tensorflow
try:
    from tflite_runtime.interpreter import Interpreter
except ImportError:
    import tensorflow as tf
    Interpreter = tf.lite.Interpreter

# --- 1. 設定參數 ---
MODEL_PATH = "your_tf_model.tflite"  # 你的 TFLite 模型
CASCADE_PATH = "your_haarcascade.xml"  # OpenCV Haar Cascade
INPUT_SHAPE = (160, 160)  # 模型輸入尺寸
SIMILARITY_THRESHOLD = 0.8  # 餘弦相似度閾值
DOOR_COOLDOWN = 5.0  # 開門冷卻時間（秒）
WHITELIST_DIR = "whitelist_embeddings"  # 白名單嵌入儲存目錄

# --- 2. 載入模型 ---
interpreter = Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# 驗證輸出形狀是否為嵌入向量
assert output_details[0]['shape'].tolist() == [1, 512], "模型輸出不是 512 維嵌入向量"

# 載入人臉偵測分類器
face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
if face_cascade.empty():
    print("錯誤：無法載入 Haar Cascade 分類器")
    exit()

# --- 3. 輔助函式 ---
def preprocess_face(face_img, target_size):
    """
    對人臉圖片進行預處理
    """
    face_img = cv2.resize(face_img, target_size)
    face_img = face_img.astype(np.float32)
    face_img = (face_img - 127.5) / 127.5  # 正規化到 [-1, 1]
    face_img = np.expand_dims(face_img, axis=0)
    face_img = np.transpose(face_img, (0, 3, 1, 2))  # [1, 3, 160, 160]
    return face_img

def get_embedding(face_img):
    """
    使用 TFLite 模型提取嵌入向量
    """
    processed_face = preprocess_face(face_img, INPUT_SHAPE)
    interpreter.set_tensor(input_details[0]['index'], processed_face)
    interpreter.invoke()
    embedding = interpreter.get_tensor(output_details[0]['index'])[0]  # 形狀 [512]
    return embedding

def save_whitelist_embedding(image_path, person_id, output_dir=WHITELIST_DIR):
    """
    從圖像生成並儲存白名單嵌入
    """
    os.makedirs(output_dir, exist_ok=True)
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"無法讀取圖像：{image_path}")
        return
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))
    if len(faces) == 0:
        print(f"未偵測到人臉：{image_path}")
        return
    x, y, w, h = faces[0]  # 取第一張人臉
    face_roi = frame[y:y+h, x:x+w]
    embedding = get_embedding(face_roi)
    np.save(os.path.join(output_dir, f"{person_id}.npy"), embedding)
    print(f"已儲存 {person_id} 的嵌入向量")

def extract_frame_from_video(video_path, output_image_path):
    """
    從影片中提取一幀作為白名單圖像
    """
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    if ret:
        cv2.imwrite(output_image_path, frame)
        print(f"已從 {video_path} 提取圖像到 {output_image_path}")
    else:
        print(f"無法從 {video_path} 提取圖像")
    cap.release()

def is_whitelist_match(face_roi, whitelist_dir=WHITELIST_DIR, threshold=SIMILARITY_THRESHOLD):
    """
    與白名單嵌入進行餘弦相似度比較
    """
    test_embedding = get_embedding(face_roi)
    for person_file in os.listdir(whitelist_dir):
        if person_file.endswith(".npy"):
            whitelist_embedding = np.load(os.path.join(whitelist_dir, person_file))
            similarity = np.dot(test_embedding, whitelist_embedding) / (
                np.linalg.norm(test_embedding) * np.linalg.norm(whitelist_embedding)
            )
            if similarity > threshold:
                return True, similarity, person_file.split(".")[0]
    return False, 0.0, None

def open_door():
    """
    模擬開門（可替換為 Coral Dev Board 的 GPIO 控制）
    """
    print("Welcome! Door open!")
    # 範例 GPIO 控制（取消註解以使用）
    # import RPi.GPIO as GPIO
    # GPIO.setmode(GPIO.BCM)
    # GPIO.setup(18, GPIO.OUT)
    # GPIO.output(18, GPIO.HIGH)
    # time.sleep(1)
    # GPIO.output(18, GPIO.LOW)
    # GPIO.cleanup()

# --- 4. 主迴圈 ---

video_path = "whitelist_video.mp4"
image_path = "whitelist_frame.jpg"
person_id = "person_X"
extract_frame_from_video(video_path, image_path)
save_whitelist_embedding(image_path, person_id)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("錯誤: 無法開啟攝影機。")
    exit()

last_unlock_time = 0
print("系統啟動，開始偵測...")

while True:
    ret, frame = cap.read()
    if not ret:
        print("無法讀取影像，結束程式。")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))

    for (x, y, w, h) in faces:
        face_roi = frame[y:y+h, x:x+w]
        is_match, similarity, person_id = is_whitelist_match(face_roi)
        current_time = time.time()
        if is_match:
            label = f"Welcome {person_id}! ({similarity:.2f})"
            color = (0, 255, 0)  # 綠色
            if current_time - last_unlock_time > DOOR_COOLDOWN:
                open_door()
                last_unlock_time = current_time
        else:
            label = f"Access Denied ({similarity:.2f})"
            color = (0, 0, 255)  # 紅色

        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    cv2.imshow('Face Whitelist System', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --- 5. 清理資源 ---
print("正在關閉系統...")
cap.release()
cv2.destroyAllWindows()