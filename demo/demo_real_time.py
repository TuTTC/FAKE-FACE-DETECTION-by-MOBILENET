import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
import timm
import cv2
import numpy as np
from PIL import Image
from streamlit_webrtc import webrtc_streamer
import av
import os
from collections import OrderedDict


# --- Device ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Hàm load riêng cho MobileNetV2 ---
def load_mobilenetv2_custom(weight_path, device):
    model = models.mobilenet_v2(pretrained=True)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.to(device)
    model.eval()
    return model

# --- Hàm load riêng cho MobileNetV3 ---
def load_mobilenetv3_custom(weight_path, device):
    model = models.mobilenet_v3_small(pretrained=True)

    # Tìm lớp Linear cuối cùng trong classifier
    for i in reversed(range(len(model.classifier))):
        if isinstance(model.classifier[i], nn.Linear):
            in_features = model.classifier[i].in_features
            model.classifier[i] = nn.Linear(in_features, 2)
            break
    else:
        raise ValueError("Không tìm thấy lớp Linear trong classifier của MobileNetV3.")

    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.to(device)
    model.eval()
    return model



# --- Hàm load riêng cho RepVGG-B0 và RepVGG-A0 từ timm ---
def load_repvggb0_custom(weight_path, device):
    model = timm.create_model("repvgg_b0", pretrained=True)
    in_features = model.get_classifier().in_features
    model.reset_classifier(num_classes=2)
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def load_repvgga0_custom(weight_path, device):
    # Tạo model chuẩn
    model = timm.create_model("repvgg_a0", pretrained=True)
    model.reset_classifier(num_classes=2)

    # Load checkpoint
    checkpoint = torch.load(weight_path, map_location=device)

    # Nếu checkpoint là dict chứa 'model' thì lấy state_dict đó
    if isinstance(checkpoint, dict) and 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint

    # Xử lý prefix 'module.' hoặc 'model.' trong keys (nếu có)
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith("module."):
            new_key = k[len("module."):]
        elif k.startswith("model."):
            new_key = k[len("model."):]
        else:
            new_key = k
        new_state_dict[new_key] = v

    # Load state_dict với strict=False để bỏ qua lỗi nhỏ nếu có
    model.load_state_dict(new_state_dict, strict=False)

    model.to(device)
    model.eval()
    return model

# --- Hàm load riêng cho ResNet18 và ResNet50 từ timm ---
def remove_module_prefix(state_dict):
    new_state = OrderedDict()
    for k, v in state_dict.items():
        name = k.replace("module.", "")  # remove `module.` if exists
        new_state[name] = v
    return new_state

def load_resnet18_custom(weight_path, device):
    model = models.resnet18(weights=None)  # Không load pre-trained từ ImageNet
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 2)

    # Load checkpoint và xử lý nếu có 'module.' prefix
    state_dict = torch.load(weight_path, map_location=device)
    state_dict = remove_module_prefix(state_dict)
    model.load_state_dict(state_dict)

    model.to(device)
    model.eval()
    return model


def load_resnet50_custom(weight_path, device):
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 2)
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.to(device)
    model.eval()
    return model

# --- Hàm load chung cho các model timm khác ---
def load_timm_model(weight_path, model_arch, device):
    model = timm.create_model(model_arch, pretrained=True)
    if hasattr(model, 'classifier'):
        model.classifier = nn.Linear(model.classifier.in_features, 2)
    elif hasattr(model, 'fc'):
        model.fc = nn.Linear(model.fc.in_features, 2)
    else:
        raise ValueError("Không tìm thấy lớp phân loại để sửa.")
    # model.load_state_dict(torch.load(weight_path, map_location=device))
    # model.to(device)
    # model.eval()
    # return model
    state_dict = torch.load(weight_path, map_location="cpu")
    new_state = OrderedDict((k.replace("module.", ""), v) for k, v in state_dict.items())
    model.load_state_dict(new_state)
    model.eval()
    return model

# --- Model options ---
model_options = {
    "MobileNetV2": ("mobilenetv2_trainv2.pth", "torchvision"),
    "MobileNetV3-Teacher": ("mobilenetv3_teacher.pth","mobilenetv3_custom"),
    "MobileNetV4Small": ("mobilenetv4small_trainv2.pth", "mobilenetv4_conv_small.e2400_r224_in1k"),
    "ResNet18-Student": ("resnet18_student_train_with_teachertrainv2.pth", "resnet18_custom"),
    "ResNet50-Teacher": ("RESNET50-TEACHER-TRAINV2.pth", "resnet50_custom"),
    "RepVGG-A0-Student": ("repvgga0_student_trainv2_with_teacherv2.pth", "repvgga0_custom"),
    "RepVGG-B0-Teacher": ("REPVGGB0-TEACHER-TRAINV2.pth", "repvggb0_custom"),
    "VGG11-Student": ("",""),
    "VGG16-Teacher": ("","")

}

# --- Sidebar Configuration ---
st.set_page_config(layout="wide")
st.sidebar.title("⚙️ Cấu hình")

# --- Sidebar Configuration ---
selected_model_names = st.sidebar.multiselect(
    "🧠 Chọn mô hình nhận diện khuôn mặt fake/real",
    list(model_options.keys())
)

mode = st.sidebar.radio(
    "📸 Chế độ nhận diện",
    ["🖼️ Ảnh tải lên", "🎥 Webcam realtime", "📹 Video", "📷 Ảnh nâng cao"]
)

# --- Cached model loader ---
@st.cache_resource
def load_model(model_name, path, arch):
    if arch == "torchvision":
        return load_mobilenetv2_custom(path, device)
    elif arch == "repvgga0_custom":
        return load_repvgga0_custom(path, device)
    elif arch == "repvggb0_custom":
        return load_repvggb0_custom(path, device)
    elif arch == "resnet18_custom":
        return load_resnet18_custom(path, device)
    elif arch == "resnet50_custom":
        return load_resnet50_custom(path, device)
    elif arch == "mobilenetv3_custom":
        return load_mobilenetv3_custom(path, device)
    else:
        return load_timm_model(path, arch, device)

# --- Load selected models ---
loaded_models = []
for name in selected_model_names:
    weight_path, model_arch = model_options[name]
    if weight_path:  # Chỉ load nếu có path
        model_path = os.path.join("models", weight_path)
        try:
            model = load_model(name, model_path, model_arch)
            loaded_models.append((name, model))
            st.sidebar.success(f"✅ Loaded: {name}")
        except Exception as e:
            st.sidebar.error(f"❌ Không thể load {name}: {e}")
    else:
        st.sidebar.warning(f"⚠️ {name} chưa có trọng số!")
models_dict = {name: model for name, model in loaded_models}
# --- Face detector ---
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

# --- Preprocessing ---
preprocess_from_pil = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406),
                         std=(0.229, 0.224, 0.225))
])

def preprocess_from_cv2(img):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    tensor = preprocess_from_pil(pil_img)
    tensor = tensor.unsqueeze(0).to(device)
    return tensor

def predict(tensor):
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)[0]
        label = torch.argmax(probs).item()
        conf = probs[label].item()
    return label, conf

# === Preprocessing ===
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

def detect_faces(pil_img):
    img_np = np.array(pil_img.convert("RGB"))
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    results = []
    for (x, y, w, h) in faces:
        face_crop = pil_img.crop((x, y, x+w, y+h))
        results.append(((x, y, x+w, y+h), face_crop))
    return results

def predict_multi_models(face_img_pil, models: dict):
    input_tensor = transform(face_img_pil).unsqueeze(0)  # shape: [1, 3, 224, 224]

    results = {}
    with torch.no_grad():
        for name, model in models.items():
            output = model(input_tensor)
            probs = torch.softmax(output, dim=1)[0].cpu().numpy()
            label_idx = np.argmax(probs)
            label = "Real" if label_idx == 1 else "Fake"

            results[name] = {
                "label": label,
                "confidence": float(probs[label_idx]),
                "softmax": [float(probs[0]), float(probs[1])]
            }
    return results

def extract_clip_to_temp(video_path, frame_index, fps, duration_sec=2, output_dir="temp_clips"):
    """
    Trích đoạn video ngắn quanh frame_index từ video gốc.
    - video_path: đường dẫn đến video gốc
    - frame_index: frame bắt đầu trích
    - fps: số frame/giây của video
    - duration_sec: độ dài clip tính bằng giây
    - output_dir: thư mục chứa clip trích
    """
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Không mở được video.")
        return None

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)

    # Chuẩn bị thông tin clip
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    clip_fps = int(fps)
    total_clip_frames = int(duration_sec * clip_fps)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    clip_filename = f"clip_f{frame_index}.mp4"
    clip_path = os.path.join(output_dir, clip_filename)

    out = cv2.VideoWriter(clip_path, fourcc, clip_fps, (width, height))

    saved = 0
    while saved < total_clip_frames:
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)
        saved += 1

    cap.release()
    out.release()
    return clip_path

def extract_frames_opencv(video_path, output_dir, num_frames=10):
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("Không thể mở video.")
        return
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, total_frames // num_frames)
    
    os.makedirs(output_dir, exist_ok=True)
    saved = 0
    current = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        if current % step == 0 and saved < num_frames:
            frame_path = os.path.join(output_dir, f"frame_{saved:03d}.jpg")
            cv2.imwrite(frame_path, frame)
            saved += 1
        
        current += 1

    cap.release()
    print(f"Đã lưu {saved} frame vào {output_dir}")

# --- Giao diện chính ---
st.title("🧠 Fake Face Detection Web Demo")

if mode == "🖼️ Ảnh tải lên":
    uploaded_file = st.file_uploader("📤 Tải ảnh khuôn mặt lên", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Ảnh đã tải lên", width = 300)

        if not selected_model_names:
            st.warning("⚠️ Vui lòng chọn ít nhất một mô hình.")
        else:
            # --- Load các mô hình đã chọn ---
            loaded_models = {}
            for name in selected_model_names:
                weight_path, model_arch = model_options[name]
                model_path = os.path.join("models", weight_path)
                loaded_models[name] = load_model(name, model_path, model_arch)

            # --- Dự đoán ---
            results = predict_multi_models(image, loaded_models)
            
            # --- Hiển thị kết quả ---
            st.markdown("### 🔍 Kết quả dự đoán:")
            for model_name, res in results.items():
                label = res["label"]
                conf = res["confidence"]
                st.markdown(f"- **{model_name}** → {label} ({conf:.2f})")

elif mode == "🎥 Webcam realtime":
    st.info("Webcam realtime với streamlit-webrtc")

    models_dict = {name: model for name, model in loaded_models}

    if selected_model_names:
        # Vùng để hiển thị kết quả dự đoán dưới video
        result_placeholder = st.empty()

        class VideoProcessor:
            def __init__(self):
                self.frame_count = 0
                self.last_results = []
                self.result_text = ""

            def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
                self.frame_count += 1
                img = frame.to_ndarray(format="bgr24")
                img = cv2.flip(img, 1)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, 1.3, 5)

                if self.frame_count % 20 == 0:
                    self.last_results = []
                    self.result_text = ""  # reset

                    for (x, y, w, h) in faces:
                        face_crop = Image.fromarray(cv2.cvtColor(img[y:y+h, x:x+w], cv2.COLOR_BGR2RGB))
                        face_crop = face_crop.resize((128, 128))

                        results = predict_multi_models(face_crop, models_dict)

                        self.last_results.append({
                            "bbox": (x, y, w, h),
                            "results": results
                        })

                        for model_name, res in results.items():
                            label = res["label"]
                            conf = res["confidence"]
                            self.result_text += f"**{model_name}** → `{label}` ({conf:.2f})  \n"

                for item in self.last_results:
                    x, y, w, h = item["bbox"]
                    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 255), 2)
                    for i, (model_name, res) in enumerate(item["results"].items()):
                        label = res["label"]
                        conf = res["confidence"]
                        color = (0, 0, 255) if label == "Fake" else (0, 255, 0)
                        text = f"{model_name[:10]}: {label} ({conf:.2f})"
                        cv2.putText(img, text, (x, y - 10 - 20 * i),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

                return av.VideoFrame.from_ndarray(img, format="bgr24")

        # Khởi tạo video processor và lấy state
        ctx = webrtc_streamer(
            key="webcam",
            video_processor_factory=VideoProcessor,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True
        )

        # Cập nhật kết quả bên ngoài recv()
        if ctx.video_processor:
            result_placeholder.markdown("### 🔍 Kết quả Realtime\n" + ctx.video_processor.result_text)

    else:
        st.warning("⚠️ Vui lòng chọn ít nhất một mô hình để chạy webcam realtime.")


# elif mode == "🎥 Webcam realtime":
#     st.info("Webcam realtime với xử lý tương tự video — multi-model")

#     def preprocess_from_cv2(img_cv):
#         pil_image = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
#         return transform(pil_image).unsqueeze(0)

#     class VideoProcessor:
#         def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
#             img = frame.to_ndarray(format="bgr24")
#             img = cv2.flip(img, 1)  # Lật để giống gương
#             gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#             faces = face_cascade.detectMultiScale(gray, 1.3, 5)

#             for (x, y, w, h) in faces:
#                 face_crop = img[y:y+h, x:x+w]
#                 pil_face = Image.fromarray(cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB))
#                 results = predict_multi_models(pil_face, models)

#                 # Chọn model đầu tiên để vẽ nhãn
#                 main_model = list(results.keys())[0]
#                 main_pred = results[main_model]
#                 label_str = f"{main_pred['label']} ({main_pred['confidence']:.2f})"
#                 color = (0, 255, 0) if main_pred['label'] == "Real" else (0, 0, 255)

#                 cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
#                 cv2.putText(img, f"{main_model}: {label_str}", (x, y-10),
#                             cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

#             return av.VideoFrame.from_ndarray(img, format="bgr24")

#     webrtc_streamer(key="webcam", video_processor_factory=VideoProcessor)


# === ẢNH TĨNH (NÂNG CAO) ===
elif mode == "📷 Ảnh nâng cao":
    uploaded_image = st.file_uploader("📤 Tải ảnh khuôn mặt lên", type=["jpg", "jpeg", "png"])
    
    if uploaded_image:
        image = Image.open(uploaded_image).convert("RGB")
        st.image(image, caption="📷 Ảnh gốc", use_column_width=True)

        results = detect_faces(image)
        st.subheader(f"🔍 Phát hiện {len(results)} khuôn mặt")

        if len(results) == 0:
            st.warning("Không phát hiện khuôn mặt nào trong ảnh!")
        else:
            for idx, (box, face) in enumerate(results):
                st.markdown(f"### 🧩 Gương mặt {idx+1}")
                cols = st.columns(2)
                cols[0].image(face, caption="Khuôn mặt", width=120)

                # Resize face (nếu cần) và dự đoán với nhiều mô hình
                resized_face = face.resize((128, 128))
                results = predict_multi_models(resized_face, models_dict)

                # Hiển thị kết quả từng model
                with cols[1]:
                    st.markdown("**Kết quả từ các mô hình:**")
                    for model_name, res in results.items():
                        label = res["label"]
                        conf = res["confidence"]
                        st.markdown(f"- **{model_name}** → `{label}` ({conf:.2f})")



# === VIDEO ===
elif mode == "📹 Video":
    uploaded_video = st.file_uploader("📂 Tải video lên", type=["mp4", "avi", "mov"])
    # models_dict = {name: model for name, model in loaded_models}
    if uploaded_video is not None:
        os.makedirs("temp", exist_ok=True)
        temp_path = os.path.join("temp", uploaded_video.name)
        with open(temp_path, "wb") as f:
            f.write(uploaded_video.read())

        st.video(temp_path)

        cap = cv2.VideoCapture(temp_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        num_to_extract = 10
        interval = max(1, total_frames // num_to_extract)

        st.info(f"🎞️ Video có {total_frames} frames, sẽ trích {num_to_extract} frame cách đều mỗi {interval} frames.")

        for i in range(num_to_extract):
            frame_index = i * interval
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ret, frame = cap.read()
            if not ret:
                continue

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(frame_rgb)
            faces = detect_faces(pil_img)
            if faces:
                st.subheader(f"📍 Frame {frame_index} - {len(faces)} gương mặt")

                for idx, (box, face) in enumerate(faces):
                    st.markdown(f"### 🧩 Gương mặt {idx+1}")
                    cols = st.columns(2)
                    cols[0].image(face, caption="Face", width=100)

                    # Dự đoán nhiều mô hình
                    resized_face = face.resize((128, 128))
                    results = predict_multi_models(resized_face, models_dict)

                    # Hiển thị kết quả từ các mô hình
                    with cols[1]:
                        st.markdown(f"**Kết quả từ các mô hình:**")
                        for model_name, res in results.items():
                            label = res["label"]
                            conf = res["confidence"]
                            st.markdown(f"- **{model_name}** → `{label}` ({conf:.2f})")

        cap.release()
    else:
        st.warning("⚠️ Vui lòng tải lên một video để bắt đầu.")


