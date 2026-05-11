import asyncio
import cv2
import datetime
import google.generativeai as genai
import json
import math
import numpy as np
import os
import subprocess
import time
import torch
import torchreid
from collections import deque, Counter
from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile, Query, BackgroundTasks, Depends, Request, Form, HTTPException, Response
from fastapi import WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.responses import HTMLResponse
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from numpy.linalg import norm
from passlib.context import CryptContext
from pydantic import BaseModel
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Session
from sqlalchemy.orm import sessionmaker
from torchvision import transforms
from ultralytics import YOLO

# ReID 모델 로드osnet 및 전처리 설정
reid_device = 'cuda' if torch.cuda.is_available() else 'cpu'
reid_model = torchreid.models.build_model(name='osnet_x1_0', num_classes=1000, pretrained=True).to(reid_device)
reid_model.eval()

reid_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((256, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])





# 현재 폴더의 .env 읽기
load_dotenv()

# 신고 내역에서 자연어 색상 파싱을 위한 api설정 (gemini2.5)
API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=API_KEY)
llm_model = genai.GenerativeModel('models/gemini-2.5-flash')



# --- [데이터베이스 설정 구간] ---
DB_USER = os.getenv("DB_USER")
DB_PW = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST")
DB_NAME = os.getenv("DB_NAME")

DATABASE_URL = f"mysql+pymysql://{DB_USER}:{DB_PW}@{DB_HOST}:3306/{DB_NAME}?charset=utf8mb4"
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# 비밀번호 암호화 설정
pwd_context = CryptContext(schemes=["pbkdf2_sha256"], deprecated="auto")

# 회원가입 시 받을 데이터 규격 (일반user)
class UserCreate(BaseModel):
    id: str
    password: str
    residentFront: str
    residentBack: str
    name: str
    phone: str

# 관리자 회원가입 시 데이터 규격 (admin)
class AdminSignupRequest(BaseModel):
    id: str
    password: str
    name: str
    phone: str
    residentFront: str
    residentBack: str
    orgCode: str

# 앱에서 서버로 신고 정보를 보낼 때의 형식
class ReportCreate(BaseModel):
    name: str
    phone_number: str
    ssn: str
    content: str
    location: str # 위치 정보 필드 추가

# 소켓 연결 관리자(WebSocket)
class ConnectionManager:
    def __init__(self):
        self.active_connections: dict[int, list[WebSocket]] = {}
        self.log_history: dict[int, list[str]] = {} # 로그 보관함

    async def connect(self, report_id: int, websocket: WebSocket):
        await websocket.accept()
        if report_id not in self.active_connections:
            self.active_connections[report_id] = []
        self.active_connections[report_id].append(websocket)

        # 연결되면 보관된 로그들을 전부 전송
        if report_id in self.log_history:
            for old_log in self.log_history[report_id]:
                await websocket.send_text(old_log)

    def disconnect(self, report_id: int, websocket: WebSocket):
        if report_id in self.active_connections:
            self.active_connections[report_id].remove(websocket)

    async def send_log(self, report_id: int, message: str):
        # 보낼 로그를 먼저 보관함에 저장
        if report_id not in self.log_history:
            self.log_history[report_id] = []
        self.log_history[report_id].append(message)

        if report_id in self.active_connections:
            for connection in self.active_connections[report_id]:
                try:
                    await connection.send_text(message)
                except:
                    pass


# 영상 감지 내역 테이블
class DetectionResult(Base):
    __tablename__ = "detection_results"

    id = Column(Integer, primary_key=True, index=True) # DB 관리용 번호
    object_id = Column(Integer) # 감지된 객체 id
    detected_color = Column(String(20)) # 감지된 color
    video_name = Column(String(100)) # video 이름
    detected_at = Column(DateTime, default=datetime.datetime.now) # 분석된 시간


# users 테이블 정의
class User(Base):
    __tablename__ = "users"

    user_no = Column(Integer, primary_key=True, index=True) # DB 관리용 번호
    id = Column(String(50), unique=True, nullable=False)   # 로그인 아이디
    password = Column(String(255), nullable=False)          # 암호화된 비밀번호
    residentFront = Column(String(6), nullable=False)           # 주민번호 앞자리
    residentBack = Column(String(255), nullable=False)          # 암호화된 주민번호 뒷자리
    name = Column(String(50), nullable=False)               # 사용자 이름
    phone = Column(String(20), nullable=False)       # 휴대폰 번호
    role = Column(String(20), default="USER")     # 역할 "USER"(일반), "ADMIN"(관계자)
    affiliation = Column(String(100), nullable=True) # 관계자일 경우 소속명 저장
    created_at = Column(DateTime, default=datetime.datetime.now) # 가입일

# 소속 코드 관리 테이블
class Affiliation(Base):
    __tablename__ = "affiliations"

    id = Column(Integer, primary_key=True, index=True) # DB 관리용 번호
    code = Column(String(50), unique=True, nullable=False)  # 인증 코드
    name = Column(String(100), nullable=False)              # 소속명 (예: xx경찰서, oo소방서)

# 신고 내역 테이블
class IncidentReport(Base):
    __tablename__ = "incident_reports"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(50), nullable=False) # 신고자 성함
    phone_number = Column(String(20), nullable=False) # 전화번호
    ssn = Column(String(255), nullable=False) # 주민번호 (암호화 저장)
    content = Column(String(500), nullable=True) # 신고 상세 내용
    location = Column(String(200), nullable=True) # 위치 정보 컬럼 추가
    video_path = Column(String(200), nullable=True) # 분석할 영상 경로
    created_at = Column(DateTime, default=datetime.datetime.now)

class HandoverEvent(Base):
    __tablename__ = "handover_events"
    id = Column(Integer, primary_key=True, index=True)
    report_id = Column(Integer, nullable=False)
    obj_id = Column(Integer, nullable=False)
    from_cam = Column(String(50), nullable=False)
    exit_time = Column(DateTime, default=datetime.datetime.now)
    vx = Column(Float, nullable=False) # X축 속도
    vy = Column(Float, nullable=False) # Y축 속도
    reid_feature = Column(String(8000), nullable=True) # reid로 지문에 해당

class CameraTopology(Base):
    __tablename__ = "camera_topology"
    id = Column(Integer, primary_key=True, index=True)
    cam_name = Column(String(50), unique=True, nullable=False)
    lat = Column(Float, nullable=False) # 위도 (Latitude)
    lon = Column(Float, nullable=False) # 경도 (Longitude)
    fov_angle = Column(Float, nullable=True) # 카메라가 바라보는 방위각 (추후)


# 테이블 자동 생성 (DB 탭에서 확인 가능)
Base.metadata.create_all(bind=engine)


app = FastAPI()
manager = ConnectionManager()
main_loop = None

@app.on_event("startup")
async def startup_event():
    global main_loop
    main_loop = asyncio.get_running_loop()

# GPU 장치 설정 및 로드
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f" 실행 장치: {device} ")

# yolov8 모델을 GPU 메모리로 로드
model = YOLO('yolov8n-pose.pt').to(device)

# 전역 메모리 설정
# 시계열 분석을 위한 color_buffer
# 프레임 skip시 전 프레임의 객체박스 위치를 기록할 latest_track_data
color_buffer = {}
latest_track_data = {}

"""
LLM(Gemini API) 이용해 자연어 신고 내용에서 타겟의 '종류'와 '색상'을 추출
return은 json형태
"""
def extract_color_with_llm(content: str) -> dict:
    if not content:
        return {}

    prompt = f"""
    당신은 CCTV 관제 시스템의 분석 AI입니다.
    다음 신고 내용에서 추적 대상의 '옷 종류'와 '색상'을 JSON 형식으로 추출하세요.
    
    [핵심 지시사항]
    사용자가 한국어로 색상을 입력하더라도, 반드시 아래의 [허용된 영어 단어] 중 가장 알맞은 것으로 번역해서 출력해야 합니다.
    (예: "빨간 옷" -> "Red", "노란색" -> "Yellow")

    '옷 종류'는 'top'(상의) 또는 'bottom'(하의) 중 하나여야 합니다.
    '색상'은 다음 [허용된 영어 단어] 중 하나여야 합니다.
    [허용된 영어 단어]: Black, White, Red, Blue, Yellow, Green, Purple, Gray, Pink, Orange, Brown, Navy, Skyblue

    신고 내용: "{content}"

    출력 형식 (JSON):
    {{
      "type": "top" or "bottom",
      "color": "ColorName"
    }}
    """
    try:
        response = llm_model.generate_content(prompt)
        clean_response = response.text.strip().replace("```json", "").replace("```", "").strip()
        result = json.loads(clean_response)

        allowed_colors = ["Black", "White", "Red", "Blue", "Yellow", "Green", "Purple", "Gray", "Pink", "Orange", "Brown", "Navy", "Skyblue"]
        if (result.get("type") in ["top", "bottom"] and result.get("color") in allowed_colors):
            return result
        return {}
    except Exception as e:
        print(f" LLM 파싱 에러: {e}")
        return {}

# ReID 특징 벡터(지문) 추출 함수
def extract_reid_feature(roi_img):
    if roi_img is None or roi_img.size == 0:
        return None
    try:
        # BGR을 RGB로 변환 후 전처리
        rgb_img = cv2.cvtColor(roi_img, cv2.COLOR_BGR2RGB)
        img_tensor = reid_transform(rgb_img).unsqueeze(0).to(reid_device)

        with torch.no_grad():
            features = reid_model(img_tensor)

        # JSON 문자열(Text)로 변환하여 리턴
        feat_list = features.cpu().numpy().flatten().tolist()
        return json.dumps(feat_list)
    except Exception as e:
        print(f" [ReID] 특징 추출 실패: {e}")
        return None

# 코사인 유사도 함수
def compute_cosine_similarity(feat1_json, feat2_json):
    if not feat1_json or not feat2_json:
        return 0.0
    try:
        v1 = np.array(json.loads(feat1_json))
        v2 = np.array(json.loads(feat2_json))
        # 0.0 ~ 1.0 사이의 유사도 값 반환
        return float(np.dot(v1, v2) / (norm(v1) * norm(v2)))
    except Exception as e:
        print(f" [ReID 비교 에러] {e}")
        return 0.0


"""
표준 Hue 범위 기반 색상 판별 로직
1. ROI 내 대표 색상을 판단(K-means)
2. BGR->HSV로 변경
3. 변경된 HSV값으로 색상을 판별해 return
"""

def detect_color_name(roi):
    if roi is None or roi.size == 0:
        return "Unknown"

    # 1. 대표 색상 추출 (K-means)
    small_roi = cv2.resize(roi, (24, 24))
    pixels = small_roi.reshape((-1, 3))
    pixels = np.float32(pixels)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, centers = cv2.kmeans(pixels, 2, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    counts = np.bincount(labels.flatten())
    dominant_color_bgr = centers[np.argmax(counts)]

    # 2. BGR -> HSV 변환
    hsv_pixel = cv2.cvtColor(np.uint8([[dominant_color_bgr]]), cv2.COLOR_BGR2HSV)[0][0]
    h, s, v = int(hsv_pixel[0]), int(hsv_pixel[1]), int(hsv_pixel[2])

    # 3. 색상 판별 로직


    if v < 50: return "Black"
    if s < 35 and v > 190: return "White"
    if s < 60 and v < 200: return "Gray"

    if (0 <= h < 8 or 172 <= h <= 180):
        if s > 160 and v > 120:
            return "Red"
        elif s > 50 and v > 130:
            return "Pink"
        else:
            return "Brown"


    elif 8 <= h < 38:
        if h >= 18: return "Yellow"
        else:
            if s > 100 and v > 100: return "Orange"
            else: return "Brown"

    elif 38 <= h < 85: return "Green"


    elif 85 <= h < 105:
        if s > 95: return "Skyblue"
        else: return "Gray"


    elif 105 <= h < 125:
        if s > 80 and v > 60: return "Blue"
        else: return "Navy"


    elif 125 <= h < 150:
        if s > 50: return "Purple"
        else: return "Gray"

    elif 150 <= h < 172:
        if s > 40: return "Pink"
        else: return "Gray"

    return "Unknown"



"""
시계열 분석
5프레임간을 분석을 통해 객체의 색상 판별하여 순간적으로 발생하는 오차를 최소화
"""
def get_smoothed_color(obj_id, new_color):
    if obj_id not in color_buffer:
        color_buffer[obj_id] = deque(maxlen=5)
    color_buffer[obj_id].append(new_color)
    if len(color_buffer[obj_id]) < 3: return new_color
    return Counter(color_buffer[obj_id]).most_common(1)[0][0]


# 두 위경도 좌표 사이의 실제 물리적 거리(미터)를 구하는 함수
def get_real_distance(lat1, lon1, lat2, lon2):
    R = 6371000 # 지구 반지름 (m)
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)

    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c # 미터(m)로 반환




# 시간에 따른 활동 반경을 계산하고, 그 안의 CCTV를 찾아내는 함수
def get_cameras_in_search_radius(db: Session, from_cam_name: str, vx: float, vy: float, elapsed_seconds: float):
    start_cam = db.query(CameraTopology).filter(CameraTopology.cam_name == from_cam_name).first()
    if not start_cam:
        return []

    # 픽셀 속도 벡터의 크기(픽셀/프레임)를 실제 이동 속도(m/s)로 변환
    # (추후 camera_topology테이블에 fov_angle, 높이에 따라 튜닝 필요. 임시로 0.3 적용)
    pixel_speed = math.sqrt(vx**2 + vy**2)
    real_speed_mps = pixel_speed * 0.3

    # 만약 객체가 멈춰서 나갔다면 최소한의 도보 속도(1.2m/s) 가정
    if real_speed_mps < 0.5:
        real_speed_mps = 1.2

    # 현재 예상되는 활동 반경 (거리 = 속력 * 시간, real_speed_mps * elapsed_seconds)
    # 최소 오차범위(Margin) 15m를 더함
    current_radius_m = (real_speed_mps * elapsed_seconds) + 15.0

    # 활성화될 cams 리스트
    activated_cams = []
    all_cams = db.query(CameraTopology).filter(CameraTopology.cam_name != from_cam_name).all()

    for cam in all_cams:
        dist = get_real_distance(start_cam.lat, start_cam.lon, cam.lat, cam.lon)
        # 카메라가 수색 반경 안으로 들어왔다면 활성화 목록에 추가
        if dist <= current_radius_m:
            activated_cams.append({"cam_name": cam.cam_name, "distance": round(dist, 1)})

    return activated_cams



def find_cameras_in_radius(db: Session, from_cam: str, vx: float, vy: float, seconds: float):
    start = db.query(CameraTopology).filter(CameraTopology.cam_name == from_cam).first()
    if not start: return []
    # 픽셀 속도를 m/s로 환산 (약 0.3 계수 사용)
    radius = (math.sqrt(vx**2 + vy**2) * 0.3 * seconds) + 15.0
    return db.query(CameraTopology).filter(CameraTopology.cam_name != from_cam).all()

"""
영상 분석할 때 사용하는 함수입니다.
신고 내역 content를 기반으로 영상 내부의 객체를 찾습니다.
신고 내역에서 llm을 통해 색상text를 추출.
test영상 파일을 load.
load된 영상에 yolo를 이용해 객체를 추출.
추출된 객체와 색상text가 일치하면 box로 강조 표시.
분석이 완료되면 web버전 영상으로 변경 후 업로드.
"""
def process_video_analysis(report_id: int, content: str = None):

    # 웹 상에서 해당 id의 신고 내역에 표시될 로그 함수.
    def emit_log(msg):
        print(f" [EMIT_LOG] {msg}")
        if main_loop:
            asyncio.run_coroutine_threadsafe(manager.send_log(report_id, msg), main_loop)
        else:
            print(" [ERROR] 메인 루프가 잡히지 않아 로그를 보낼 수 없습니다.")

    db = SessionLocal()
    video_sources = ["test1.mp4", "test2.mp4", "test3.mp4", "test4.mp4"]
    output_dir = os.path.join("static", "outputs", f"report_{report_id}")
    os.makedirs(output_dir, exist_ok=True)

    # API 제한으로 잠시 비활성화
    target_info = extract_color_with_llm(content)
    target_color = target_info.get("color", "")
    target_type = target_info.get("type", "")
    # [테스트용 강제 고정]
    #target_color = "yellow"   # 무조건 yellow만 찾도록 고정
    #target_type = ""       # 옷 부위 조건은 무시

    print(f" LLM 추출 결과: 색상={target_color}, 부위={target_type}")
    log_message = f" 신고 - 타겟 조건: {target_color} {target_type}" if target_color else "모든 객체"
    emit_log(log_message)

    try:
        for i, src_name in enumerate(video_sources, 1):
            if not os.path.exists(src_name):
                print(f" [경고] {src_name} 파일이 없습니다. 건너뜁니다.")
                continue

            print(f" [CH {i}] 분석 시작: {src_name} (Target: {target_color} {target_type})")
            emit_log(f"CAM 0{i} 채널 분석 시작...")

            latest_handover = db.query(HandoverEvent).filter(HandoverEvent.report_id == report_id).order_by(HandoverEvent.id.desc()).first()
            # 현재 카메라가 아닌, '이전 카메라'에서 찍힌 지문일 때만 가져옴
            target_saved_feat = latest_handover.reid_feature if latest_handover and latest_handover.from_cam != f"CAM_0{i}" else None
            if target_saved_feat:
                print(f" [ReID] 이전 카메라({latest_handover.from_cam})의 타겟 지문 데이터 로드 완료.")

            web_out_filename = os.path.join(output_dir, f"web_out_{report_id}_{i}.mp4")

            cap = cv2.VideoCapture(src_name)
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            orig_w, orig_h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            # 영상 조정
            PROC_WIDTH = 1280
            w = PROC_WIDTH
            h = int(orig_h * (PROC_WIDTH / orig_w))

            # FFmpeg 설정
            ffmpeg_command = [
                'ffmpeg', '-y', '-f', 'rawvideo', '-vcodec', 'rawvideo',
                '-pix_fmt', 'bgr24', '-s', f'{w}x{h}', '-r', str(fps),
                '-i', '-', '-vcodec', 'libx264', '-preset', 'ultrafast',
                '-crf', '28', '-pix_fmt', 'yuv420p', web_out_filename
            ]
            ffmpeg_process = subprocess.Popen(ffmpeg_command, stdin=subprocess.PIPE)

            local_track_data = {}
            local_color_buffer = {}
            local_roi_buffer = {}
            frame_count = 0
            last_log_time = 0

            while cap.isOpened():
                success, original_frame = cap.read()
                if not success: break

                # Resize frame before any processing
                frame = cv2.resize(original_frame, (w, h))
                frame_count += 1

                if frame_count % 3 == 0:
                    results = model.track(frame, persist=True, verbose=False, conf=0.3, device=device, tracker="bytetrack.yaml")
                    new_tracks = {}

                    if results[0].boxes.id is not None:
                        boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
                        ids = results[0].boxes.id.cpu().numpy().astype(int)

                        has_keypoints = results[0].keypoints is not None
                        if has_keypoints:
                            kpts = results[0].keypoints.xy.cpu().numpy().astype(int)

                        for j, obj_id in enumerate(ids):
                            x1, y1, x2, y2 = boxes[j]
                            rois = []
                            roi_boxes_for_vis = []

                            if has_keypoints and len(kpts[j]) >= 17:
                                if target_type != "bottom": # 상의 또는 기본
                                    shoulder_l, shoulder_r = kpts[j][5], kpts[j][6]
                                    hip_l, hip_r = kpts[j][11], kpts[j][12]
                                    valid_pts = [p for p in [shoulder_l, shoulder_r, hip_l, hip_r] if p[0] > 0 and p[1] > 0]
                                    if len(valid_pts) >= 3:
                                        pts_array = np.array(valid_pts)
                                        tx1, ty1 = np.min(pts_array, axis=0)
                                        tx2, ty2 = np.max(pts_array, axis=0)
                                        margin_x = int((tx2 - tx1) * 0.1)
                                        crop_x1, crop_y1 = max(0, tx1 + margin_x), max(0, ty1)
                                        crop_x2, crop_y2 = min(w, tx2 - margin_x), min(h, ty2)
                                        if crop_y2 > crop_y1 and crop_x2 > crop_x1:
                                            rois.append(frame[crop_y1:crop_y2, crop_x1:crop_x2])
                                            roi_boxes_for_vis.append((crop_x1, crop_y1, crop_x2, crop_y2))

                                elif target_type == "bottom": # 하의
                                    # 왼쪽 다리
                                    hip_l, knee_l, ankle_l = kpts[j][11], kpts[j][13], kpts[j][15]
                                    leg_l_pts = [p for p in [hip_l, knee_l, ankle_l] if p[0] > 0 and p[1] > 0]
                                    if len(leg_l_pts) >= 2:
                                        pts_array = np.array(leg_l_pts)
                                        lx1, ly1 = np.min(pts_array, axis=0)
                                        lx2, ly2 = np.max(pts_array, axis=0)
                                        margin_x = int((lx2 - lx1) * 0.5)
                                        crop_x1, crop_y1 = max(0, lx1 - margin_x), max(0, ly1)
                                        crop_x2, crop_y2 = min(w, lx2 + margin_x), min(h, ly2)
                                        if crop_y2 > crop_y1 and crop_x2 > crop_x1:
                                            rois.append(frame[crop_y1:crop_y2, crop_x1:crop_x2])
                                            roi_boxes_for_vis.append((crop_x1, crop_y1, crop_x2, crop_y2))

                                    # 오른쪽 다리
                                    hip_r, knee_r, ankle_r = kpts[j][12], kpts[j][14], kpts[j][16]
                                    leg_r_pts = [p for p in [hip_r, knee_r, ankle_r] if p[0] > 0 and p[1] > 0]
                                    if len(leg_r_pts) >= 2:
                                        pts_array = np.array(leg_r_pts)
                                        rx1, ry1 = np.min(pts_array, axis=0)
                                        rx2, ry2 = np.max(pts_array, axis=0)
                                        margin_x = int((rx2 - rx1) * 0.5)
                                        crop_x1, crop_y1 = max(0, rx1 - margin_x), max(0, ry1)
                                        crop_x2, crop_y2 = min(w, rx2 + margin_x), min(h, ry2)
                                        if crop_y2 > crop_y1 and crop_x2 > crop_x1:
                                            rois.append(frame[crop_y1:crop_y2, crop_x1:crop_x2])
                                            roi_boxes_for_vis.append((crop_x1, crop_y1, crop_x2, crop_y2))

                            if not rois: # 백업 로직
                                roi_h, roi_w = y2 - y1, x2 - x1
                                if target_type == "bottom":
                                    crop_y1, crop_y2 = int(y1 + roi_h * 0.55), int(y1 + roi_h * 0.95)
                                else:
                                    crop_y1, crop_y2 = int(y1 + roi_h * 0.20), int(y1 + roi_h * 0.45)
                                crop_x1, crop_x2 = int(x1 + roi_w * 0.35), int(x2 - roi_w * 0.35)
                                crop_y1, crop_y2 = max(0, crop_y1), min(h, crop_y2)
                                crop_x1, crop_x2 = max(0, crop_x1), min(w, crop_x2)
                                if crop_y2 > crop_y1 and crop_x2 > crop_x1:
                                    rois.append(frame[crop_y1:crop_y2, crop_x1:crop_x2])
                                    roi_boxes_for_vis.append((crop_x1, crop_y1, crop_x2, crop_y2))

                            detected_colors = [detect_color_name(r) for r in rois if r is not None and r.size > 0]
                            stable_color = "Unknown"

                            if detected_colors:
                                if obj_id not in local_color_buffer:
                                    local_color_buffer[obj_id] = deque(maxlen=10)
                                for color in detected_colors:
                                    if color != "Unknown":
                                        local_color_buffer[obj_id].append(color)

                                current_buffer = local_color_buffer.get(obj_id)
                                if current_buffer:
                                    if len(current_buffer) >= 3:
                                        stable_color = Counter(current_buffer).most_common(1)[0][0]
                                    else:
                                        stable_color = current_buffer[-1]

                            #마스터 지문을 생성을 위해 매 프레임 ROI 누적
                            if obj_id not in local_roi_buffer:
                                local_roi_buffer[obj_id] = deque(maxlen=15) # 최대 15장 보관

                            best_roi = None
                            if roi_boxes_for_vis:
                                rx1, ry1, rx2, ry2 = roi_boxes_for_vis[0]
                                best_roi = frame[ry1:ry2, rx1:rx2]
                            elif boxes[j] is not None:
                                bx1, by1, bx2, by2 = boxes[j]
                                best_roi = frame[max(0, by1):by2, max(0, bx1):bx2]

                            if best_roi is not None and best_roi.size > 0:
                                local_roi_buffer[obj_id].append(best_roi)

                            # 현재 중심 좌표 및 속도 계산
                            curr_c = ((x1 + x2) // 2, (y1 + y2) // 2)
                            prev = local_track_data.get(obj_id, {})
                            vx = (curr_c[0] - prev.get("prev_c", curr_c)[0]) // 3
                            vy = (curr_c[1] - prev.get("prev_c", curr_c)[1]) // 3

                            is_target = (target_color == "") or (target_color.lower() in stable_color.lower())
                            is_exiting = (x1 < 40 or x2 > w - 40 or y1 < 40 or y2 > h - 40)
                            is_confirmed = prev.get("confirmed", False) or is_reid_matched
                            is_valid_target = is_target if not target_saved_feat else (is_target and is_confirmed)

                            # 실시간 ReID 매칭 로직
                            reid_score = 0.0
                            is_reid_matched = False

                            # 타겟 색상이고, 비교할 이전 지문이 존재할 때만 실행
                            if is_valid_target and is_exiting and not prev.get("ex_sent") and frame_count > 10:
                                curr_feat_str = None
                                rois_to_process = local_roi_buffer.get(obj_id, [])
                                if roi_boxes_for_vis:
                                    rx1, ry1, rx2, ry2 = roi_boxes_for_vis[0]
                                    curr_feat_str = extract_reid_feature(frame[ry1:ry2, rx1:rx2])
                                elif boxes[j] is not None:
                                    curr_feat_str = extract_reid_feature(frame[max(0, by1):by2, max(0, bx1):bx2])

                                if curr_feat_str:
                                    reid_score = compute_cosine_similarity(target_saved_feat, curr_feat_str)

                                    # reID 유사도가 65% 이상일시
                                    if reid_score >= 0.65:
                                        curr_cam_name = f"CAM_0{i}"

                                        # DB에서 이전 카메라와 현재 카메라의 위치 정보(lat, lon) 가져오기
                                        prev_cam = db.query(CameraTopology).filter(CameraTopology.cam_name == latest_handover.from_cam).first()
                                        curr_cam = db.query(CameraTopology).filter(CameraTopology.cam_name == curr_cam_name).first()

                                        if prev_cam and curr_cam and latest_handover.exit_time:
                                            # 두 카메라 간의 실제 물리적 거리(m) 계산
                                            dist = get_real_distance(prev_cam.lat, prev_cam.lon, curr_cam.lat, curr_cam.lon)

                                            # 경과 시간(초) 계산: 현재 분석 시간 - 과거 이탈 시간
                                            time_elapsed = (datetime.datetime.now() - latest_handover.exit_time).total_seconds()
                                            time_elapsed = max(time_elapsed, 0.1)

                                            # 해당 거리와 시간 동안의 이동 속도(m/s)
                                            required_speed = dist / time_elapsed

                                            # 인간의 이동 한계 속도 판별
                                            if required_speed <= 15.0:
                                                is_reid_matched = True
                                            else:
                                                is_reid_matched = False
                                        else:
                                            # 좌표 정보가 DB에 누락된 경우 일단 매칭 허용 (예외)
                                            is_reid_matched = True

                            if is_target and is_exiting and not prev.get("ex_sent") and frame_count > 10:
                                # 15장 이미지를 병합해 마스터 지문 생성
                                feat_str = None
                                rois_to_process = local_roi_buffer.get(obj_id, [])

                                if rois_to_process:
                                    extracted_feats = []
                                    for r_img in rois_to_process:
                                        f_json = extract_reid_feature(r_img)
                                        if f_json:
                                            extracted_feats.append(np.array(json.loads(f_json)))

                                    if extracted_feats:
                                        # 여러 지문을 수학적으로 평균(Mean) 내기
                                        avg_feat = np.mean(extracted_feats, axis=0)
                                        # 길이를 1로 맞춰 정규화(Normalize)하여 오차 제거
                                        avg_feat = avg_feat / norm(avg_feat)
                                        feat_str = json.dumps(avg_feat.tolist())

                                try:
                                    new_event = HandoverEvent(
                                        report_id=report_id, obj_id=int(obj_id),
                                        from_cam=f"CAM_0{i}", vx=float(vx), vy=float(vy),
                                        reid_feature=feat_str # 마스터 지문 저장
                                    )
                                    db.add(new_event)
                                    db.commit()
                                    emit_log(f"[AI] [EXIT] 타겟(ID:{obj_id}) 이탈. 15프레임 기반 '마스터 지문' 생성 완료.")
                                    ex_sent_flag = True
                                except:
                                    db.rollback()
                                    ex_sent_flag = False

                            else:
                                ex_sent_flag = prev.get("ex_sent", False)

                            new_tracks[obj_id] = {
                                "box": boxes[j],
                                "color": stable_color,
                                "roi_boxes": roi_boxes_for_vis,
                                "prev_c": curr_c,
                                "ex_sent": ex_sent_flag,
                                "reid_score": reid_score,
                                "is_reid_matched": is_reid_matched,
                                "confirmed": is_confirmed
                            }
                    local_track_data = new_tracks

                current_time = time.time()
                for obj_id, data in local_track_data.items():
                    if target_color == "" or target_color.lower() in data["color"].lower():
                        bx1, by1, bx2, by2 = data["box"]
                        match_text = f" (Match: {data.get('reid_score', 0.0)*100:.1f}%)" if data.get('reid_score', 0.0) > 0 else ""
                        cv2.rectangle(frame, (bx1, by1), (bx2, by2), (0, 0, 255), 3)
                        cv2.putText(frame, f"TARGET ID:{obj_id} {data['color']}{match_text}", (bx1, by1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                        for rx1, ry1, rx2, ry2 in data.get("roi_boxes", []):
                            cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), (0, 255, 0), 2)

                        if data.get("roi_boxes"):
                            rx1, ry1, _, _ = data["roi_boxes"][0]
                            cv2.putText(frame, "ROI", (rx1, ry1 - 5),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                        if current_time - last_log_time > 1.5:
                            if target_saved_feat and data.get("is_reid_matched"):
                                # 동일인 확인 시 붉은색 사이렌 이모지와 함께 확실한 매칭 알림
                                emit_log(f" [ReID MATCH] CAM 0{i} 타겟 재식별 확정! (일치율: {data['reid_score']*100:.1f}%)")
                            elif not target_saved_feat:
                                emit_log(f"MATCH: CAM 0{i}에서 {data['color']} {target_type if target_type else '객체'}(ID:{obj_id}) 추적 중")
                            last_log_time = current_time

                try:
                    ffmpeg_process.stdin.write(frame.tobytes())
                except (IOError, BrokenPipeError) as e:
                    print(f" FFmpeg 파이프 에러: {e}. 스트리밍을 중단합니다.")
                    break

            cap.release()
            ffmpeg_process.stdin.close()
            ffmpeg_process.wait()
            print(f" [CH {i}] 변환 완료: {web_out_filename}")
            emit_log(f"VIDEO_READY_CH:{i}")

        emit_log("모든 분석 프로세스 종료")

        report = db.query(IncidentReport).filter(IncidentReport.id == report_id).first()
        if report:
            report.video_path = f"outputs/report_{report_id}/web_out_{report_id}_1.mp4"
            db.commit()

    except Exception as e:
        db.rollback()
        print(f" 분석 프로세스 전체 오류: {e}")
    finally:
        db.close()


# DB 세션을 가져오는 헬퍼 함수
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# WebSocket 엔드포인트
@app.websocket("/ws/{report_id}")
async def websocket_endpoint(websocket: WebSocket, report_id: str):
    r_id = int(report_id) if report_id.isdigit() else report_id
    print(f" [DEBUG] 새 연결 시도: Report ID {r_id}")
    await manager.connect(r_id, websocket)
    try:
        while True:
            await websocket.receive_text() # 연결 유지를 위한 대기
    except WebSocketDisconnect:
        manager.disconnect(report_id, websocket)


"""
앱이나 웹에서 탐지 기록을 조회할 때 사용하는 API입니다.
특정 색상만 골라서 보고 싶을 때 /logs?color=Red 처럼 요청할 수 있습니다.
"""
@app.get("/logs")
async def get_detection_logs(color: str = Query(None)):

    db = SessionLocal()
    try:
        # 1. DB에서 쿼리 생성
        query = db.query(DetectionResult)

        # 2. 색상 필터링이 들어온 경우 처리
        if color:
            query = query.filter(DetectionResult.detected_color == color)

        # 3. 최신순으로 정렬하여 데이터 가져오기 (최신 발견이 위로)
        logs = query.order_by(DetectionResult.detected_at.desc()).all()

        # 4. 앱/웹이 이해하기 쉬운 리스트 형식으로 반환
        return [
            {
                "id": log.id,
                "object_id": log.object_id,
                "color": log.detected_color,
                "video_url": f"http://서버주소/download/{log.video_name}", # 영상 링크
                "detected_at": log.detected_at.strftime("%Y-%m-%d %H:%M:%S")
            } for log in logs
        ]
    except Exception as e:
        return {"error": str(e)}
    finally:
        db.close()




# 회원가입 API
@app.post("/signup/user")
async def register_user(
        request: Request,
        id: str = Form(...),            # 앱의 JSON 대신 웹의 Form 데이터를 받습니다.
        password: str = Form(...),
        residentFront: str = Form(...),
        residentBack: str = Form(...),
        name: str = Form(...),
        phone: str = Form(...),
        db: Session = Depends(get_db)
):
    # 2. 아이디 중복 확인 (기존 로직 동일)
    existing_user = db.query(User).filter(User.id == id).first()
    if existing_user:
        return HTMLResponse(content="<script>alert('이미 사용 중인 아이디입니다.'); history.back();</script>", status_code=400)

    # 1. 기존 로직 그대로 사용 (암호화)
    hashed_pwd = pwd_context.hash(password)
    hashed_res_back = pwd_context.hash(residentBack)

    # 2. DB에 저장할 사용자 객체 생성
    new_user = User(
        id=id,
        password=hashed_pwd,
        residentFront=residentFront,
        residentBack=hashed_res_back,
        name=name,
        phone=phone
    )

    # 3. DB에 저장 (SQLAlchemy가 INSERT 쿼리를 자동 생성)
    db.add(new_user)
    db.commit()
    db.refresh(new_user)

    return RedirectResponse(url="/", status_code=303)

# 신고 접수 API
@app.post("/report/submit")
async def submit_report(
        report: ReportCreate,
        background_tasks: BackgroundTasks,
        db: Session = Depends(get_db)
):
    # 주민번호 암호화 저장
    hashed_ssn = pwd_context.hash(report.ssn)

    new_report = IncidentReport(
        name=report.name,
        phone_number=report.phone_number,
        ssn=hashed_ssn,
        content=report.content,
        location=report.location # 위치 정보 저장
    )

    db.add(new_report)
    db.commit()
    db.refresh(new_report)

    # 비디오 분석 함수를 background_task로 등록
    # 사용자가 보낸 신고내용(content)을 함께 넘겨 특정 색상을 찾게 합니다.
    background_tasks.add_task(process_video_analysis, report_id=new_report.id, content=report.content)

    return {
        "status": "success",
        "message": "신고 접수가 완료되었습니다. 비디오 분석이 백그라운드에서 시작됩니다.",
        "report_id": new_report.id
    }

# 관리자(admin) 회원가입 API
@app.post("/signup/admin")
def signup_admin(request: Request,
                 id: str = Form(...),
                 password: str = Form(...),
                 name: str = Form(...),
                 phone: str = Form(...),
                 residentFront: str = Form(...),
                 residentBack: str = Form(...),
                 orgCode: str = Form(...), # [핵심] 관리자용 추가 필드
                 db: Session = Depends(get_db)
):

    # 소속 코드 검증
    # 입력한 코드가 DB에 있는지 확인하고, 없으면 에러 발생
    affiliation_data = db.query(Affiliation).filter(Affiliation.code == orgCode).first()

    if not affiliation_data:
        return HTMLResponse(content="<script>alert('유효하지 않은 소속 코드입니다.'); history.back();</script>", status_code=400)

    # 아이디 중복 확인
    existing_user = db.query(User).filter(User.id == id).first()
    if existing_user:
        return HTMLResponse(content="<script>alert('이미 사용 중인 아이디입니다.'); history.back();</script>", status_code=400)

    # 비밀번호 해싱
    hashed_password = pwd_context.hash(password)
    hashed_res_back = pwd_context.hash(residentBack)

    # DB 저장
    # role은 "ADMIN", affiliation은 코드에 해당하는 이름(ex: oo경찰서)으로 자동 저장
    new_user = User(
        id=id,
        password=hashed_password,      # 암호화된 비밀번호 저장
        name=name,
        phone=phone,
        residentFront=residentFront,
        residentBack=hashed_res_back,
        role="ADMIN",                  # 관리자 권한 부여
        affiliation=affiliation_data.name # 소속명 입력
    )

    db.add(new_user)
    db.commit()
    db.refresh(new_user)

    return RedirectResponse(url="/", status_code=303)


# 1. 정적 파일(CSS/JS) 연결
app.mount("/static", StaticFiles(directory="static"), name="static")

# 2. Jinja2 템플릿 엔진 설정
templates = Jinja2Templates(directory="templates")

# 메인 페이지 (http://IP:8000/ 접속 시)
@app.get("/", response_class=HTMLResponse)
async def read_index(request: Request, db: Session = Depends(get_db)):
    # 브라우저가 가져온 쿠키에서 "session_user"가 있는지 확인합니다.
    user_id = request.cookies.get("session_user")
    user_info = None
    if user_id:
        # 2. ID가 있다면 DB에서 사용자 전체 정보를 조회 (이름, 소속, 권한 등)
        user_info = db.query(User).filter(User.id == user_id).first()
    # 템플릿 파일명과 함께 데이터를 넘겨줍니다.
    return templates.TemplateResponse("home.html", {
        "request": request,
        "user": user_info
    })


# 로그인 기능(앱)
@app.post("/login")
async def login(
        request: Request,
        id: str = Form(...), # [수정] user_data 대신 직접 Form 데이터를 받습니다.
        password: str = Form(...),
        db: Session = Depends(get_db)
):
    # 1. DB에서 해당 아이디의 사용자 찾기
    user = db.query(User).filter(User.id == id).first()

    # 2. 사용자가 없거나 비밀번호가 틀린 경우
    if not user or not pwd_context.verify(password, user.password):
        # 웹 환경에서는 에러 페이지로 리다이렉트하거나 에러 파라미터를 붙여 리턴하는 게 좋습니다.
        return templates.TemplateResponse("login.html", {
            "request": request,
            "error": True
        })

    # 3. 로그인 성공 시 관리자 페이지로 이동 (Redirect)
    response = RedirectResponse(url="/", status_code=303)
    # "session_user"라는 이름으로 아이디를 브라우저에 저장합니다. (유효기간 1시간)
    response.set_cookie(key="session_user", value=user.id, httponly=True, max_age=3600)
    # [중요] 여기에 세션 정보를 쿠키 등으로 저장하는 로직이 추가되어야 나중에 /admin 페이지에 들어갈 수 있습니다.
    return response


# 1. 로그인 페이지 보여주기 (GET)
@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request, error: bool = False, logout: bool = False):
    return templates.TemplateResponse("login.html", {
        "request": request,
        "error": error,
        "logout": logout
    })

# 1. 로그아웃으로 메인화면으로 이동 (post)
@app.post("/logout")
async def logout():
    response = RedirectResponse(url="/", status_code=303)
    response.delete_cookie("session_user") # 쿠키 삭제로 로그아웃 처리
    return response

# 1. 회원가입 유형 선택 페이지
@app.get("/signup/select", response_class=HTMLResponse)
async def signup_select_page(request: Request):
    return templates.TemplateResponse("signup/select.html", {"request": request})

# 2. 일반 회원가입 양식 페이지
@app.get("/signup/user", response_class=HTMLResponse)
async def signup_user_page(request: Request):
    return templates.TemplateResponse("signup/user.html", {"request": request})

# 3. 관계자용 회원가입 양식 페이지
@app.get("/signup/admin", response_class=HTMLResponse)
async def signup_admin_page(request: Request):
    return templates.TemplateResponse("signup/admin.html", {"request": request})

# 관리자 화면 연결
@app.get("/admin", response_class=HTMLResponse)
async def admin_dashboard(request: Request, db: Session = Depends(get_db)):

    # 로그인 여부 확인
    user_id = request.cookies.get("session_user")
    if not user_id:
        # 로그인이 안 되어 있다면 로그인 페이지로 튕김
        return RedirectResponse(url="/login", status_code=303)

    # 사용자 정보 및 권한 조회
    user = db.query(User).filter(User.id == user_id).first()

    # 관리자(ADMIN) 권한 체크
    # 유저가 없거나, 역할이 ADMIN이 아니라면 '접근 거부' 페이지 리턴
    if not user or user.role != "ADMIN":
        return templates.TemplateResponse("access-denied.html", {"request": request})

    # 권한 통과 시: IncidentReport 테이블에서 모든 데이터를 최신순으로 가져옴
    reports = db.query(IncidentReport).order_by(IncidentReport.id.desc()).all()

    # admin_dashboard.html로 렌더링
    return templates.TemplateResponse("admin_dashboard.html", {
        "request": request,
        "reports": reports,
        "user": user # 상단에 이름 표시용
    })

# 관리자 화면 접근 거부 (일반 user 또는 로그인을 안한상태)
@app.get("/access-denied", response_class=HTMLResponse)
async def access_denied(request: Request):
    return templates.TemplateResponse("access-denied.html", {"request": request})

# CCTV 모니터링 페이지
@app.get("/admin/cctv/{report_id}", response_class=HTMLResponse)
async def admin_cctv_page(report_id: int, request: Request, db: Session = Depends(get_db)):

    # 관리자 권한 체크
    user_id = request.cookies.get("session_user")
    user = db.query(User).filter(User.id == user_id).first()
    if not user or user.role != "ADMIN":
        return templates.TemplateResponse("access-denied.html", {"request": request})

    # DB에서 신고 내역 조회
    report = db.query(IncidentReport).filter(IncidentReport.id == report_id).first()
    if not report:
        raise HTTPException(status_code=404, detail="영상을 찾을 수 없는 신고 내역입니다.")

    return templates.TemplateResponse("admin_cctv.html", {
        "request": request,
        "report": report, # HTML로 신고 정보 전달
        "user": user
    })

# 신고 내역 상세 보기
@app.get("/admin/{report_id}", response_class=HTMLResponse)
async def admin_report_detail(report_id: int, request: Request, db: Session = Depends(get_db)):
    # 권한 체크 (admin만 가능)
    user_id = request.cookies.get("session_user")
    user = db.query(User).filter(User.id == user_id).first()
    if not user or user.role != "ADMIN":
        return templates.TemplateResponse("access-denied.html", {"request": request})

    # 특정 신고 내역 조회
    report = db.query(IncidentReport).filter(IncidentReport.id == report_id).first()
    if not report:
        raise HTTPException(status_code=404, detail="신고 내역을 찾을 수 없습니다.")

    # 상세 페이지 렌더링
    return templates.TemplateResponse("admin_detail.html", {
        "request": request,
        "report": report,
        "user": user
    })



# 분석된 영상 파일을 브라우저에서 볼 수 있게 해주는 경로
@app.get("/video/{video_name}")
async def get_video(video_name: str):
    file_path = os.path.join(os.getcwd(), video_name)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404)

    return FileResponse(file_path, media_type="video/mp4")

@app.get("/api/camera-locations")
async def get_camera_locations(db: Session = Depends(get_db)):
    cams = db.query(CameraTopology).all()
    return {cam.cam_name: {"lat": cam.lat, "lon": cam.lon} for cam in cams}

# 특정 신고에 대한 최신 이탈 이벤트를 가져오는 API
@app.get("/api/latest-handover/{report_id}")
async def get_latest_handover(report_id: int, db: Session = Depends(get_db)):
    event = db.query(HandoverEvent).filter(HandoverEvent.report_id == report_id).order_by(HandoverEvent.id.desc()).first()
    if not event:
        return {"status": "none"}

    elapsed_seconds = (datetime.datetime.now() - event.exit_time).total_seconds()

    return {
        "status": "success",
        "from_cam": event.from_cam,
        "vx": event.vx,
        "vy": event.vy,
        "elapsed_seconds": max(0, elapsed_seconds)
    }


# 타겟의 전체 이동 궤적 좌표를 가져오는 API
@app.get("/api/tracking-path/{report_id}")
async def get_tracking_path(report_id: int, db: Session = Depends(get_db)):
    # 해당 신고 번호의 모든 이탈/핸드오버 이벤트를 시간순(오름차순)으로 가져옵니다.
    events = db.query(HandoverEvent).filter(HandoverEvent.report_id == report_id).order_by(HandoverEvent.id.asc()).all()

    path_coords = []
    seen_cams = [] # 중복 카메라 선 긋기 방지

    for ev in events:
        if ev.from_cam not in seen_cams:
            cam = db.query(CameraTopology).filter(CameraTopology.cam_name == ev.from_cam).first()
            if cam:
                path_coords.append({"lat": cam.lat, "lon": cam.lon, "cam_name": cam.cam_name})
            seen_cams.append(ev.from_cam)

    return {"status": "success", "path": path_coords}


# 카메라 GPS 초기 세팅
def init_camera_topology():
    db = SessionLocal()
    # 이미 데이터가 있으면 건너뜁니다.
    if db.query(CameraTopology).first():
        db.close()
        return

    # 테스트를 위한 가상의 거리 위경도 좌표 세팅
    cams = [
        CameraTopology(cam_name="CAM_01", lat=37.34000, lon=126.73300), # 메인 도로
        CameraTopology(cam_name="CAM_02", lat=37.34045, lon=126.73345), # 북쪽 거리 (약 60m 거리)
        CameraTopology(cam_name="CAM_03", lat=37.33950, lon=126.73250), # 남쪽 골목 (약 70m 거리)
        CameraTopology(cam_name="CAM_04", lat=37.34100, lon=126.73400)  # 외곽 뷰 (약 140m 거리)
    ]
    db.add_all(cams)
    db.commit()
    db.close()
    print(" [SYSTEM] 카메라 공간 정보(Topology) DB 초기화 완료")

# 서버 시작 시 딱 한 번 실행
init_camera_topology()

# 로컬 환경에서 실행시.
if __name__ == "__main__":
    import uvicorn
    # 로컬 환경(127.0.0.1)에서 8000번 포트로 서버를 즉시 실행합니다.
    uvicorn.run(app, host="127.0.0.1", port=8000)