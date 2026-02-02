import cv2
import os
import numpy as np
import time
import torch
from collections import deque, Counter
from fastapi import FastAPI, File, UploadFile, Query
from fastapi.responses import FileResponse
from ultralytics import YOLO
from sqlalchemy import create_engine, Column, Integer, String, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import datetime
from passlib.context import CryptContext
from pydantic import BaseModel
from fastapi import Depends
from sqlalchemy.orm import Session
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi import Request


# --- [1. 데이터베이스 설정 구간] ---
DATABASE_URL = "mysql+pymysql://root:0727@localhost:3306/safety_db"
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class DetectionResult(Base):
    __tablename__ = "detection_results"
    id = Column(Integer, primary_key=True, index=True)
    object_id = Column(Integer)
    detected_color = Column(String(20))
    video_name = Column(String(100))
    detected_at = Column(DateTime, default=datetime.datetime.now)

# 테이블 자동 생성 (DB 탭에서 확인 가능)
Base.metadata.create_all(bind=engine)

# 비밀번호 암호화 설정
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
# 모바일에서 받을 데이터 규격 (추후 수정)
class UserCreate(BaseModel):
    username: str
    password: str
    email: str = None
# 비밀번호 암호화 함수
def get_password_hash(password):
    return pwd_context.hash(password)
# 기존 Base를 상속받아 새로운 테이블 정의
class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, nullable=False) # 아이디
    password = Column(String(255), nullable=False)             # 암호화된 비번
    email = Column(String(100), nullable=True)                # 이메일
    created_at = Column(DateTime, default=datetime.datetime.now) # 가입일


app = FastAPI()

# 1. GPU 장치 설정 및 로드
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"✅ 실행 장치: {device} ")

# Nano 모델을 GPU 메모리로 로드
model = YOLO('yolov8n-pose.pt').to(device)

# 2. 전역 메모리 설정
# 시계열 분석을 위한 color_buffer
# 프레임 skip시 전 프레임의 객체박스 위치를 기록할 latest_track_data
color_buffer = {}
latest_track_data = {}


"""
표준 Hue 범위 기반 색상 판별 로직
"""
def detect_color_name(roi):

    if roi is None or roi.size == 0: return "Unknown"
    # 속도를 위해 24x24 리사이징
    small_roi = cv2.resize(roi, (24, 24), interpolation=cv2.INTER_AREA)
    data = small_roi.reshape((-1, 3)).astype(np.float32)

    # K-Means (K=2)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, centers = cv2.kmeans(data, 2, None, criteria, 5, cv2.KMEANS_RANDOM_CENTERS)
    dom_bgr = centers[np.argmax(np.bincount(labels.flatten()))]
    h, s, v = cv2.cvtColor(np.uint8([[dom_bgr]]), cv2.COLOR_BGR2HSV)[0][0]

    # 표준 HSV 색상 정의
    if v < 75: return "Black"
    if s < 50: return "White" if v > 200 else "Gray"
    if h < 8 or h > 170: return "Red"
    if 155 <= h <= 170: return "Pink"
    if 8 <= h < 16: return "Orange"
    if 16 <= h < 38: return "Yellow"
    if 38 <= h < 88: return "Green"
    if 88 <= h < 125: return "Blue"
    if 125 <= h < 155: return "Purple"
    return "Color"


"""
시계열 분석
10프레임을 비교해 객체의 색상 판별해 오차를 줄임
"""
def get_smoothed_color(obj_id, new_color):

    if obj_id not in color_buffer:
        color_buffer[obj_id] = deque(maxlen=10)
    color_buffer[obj_id].append(new_color)
    if len(color_buffer[obj_id]) < 4: return new_color
    return Counter(color_buffer[obj_id]).most_common(1)[0][0]


"""
영상 분석할 때 사용하는 API입니다.
"""
@app.post("/analyze/video")
async def analyze_video(content: str = Query(None), file: UploadFile = File(...)):
    global latest_track_data, color_buffer
    ts = int(time.time())
    in_path, out_path = f"in_{ts}.mp4", f"out_{ts}.mp4"
    cap, out = None, None

    latest_track_data = {}
    color_buffer = {}
    # [추가] 한 영상 내에서 분석된 최종 결과를 담는 딕셔너리
    final_results = {}

    try:
        with open(in_path, "wb") as f:
            f.write(await file.read())

        color_map = {"검은": "Black", "흰": "White", "빨간": "Red", "파란": "Blue",
                     "노란": "Yellow", "초록": "Green", "보라": "Purple", "회": "Gray", "분홍": "Pink"}
        target_color = next((v for k, v in color_map.items() if content and k in content), "")

        cap = cv2.VideoCapture(in_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

        frame_count = 0
        while cap.isOpened():
            success, frame = cap.read()
            if not success: break
            frame_count += 1

            if frame_count % 3 == 0:
                results = model.track(frame, persist=True, verbose=False, conf=0.3, device=device)
                new_tracks = {}

                if results[0].boxes.id is not None:
                    boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
                    ids = results[0].boxes.id.cpu().numpy().astype(int)
                    kpts = results[0].keypoints.xy.cpu().numpy() if results[0].keypoints is not None else []

                    for i, obj_id in enumerate(ids):
                        if obj_id in latest_track_data and frame_count % 15 != 0:
                            new_tracks[obj_id] = latest_track_data[obj_id]
                            new_tracks[obj_id]["box"] = boxes[i]
                            continue

                        roi = None
                        if i < len(kpts) and all(kpts[i][j][1] > 0 for j in [5, 6]):
                            pk = kpts[i]
                            sw = abs(pk[5][0] - pk[6][0])
                            ry1, ry2 = int(min(pk[5][1], pk[6][1])), int(min(pk[5][1], pk[6][1]) + sw * 0.5)
                            rx1, rx2 = int(min(pk[5][0], pk[6][0])), int(max(pk[5][0], pk[6][0]))
                            roi = frame[ry1:ry2, rx1:rx2]

                        if roi is not None and roi.size > 0:
                            stable_color = get_smoothed_color(obj_id, detect_color_name(roi))
                            new_tracks[obj_id] = {"box": boxes[i], "color": stable_color}
                            # [추가] 최종 결과 업데이트 (가장 최신의 안정적인 색상으로 갱신)
                            final_results[obj_id] = stable_color

                latest_track_data = new_tracks

            for obj_id, data in latest_track_data.items():
                if target_color == "" or target_color.lower() in data["color"].lower():
                    x1, y1, x2, y2 = data["box"]
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                    cv2.putText(frame, f"ID:{obj_id} {data['color']}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            out.write(frame)

        # --- [3. 분석 완료 후 DB 저장] ---
        db = SessionLocal()
        try:
            for obj_id, color in final_results.items():
                new_record = DetectionResult(
                    object_id=int(obj_id),
                    detected_color=color,
                    video_name=out_path
                )
                db.add(new_record)
            db.commit()
            print(f"✅ DB 저장 성공: {len(final_results)}건의 탐지 기록이 저장되었습니다.")
        except Exception as e:
            db.rollback()
            print(f"❌ DB 저장 중 에러 발생: {e}")
        finally:
            db.close()

    finally:
        if out: out.release()
        if cap: cap.release()
        if os.path.exists(in_path): os.remove(in_path)

    return FileResponse(out_path, media_type="video/mp4", filename=out_path)

"""
앱이나 웹에서 탐지 기록을 조회할 때 사용하는 API입니다.
특정 색상만 골라서 보고 싶을 때 /logs?color=Red 처럼 요청할 수 있습니다.
"""
@app.get("/logs")
async def get_detection_logs(color: str = Query(None)):
    """
    앱이나 웹에서 탐지 기록을 조회할 때 사용하는 API입니다.
    특정 색상만 골라서 보고 싶을 때 /logs?color=Red 처럼 요청할 수 있습니다.
    """
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



# DB 세션을 가져오는 헬퍼 함수
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
# 회원가입 API
@app.post("/register")
async def register_user(user_data: UserCreate, db: Session = Depends(get_db)):
    # 1. 비밀번호 암호화
    hashed_pwd = pwd_context.hash(user_data.password)

    # 2. 새로운 사용자 객체 생성
    new_user = User(
        username=user_data.username,
        password=hashed_pwd,
        email=user_data.email
    )

    # 3. DB에 저장 (SQLAlchemy가 INSERT 쿼리를 자동 생성)
    db.add(new_user)
    db.commit()
    db.refresh(new_user)

    return {"status": "success", "username": new_user.username}



# 템플릿 파일이 위치한 폴더를 지정합니다. (현재 static 폴더를 그대로 사용)
templates = Jinja2Templates(directory="static")

# 1. 정적 파일(CSS/JS) 연결
app.mount("/static", StaticFiles(directory="static"), name="static")

# 2. Jinja2 템플릿 엔진 설정
templates = Jinja2Templates(directory="templates")

# 메인 페이지 (http://IP:8000/ 접속 시)
@app.get("/")
async def read_index(request: Request):
    # 실제로는 세션이나 쿠키에서 로그인 여부를 확인해야 합니다.
    # 테스트를 위해 임시로 False를 넣습니다.
    is_logged_in = False

    # 템플릿 파일명과 함께 데이터를 넘겨줍니다.
    return templates.TemplateResponse("home.html", {
        "request": request,
        "is_logged_in": is_logged_in
    })



@app.get("/login")
async def read_login(request: Request, error: bool = False):
    return templates.TemplateResponse("login.html", {
        "request": request,
        "error": error # URL에 ?error=true가 붙으면 에러 메시지를 띄웁니다.
    })


