import cv2
import os
import numpy as np
import time
import torch
from collections import deque, Counter
from fastapi import FastAPI, File, UploadFile, Query, BackgroundTasks
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
from fastapi import HTTPException, Response

# --- [1. 데이터베이스 설정 구간] ---
DATABASE_URL = "mysql+pymysql://root:0727@localhost:3306/safety_db?charset=utf8mb4"
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


# 비밀번호 암호화 설정
pwd_context = CryptContext(schemes=["pbkdf2_sha256"], deprecated="auto")
# 회원가입 시 받을 데이터 규격 (추후 수정)
class UserCreate(BaseModel):
    id: str
    password: str
    residentFront: str
    residentBack: str
    name: str
    phone: str

# 앱에서 서버로 신고 정보를 보낼 때의 형식
class ReportCreate(BaseModel):
    name: str
    phone_number: str
    ssn: str
    content: str

# 로그인 시 정보 확인용 class
class UserLogin(BaseModel):
    username: str
    password: str

# 비밀번호 암호화 함수
def get_password_hash(password):
    return pwd_context.hash(password)

# 기존 Base를 상속받아 users 테이블 정의
class User(Base):
    __tablename__ = "users"

    user_no = Column(Integer, primary_key=True, index=True) # DB 관리용 번호
    id = Column(String(50), unique=True, nullable=False)   # 로그인 아이디
    password = Column(String(255), nullable=False)          # 암호화된 비밀번호
    residentFront = Column(String(6), nullable=False)           # 주민번호 앞자리
    residentBack = Column(String(255), nullable=False)          # 암호화된 주민번호 뒷자리
    name = Column(String(50), nullable=False)               # 사용자 이름
    phone_number = Column(String(20), nullable=False)       # 휴대폰 번호
    created_at = Column(DateTime, default=datetime.datetime.now) # 가입일

# 신고 내역 저장을 위한 테이블 모델
class IncidentReport(Base):
    __tablename__ = "incident_reports"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(50), nullable=False) # 신고자 성함
    phone_number = Column(String(20), nullable=False) # 전화번호
    ssn = Column(String(255), nullable=False) # 주민번호 (암호화 저장)
    content = Column(String(500), nullable=True) # 신고 상세 내용
    video_path = Column(String(200), nullable=True) # 분석할 영상 경로
    created_at = Column(DateTime, default=datetime.datetime.now)



# 테이블 자동 생성 (DB 탭에서 확인 가능)
Base.metadata.create_all(bind=engine)


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
영상 분석할 때 사용하는 함수입니다.
"""
# [수정] 기존 analyze_video의 핵심 로직을 함수로 분리
def process_video_analysis(report_id: int, content: str = None):
    # 별도의 DB 세션을 생성합니다. (백그라운드 작업용)
    db = SessionLocal()

    global latest_track_data, color_buffer
    ts = int(time.time())

    # [주의] 실제 서비스에서는 신고 내용에 포함된 비디오 경로를 사용해야 합니다.
    # 일단 현재 코드의 test.mp4를 유지합니다.
    in_path, out_path = "test.mp4", f"out_{ts}_{report_id}.mp4"
    cap, out = None, None

    latest_track_data = {}
    color_buffer = {}
    final_results = {}

    try:
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
                # YOLO 추적 로직 실행
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

                        # 포즈 기반 ROI 추출 및 색상 감지
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
                            final_results[obj_id] = stable_color

                latest_track_data = new_tracks

            # 결과 그리기 및 저장
            for obj_id, data in latest_track_data.items():
                if target_color == "" or target_color.lower() in data["color"].lower():
                    x1, y1, x2, y2 = data["box"]
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                    cv2.putText(frame, f"ID:{obj_id} {data['color']}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            out.write(frame)

        # --- [DB 저장 단계] ---
        try:
            for obj_id, color in final_results.items():
                new_record = DetectionResult(
                    object_id=int(obj_id),
                    detected_color=color,
                    video_name=out_path
                    # 여기에 report_id 컬럼이 있다면 추가: report_id=report_id
                )
                db.add(new_record)

            # [추가] 신고서 테이블에도 분석된 비디오 경로 업데이트
            report = db.query(IncidentReport).filter(IncidentReport.id == report_id).first()
            if report:
                report.video_path = out_path

            db.commit()
            print(f"✅ 신고번호 {report_id}: {len(final_results)}건 탐지 기록 저장 완료")
        except Exception as e:
            db.rollback()
            print(f"❌ 분석 결과 저장 실패: {e}")

    finally:
        if out: out.release()
        if cap: cap.release()
        db.close()


# DB 세션을 가져오는 헬퍼 함수
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

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




# 회원가입 API
@app.post("/register")
async def register_user(user_data: UserCreate, db: Session = Depends(get_db)):
    # 1. 비밀번호와 주민번호 뒷자리를 암호화합니다.
    hashed_pwd = pwd_context.hash(user_data.password)
    hashed_resident_back = pwd_context.hash(user_data.residentBack)

    # 2. 새로운 사용자 객체 생성
    new_user = User(
        id=user_data.id,
        username=user_data.username,
        password=hashed_pwd,
        residentFront=user_data.residentFront,
        ssn_back=hashed_resident_back,
        name=user_data.name,
        phone=user_data.phone
    )

    # 3. DB에 저장 (SQLAlchemy가 INSERT 쿼리를 자동 생성)
    db.add(new_user)
    db.commit()
    db.refresh(new_user)

    return {"status": "success", "username": new_user.username, "name": new_user.name}

# 신고 접수 API
@app.post("/report/submit")
async def submit_report(
        report: ReportCreate,
        background_tasks: BackgroundTasks,
        db: Session = Depends(get_db)
):
    # 1. 주민번호 암호화 저장
    hashed_ssn = pwd_context.hash(report.ssn)

    new_report = IncidentReport(
        name=report.name,
        phone_number=report.phone_number,
        ssn=hashed_ssn,
        content=report.content
    )

    db.add(new_report)
    db.commit()
    db.refresh(new_report)

    # 2. [핵심] 비디오 분석 함수를 백그라운드 작업으로 등록
    # 사용자가 보낸 신고내용(content)을 함께 넘겨 특정 색상을 찾게 합니다.
    background_tasks.add_task(process_video_analysis, report_id=new_report.id, content=report.content)

    return {
        "status": "success",
        "message": "신고 접수가 완료되었습니다. 비디오 분석이 백그라운드에서 시작됩니다.",
        "report_id": new_report.id
    }

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


# 로그인 화면 연결
@app.post("/login")
async def login(user_data: UserLogin, db: Session = Depends(get_db)):
    # 1. DB에서 해당 아이디의 사용자 찾기
    user = db.query(User).filter(User.username == user_data.username).first()

    # 2. 사용자가 없거나 비밀번호가 틀린 경우
    if not user or not pwd_context.verify(user_data.password, user.password):
        raise HTTPException(status_code=400, detail="아이디 또는 비밀번호가 틀렸습니다.")

    # 3. 로그인 성공 시 응답 (나중에 세션이나 토큰을 추가할 수 있습니다)
    return {
        "status": "success",
        "message": f"{user.name}님 환영합니다!",
        "redirect_url": "/admin" # 로그인 성공 시 관리자 페이지로 이동 제안
    }

# 회원가입 화면 연결
@app.get("/signup")
async def signup_page(request: Request):
    return templates.TemplateResponse("signup.html", {"request": request})

# 관리자 화면 연결
@app.get("/admin")
async def admin_page(request: Request):
    return templates.TemplateResponse("admin.html", {"request": request})

# (선택) 아까 home.html 메뉴에 있던 다른 페이지들도 미리 만들어두면 좋습니다.
@app.get("/news")
async def news_page(request: Request):
    return templates.TemplateResponse("news.html", {"request": request})