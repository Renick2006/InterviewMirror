import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import cv2
import time
import threading
import mediapipe as mp
import uuid
import json
import numpy as np
import sounddevice as sd
from datetime import datetime

from gtts import gTTS
from playsound import playsound

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas


# =========================
# QUESTIONS
# =========================
QUESTIONS = {
    "Machine Learning": [
        "Introduce yourself in 30 seconds.",
        "Explain overfitting and how to handle it.",
        "Difference between supervised and unsupervised learning?",
        "Explain precision, recall, and F1-score.",
        "What is gradient descent?"
    ],
    "Web Development": [
        "Introduce yourself in 30 seconds.",
        "Difference between GET and POST?",
        "Explain REST API in simple terms.",
        "What is CORS and why does it happen?",
        "Explain authentication using JWT or sessions."
    ],
    "AWS/Cloud": [
        "Introduce yourself in 30 seconds.",
        "What is EC2 and why is it used?",
        "Difference between S3 and EBS?",
        "Explain IAM roles and policies.",
        "What is auto scaling and load balancing?"
    ]
}

QUESTION_TIME = 45
HISTORY_FILE = "history.json"
REPORTS_FOLDER = "reports"


# =========================
# DOMAIN SELECTION
# =========================
def choose_domain():
    print("\nChoose Interview Domain:")
    print("1. Machine Learning")
    print("2. Web Development")
    print("3. AWS/Cloud")

    choice = input("\nEnter choice (1/2/3): ").strip()

    if choice == "1":
        return "Machine Learning"
    elif choice == "2":
        return "Web Development"
    elif choice == "3":
        return "AWS/Cloud"
    else:
        print("Invalid choice. Defaulting to Machine Learning.")
        return "Machine Learning"


# =========================
# AUDIO SPEAK (gTTS)
# =========================
def speak(text):
    def run():
        try:
            filename = f"tts_{uuid.uuid4().hex}.mp3"
            tts = gTTS(text=text, lang="en")
            tts.save(filename)
            playsound(filename)
            os.remove(filename)
        except Exception as e:
            print("TTS Error:", e)

    threading.Thread(target=run, daemon=True).start()


# =========================
# HISTORY FUNCTIONS
# =========================
def load_history():
    if not os.path.exists(HISTORY_FILE):
        return []
    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except:
        return []


def save_history(history):
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=4)


def add_session_to_history(session):
    history = load_history()
    history.append(session)
    save_history(history)


def get_last_sessions(domain=None, n=5):
    history = load_history()
    if domain:
        history = [h for h in history if h.get("domain") == domain]
    return history[-n:]


def best_session(domain=None):
    history = load_history()
    if domain:
        history = [h for h in history if h.get("domain") == domain]
    if not history:
        return None
    return max(history, key=lambda x: x.get("confidence", 0))


# =========================
# PDF REPORT EXPORT
# =========================
def export_pdf_report(session, tips, questions):
    os.makedirs(REPORTS_FOLDER, exist_ok=True)

    timestamp = session["timestamp"].replace(":", "-")
    filename = f"InterviewMirror_Report_{timestamp}.pdf"
    filepath = os.path.join(REPORTS_FOLDER, filename)

    c = canvas.Canvas(filepath, pagesize=A4)
    width, height = A4

    y = height - 60

    c.setFont("Helvetica-Bold", 18)
    c.drawString(60, y, "InterviewMirror - Interview Report")
    y -= 35

    c.setFont("Helvetica", 11)
    c.drawString(60, y, f"Date: {session['timestamp']}")
    y -= 20
    c.drawString(60, y, f"Domain: {session['domain']}")
    y -= 30

    c.setFont("Helvetica-Bold", 13)
    c.drawString(60, y, "Scores")
    y -= 18

    c.setFont("Helvetica", 11)
    c.drawString(70, y, f"Eye Contact: {session['eye_contact']}%")
    y -= 18
    c.drawString(70, y, f"Head Stability: {session['head_stability']}%")
    y -= 18
    c.drawString(70, y, f"Voice Energy: {session['voice_energy_score']}%")
    y -= 18
    c.drawString(70, y, f"Silence Control: {session['silence_score']}%")
    y -= 18
    c.drawString(70, y, f"Overall Confidence: {session['confidence']}%")
    y -= 18
    c.drawString(70, y, f"Avg Time per Question: {session['avg_time_per_question']} seconds")
    y -= 30

    c.setFont("Helvetica-Bold", 13)
    c.drawString(60, y, "Questions Asked")
    y -= 18

    c.setFont("Helvetica", 11)
    for i, q in enumerate(questions, 1):
        if y < 140:
            c.showPage()
            y = height - 60
            c.setFont("Helvetica", 11)
        c.drawString(70, y, f"{i}. {q}")
        y -= 18

    y -= 10
    c.setFont("Helvetica-Bold", 13)
    c.drawString(60, y, "Feedback & Tips")
    y -= 18

    c.setFont("Helvetica", 11)
    for i, tip in enumerate(tips, 1):
        if y < 140:
            c.showPage()
            y = height - 60
            c.setFont("Helvetica", 11)
        short_tip = tip[:110] + "..." if len(tip) > 112 else tip
        c.drawString(70, y, f"{i}. {short_tip}")
        y -= 18

    y -= 20
    c.setFont("Helvetica-Oblique", 10)
    c.drawString(60, y, "Generated by InterviewMirror (OpenCV + MediaPipe + Audio Analytics)")

    c.save()
    return filepath


# =========================
# VOICE ANALYSIS
# =========================
voice_rms_values = []
speaking_frames = 0
silent_frames = 0
VOICE_THRESHOLD = 0.015


def audio_callback(indata, frames, time_info, status):
    global speaking_frames, silent_frames

    volume_norm = np.linalg.norm(indata) / len(indata)
    voice_rms_values.append(volume_norm)

    if volume_norm > VOICE_THRESHOLD:
        speaking_frames += 1
    else:
        silent_frames += 1


def start_audio_stream():
    stream = sd.InputStream(callback=audio_callback, channels=1, samplerate=16000)
    stream.start()
    return stream


# =========================
# UI UTILS
# =========================
def draw_box_text(frame, text, x, y, w, h):
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x + w, y + h), (30, 30, 30), -1)
    alpha = 0.65
    frame[:] = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

    words = text.split(" ")
    lines = []
    line = ""
    for word in words:
        if len(line + " " + word) < 42:
            line += " " + word
        else:
            lines.append(line.strip())
            line = word
    lines.append(line.strip())

    for i, l in enumerate(lines[:3]):
        cv2.putText(frame, l, (x + 15, y + 35 + i * 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)


def draw_center_text(frame, text, y, scale=1.0, thickness=2):
    h, w, _ = frame.shape
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    x = (w - tw) // 2
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), thickness)


# =========================
# FEEDBACK ENGINE
# =========================
def generate_tips(eye_score, head_score, confidence, avg_time, voice_score, silence_score):
    tips = []

    if eye_score < 45:
        tips.append("Eye contact was low. Look at the camera lens more often.")
    elif eye_score < 70:
        tips.append("Eye contact is decent. Hold it for 2 to 3 seconds.")
    else:
        tips.append("Strong eye contact. Keep it consistent.")

    if head_score < 45:
        tips.append("High head movement. Keep your head steadier while speaking.")
    elif head_score < 70:
        tips.append("Head stability is okay. Reduce unnecessary movement.")
    else:
        tips.append("Good head stability. You look calm.")

    if voice_score < 50:
        tips.append("Voice energy was low. Speak louder and with clarity.")
    elif voice_score < 75:
        tips.append("Voice energy is okay. Add more emphasis on key points.")
    else:
        tips.append("Strong voice energy. You sound confident.")

    if silence_score < 55:
        tips.append("Many silent pauses. Use a structure to reduce hesitation.")
    else:
        tips.append("Pauses are controlled. Good speaking flow.")

    if avg_time < 18:
        tips.append("Answers were too short. Add an example or detail.")
    elif avg_time > 38:
        tips.append("Answers may be too long. Use: What, Why, Example, then stop.")
    else:
        tips.append("Answer length is balanced. Keep it crisp.")

    tips.append(
        "If you don’t know an answer, say: "
        "'I’m not sure right now, but I’ll read up on that and get back with a clearer answer.'"
    )

    tips.append("Avoid filler words. Pause instead.")
    tips.append("End answers with a confident summary line.")

    return tips


# =========================
# FINAL REPORT SCREEN
# =========================
def show_final_report(scores, tips, pdf_path, domain):
    last_sessions = get_last_sessions(domain, n=5)
    best = best_session(domain)

    # Improvement
    if len(last_sessions) >= 2:
        change = last_sessions[-1]["confidence"] - last_sessions[-2]["confidence"]
    else:
        change = None

    # Prepare confidence values
    conf_values = [s["confidence"] for s in last_sessions]

    while True:
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)

        draw_center_text(frame, "InterviewMirror - Final Report", 55, 1.2, 3)

        # Scores
        y = 110
        for line in scores:
            draw_center_text(frame, line, y, 0.85, 2)
            y += 40

        cv2.putText(frame, f"PDF Saved: {pdf_path}", (60, 350),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)

        # Tips panel
        cv2.rectangle(frame, (60, 380), (620, 680), (25, 25, 25), -1)
        cv2.putText(frame, "Top Tips", (90, 420),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

        tip_y = 460
        for i, tip in enumerate(tips[:5], 1):
            short = tip[:65] + "..." if len(tip) > 68 else tip
            cv2.putText(frame, f"{i}. {short}", (90, tip_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            tip_y += 38

        # History + Graph panel
        cv2.rectangle(frame, (660, 380), (1220, 680), (25, 25, 25), -1)
        cv2.putText(frame, "Confidence Trend (Last 5)", (690, 420),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        # Graph area
        gx1, gy1 = 700, 450
        gx2, gy2 = 1180, 620

        # Graph border
        cv2.rectangle(frame, (gx1, gy1), (gx2, gy2), (120, 120, 120), 1)

        # Draw y-axis labels
        cv2.putText(frame, "100", (gx1 - 45, gy1 + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)
        cv2.putText(frame, "0", (gx1 - 25, gy2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)

        # Plot points
        if len(conf_values) >= 1:
            step = (gx2 - gx1) // max(1, len(conf_values) - 1)

            points = []
            for i, conf in enumerate(conf_values):
                x = gx1 + (i * step)
                y = gy2 - int((conf / 100) * (gy2 - gy1))
                points.append((x, y))

            # Connect points
            for i in range(len(points) - 1):
                cv2.line(frame, points[i], points[i + 1], (0, 255, 255), 2)

            # Draw dots + values
            for i, (x, y) in enumerate(points):
                cv2.circle(frame, (x, y), 6, (0, 255, 0), -1)
                cv2.putText(frame, f"{conf_values[i]}",
                            (x - 10, y - 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

        # Best score
        if best:
            cv2.putText(frame, f"Best: {best['confidence']}% ({best['timestamp']})",
                        (690, 655), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)

        # Improvement
        if change is not None:
            sign = "+" if change >= 0 else ""
            cv2.putText(frame, f"Improvement from last session: {sign}{change}%",
                        (60, 710), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 2)

        cv2.putText(frame, "Press R to Restart | Press Q to Quit",
                    (820, 715), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (200, 200, 200), 2)

        cv2.imshow("InterviewMirror Report", frame)

        key = cv2.waitKey(0) & 0xFF
        cv2.destroyWindow("InterviewMirror Report")

        if key == ord("r"):
            return True
        elif key == ord("q"):
            return False


# =========================
# MAIN INTERVIEW
# =========================
def run_interview():
    global voice_rms_values, speaking_frames, silent_frames
    voice_rms_values = []
    speaking_frames = 0
    silent_frames = 0

    domain = choose_domain()
    questions = QUESTIONS[domain]

    print("🎤 Microphone listening started...")
    audio_stream = start_audio_stream()

    mp_face = mp.solutions.face_mesh
    face_mesh = mp_face.FaceMesh(
        refine_landmarks=True,
        max_num_faces=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Webcam not detected!")
        audio_stream.stop()
        return False

    cv2.namedWindow("InterviewMirror", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("InterviewMirror", 1280, 720)

    q_index = 0
    q_start_time = time.time()

    total_frames = 0
    eye_contact_frames = 0
    stable_head_frames = 0

    last_nose = None
    STABLE_THRESHOLD = 0.012

    question_times = []
    question_enter_time = time.time()

    speak(f"Domain selected: {domain}.")
    time.sleep(0.4)
    speak(f"Question 1. {questions[q_index]}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        total_frames += 1

        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]

            left_eye = face_landmarks.landmark[33]
            right_eye = face_landmarks.landmark[263]
            nose = face_landmarks.landmark[1]

            eye_mid_x = (left_eye.x + right_eye.x) / 2
            if abs(nose.x - eye_mid_x) < 0.04:
                eye_contact_frames += 1

            if last_nose is not None:
                dx = abs(nose.x - last_nose[0])
                dy = abs(nose.y - last_nose[1])
                if (dx + dy) < STABLE_THRESHOLD:
                    stable_head_frames += 1

            last_nose = (nose.x, nose.y)

        elapsed = time.time() - q_start_time
        remaining = max(0, QUESTION_TIME - int(elapsed))

        if elapsed >= QUESTION_TIME:
            question_times.append(time.time() - question_enter_time)

            if q_index < len(questions) - 1:
                q_index += 1
                q_start_time = time.time()
                question_enter_time = time.time()
                speak(f"Question {q_index+1}. {questions[q_index]}")
            else:
                break

        eye_score = int((eye_contact_frames / max(1, total_frames)) * 100)
        head_score = int((stable_head_frames / max(1, total_frames)) * 100)
        confidence = int((0.6 * eye_score) + (0.4 * head_score))

        voice_energy = np.mean(voice_rms_values) if len(voice_rms_values) else 0
        voice_score = int(min(100, max(0, (voice_energy / 0.04) * 100)))

        total_voice_frames = speaking_frames + silent_frames
        silence_ratio = silent_frames / max(1, total_voice_frames)
        silence_score = int(max(0, 100 - (silence_ratio * 100)))

        cv2.putText(frame, f"InterviewMirror | Domain: {domain}",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 255), 2)

        cv2.putText(frame, f"Q{q_index+1}/{len(questions)}",
                    (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        draw_box_text(frame, questions[q_index], 20, 100, 900, 120)

        cv2.putText(frame, f"Time Left: {remaining}s",
                    (20, 260), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 0), 2)

        panel_x1 = w - 360
        panel_x2 = w - 20

        cv2.rectangle(frame, (panel_x1, 290), (panel_x2, 500), (25, 25, 25), -1)

        cv2.putText(frame, f"Eye Contact: {eye_score}%", (panel_x1 + 15, 330),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2)

        cv2.putText(frame, f"Head Stability: {head_score}%", (panel_x1 + 15, 365),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2)

        cv2.putText(frame, f"Voice Energy: {voice_score}%", (panel_x1 + 15, 400),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2)

        cv2.putText(frame, f"Silence Control: {silence_score}%", (panel_x1 + 15, 435),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2)

        cv2.putText(frame, f"Confidence: {confidence}%", (panel_x1 + 15, 475),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 255, 255), 2)

        cv2.putText(frame, "N:Next  P:Prev  Q:Quit  F:Full  E:Exit",
                    (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

        cv2.imshow("InterviewMirror", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            question_times.append(time.time() - question_enter_time)
            break

        elif key == ord("n"):
            question_times.append(time.time() - question_enter_time)
            if q_index < len(questions) - 1:
                q_index += 1
                q_start_time = time.time()
                question_enter_time = time.time()
                speak(f"Question {q_index+1}. {questions[q_index]}")

        elif key == ord("p"):
            question_times.append(time.time() - question_enter_time)
            if q_index > 0:
                q_index -= 1
                q_start_time = time.time()
                question_enter_time = time.time()
                speak(f"Question {q_index+1}. {questions[q_index]}")

        elif key == ord("f"):
            cv2.setWindowProperty("InterviewMirror", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        elif key == ord("e"):
            cv2.setWindowProperty("InterviewMirror", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)

    cap.release()
    cv2.destroyAllWindows()

    audio_stream.stop()
    audio_stream.close()

    eye_score = int((eye_contact_frames / max(1, total_frames)) * 100)
    head_score = int((stable_head_frames / max(1, total_frames)) * 100)
    confidence = int((0.6 * eye_score) + (0.4 * head_score))

    avg_time = sum(question_times) / len(question_times) if question_times else 0

    voice_energy = np.mean(voice_rms_values) if len(voice_rms_values) else 0
    voice_score = int(min(100, max(0, (voice_energy / 0.04) * 100)))

    total_voice_frames = speaking_frames + silent_frames
    silence_ratio = silent_frames / max(1, total_voice_frames)
    silence_score = int(max(0, 100 - (silence_ratio * 100)))

    session = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "domain": domain,
        "eye_contact": eye_score,
        "head_stability": head_score,
        "voice_energy_score": voice_score,
        "silence_score": silence_score,
        "confidence": confidence,
        "avg_time_per_question": round(avg_time, 1)
    }

    add_session_to_history(session)

    tips = generate_tips(eye_score, head_score, confidence, avg_time, voice_score, silence_score)
    pdf_path = export_pdf_report(session, tips, questions)

    scores = [
        f"Domain: {domain}",
        f"Eye Contact: {eye_score}%",
        f"Head Stability: {head_score}%",
        f"Voice Energy: {voice_score}%",
        f"Silence Control: {silence_score}%",
        f"Confidence: {confidence}%"
    ]

    # Improvement voice summary
    last_sessions = get_last_sessions(domain, n=5)
    if len(last_sessions) >= 2:
        change = last_sessions[-1]["confidence"] - last_sessions[-2]["confidence"]
        if change > 0:
            speak(f"Nice. Your confidence improved by {change} percent from last time.")
        elif change < 0:
            speak(f"Your confidence dropped by {abs(change)} percent from last time. Focus on the tips.")
        else:
            speak("Your confidence stayed the same as last time.")

    speak("Interview complete. Here is your feedback.")
    time.sleep(0.6)

    for tip in tips[:4]:
        speak(tip)
        time.sleep(0.8)

    print("\n✅ PDF report saved at:", pdf_path)

    restart = show_final_report(scores, tips, pdf_path, domain)
    return restart


# =========================
# APP LOOP
# =========================
if __name__ == "__main__":
    while True:
        again = run_interview()
        if not again:
            break
