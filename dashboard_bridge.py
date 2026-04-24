"""
Althera Dashboard Bridge Module
================================
Re-implements core logic from existing project modules for web dashboard integration.
Does NOT modify any existing files — all logic is mirrored here for Streamlit compatibility.

Wraps:
  - import serial.py  → SensorReader class
  - predict_score.py  → predict_cognitive_score()
  - ai_interpretation.py → generate_ai_report()
  - emotion_detection.py → detect_emotion_from_frame()
"""

import threading
import time
import os
from datetime import datetime
from collections import deque

import numpy as np
import cv2
import joblib
import pandas as pd

# ──────────────────────────────────────────────────────────────
# 1. SENSOR READER  (mirrors import serial.py)
# ──────────────────────────────────────────────────────────────

class SensorReader:
    """
    Background thread that reads sensor data from an ESP8266/Arduino
    over serial.  Auto-detects the COM port and streams data into
    an in-memory deque (no CSV dependency).
    """

    def __init__(self, baud_rate=115200, buffer_size=500):
        self.baud_rate = baud_rate
        self.buffer = deque(maxlen=buffer_size)
        self.latest = {}
        self.connected = False
        self.port_opened = False        # True as soon as serial port opens (before MCU reset)
        self.last_error = None          # exposed for UI diagnostics
        self.port_name = None
        self._thread = None
        self._stop_event = threading.Event()
        self._serial = None

    # ---- auto-detect serial port ----
    @staticmethod
    def find_serial_port():
        """Return the first likely sensor COM port, or None.
        Uses a broad match so CH340 / CP2102 / FTDI / plain 'USB Serial' all work.
        """
        try:
            import serial.tools.list_ports
            ports = list(serial.tools.list_ports.comports())
            priority_keywords = ("ch340", "cp210", "arduino", "esp", "ftdi", "usb serial", "usb-serial")
            fallback_keywords = ("usb", "serial", "com")

            # First pass: high-confidence keywords
            for p in ports:
                desc = (p.description or "").lower()
                hwid = (p.hwid or "").lower()
                combined = desc + " " + hwid
                if any(kw in combined for kw in priority_keywords):
                    return p.device

            # Second pass: any port with a USB VID (likely a USB-serial adapter)
            for p in ports:
                hwid = (p.hwid or "").lower()
                if "vid" in hwid:
                    return p.device

            # Fallback: any port that isn't Bluetooth
            for p in ports:
                desc = (p.description or "").lower()
                if "bluetooth" not in desc and "bth" not in desc:
                    return p.device
        except Exception:
            pass
        return None

    @staticmethod
    def list_all_ports():
        """Return list of (device, description) for all available COM ports."""
        try:
            import serial.tools.list_ports
            return [(p.device, p.description) for p in serial.tools.list_ports.comports()]
        except Exception:
            return []

    # ---- start / stop ----
    def start(self, port=None):
        """Begin reading in a background thread."""
        if self._thread and self._thread.is_alive():
            return  # already running
        self._stop_event.clear()
        self.last_error = None
        self.port_name = port or self.find_serial_port()
        if not self.port_name:
            raise ConnectionError("No serial port detected. Please connect the sensor.")
        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()

    def connect_blocking(self, port=None, timeout=10):
        """
        Start the reader and block until the serial port is open OR timeout.
        Returns (True, None) on success, (False, error_str) on failure.

        We wait for port_opened (set as soon as Serial() succeeds, BEFORE
        the 2-second MCU-reset sleep) so the UI is responsive immediately.
        """
        try:
            self.start(port=port)
        except ConnectionError as e:
            return False, str(e)

        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.port_opened or self.connected:
                return True, None
            if self.last_error:           # thread reported an error
                return False, self.last_error
            time.sleep(0.15)

        if not (self.port_opened or self.connected):
            return False, (
                f"Timeout: could not open {self.port_name} within {timeout}s. "
                "Make sure Arduino IDE / Serial Monitor is closed and the device is plugged in."
            )
        return True, None

    def is_healthy(self):
        """Return True if the background thread is alive and the port is open."""
        thread_ok = self._thread is not None and self._thread.is_alive()
        return thread_ok and (self.port_opened or self.connected)

    @property
    def sample_count(self):
        return len(self.buffer)

    def stop(self):
        self._stop_event.set()
        if self._serial:
            try:
                self._serial.close()
            except Exception:
                pass
        self.connected = False

    # ---- internal read loop (mirrors import serial.py logic) ----
    def _read_loop(self):
        import serial as pyserial
        try:
            self._serial = pyserial.Serial(self.port_name, self.baud_rate, timeout=1)
            # Mark port as open IMMEDIATELY so connect_blocking() can return
            self.port_opened = True
            time.sleep(2)  # allow MCU reset / bootloader
            self.connected = True

            consecutive_empty = 0
            while not self._stop_event.is_set():
                try:
                    raw = self._serial.readline()
                except Exception as read_err:
                    self.last_error = f"Read error: {read_err}"
                    break

                if not raw:
                    consecutive_empty += 1
                    if consecutive_empty > 30:   # ~30s of silence
                        self.last_error = "No data received from sensor — is it sending data?"
                    continue
                consecutive_empty = 0

                line = raw.decode("utf-8", errors="ignore").strip()

                # ── Accept lines that start with "AX" OR contain "HR" (flexible format) ──
                if not (line.startswith("AX") or "HR" in line or "HeartRate" in line):
                    continue

                parts = line.split(",")
                data = {}
                for item in parts:
                    if ":" in item:
                        key, value = item.split(":", 1)
                        try:
                            data[key] = float(value.strip())
                        except ValueError:
                            data[key] = value.strip()

                # Support both "HR" and "HeartRate" keys from the sensor
                heart_rate = data.get("HR", data.get("HeartRate", 0))
                spo2 = data.get("SpO2", data.get("SPO2", 0))

                timestamp = datetime.now().strftime("%H:%M:%S")
                record = {
                    "Timestamp": timestamp,
                    "AX": data.get("AX", 0),
                    "AY": data.get("AY", 0),
                    "AZ": data.get("AZ", 0),
                    "Motion": data.get("Motion", 0),
                    "HeartRate": heart_rate,
                    "SpO2": spo2,
                }
                self.buffer.append(record)
                self.latest = record
                self.last_error = None   # clear any stale error on successful read

        except Exception as e:
            self.connected = False
            self.last_error = str(e)
            self.latest = {"error": str(e)}
        finally:
            if self._serial:
                try:
                    self._serial.close()
                except Exception:
                    pass
            self.connected = False

    # ---- convenience ----
    def get_latest(self):
        return dict(self.latest)

    def get_buffer_df(self):
        if not self.buffer:
            return pd.DataFrame()
        return pd.DataFrame(list(self.buffer))

    def get_averages(self):
        df = self.get_buffer_df()
        if df.empty:
            return {"HeartRate": 0, "SpO2": 0, "Motion": 0}
        return {
            "HeartRate": round(df["HeartRate"].astype(float).mean(), 1),
            "SpO2": round(df["SpO2"].astype(float).mean(), 1),
            "Motion": round(df["Motion"].astype(float).mean(), 2),
        }


# ──────────────────────────────────────────────────────────────
# 2. ML PREDICTION  (mirrors predict_score.py + train_model.py)
# ──────────────────────────────────────────────────────────────

MODEL_PATH = os.path.join(os.path.dirname(__file__), "cognitive_model.pkl")

FEATURE_COLUMNS = [
    "HeartRate",
    "SpO2",
    "Motion",
    "simple_reaction_ms",
    "choice_reaction_ms",
    "finger_taps",
    "word_recall_score",
    "number_recall_score",
    "stroop_accuracy_percent",
]


def predict_cognitive_score(data_dict: dict) -> tuple:
    """
    Load the trained RandomForest model and predict a cognitive score.

    Parameters
    ----------
    data_dict : dict
        Must contain keys matching FEATURE_COLUMNS.

    Returns
    -------
    (score: float, status: str)
    """
    model = joblib.load(MODEL_PATH)
    row = {col: float(data_dict.get(col, 0)) for col in FEATURE_COLUMNS}
    df = pd.DataFrame([row])
    score = float(model.predict(df)[0])
    score = round(score, 2)

    # Same interpretation thresholds as predict_score.py
    if score > 85:
        status = "Excellent cognitive performance"
    elif score > 70:
        status = "Normal cognitive performance"
    elif score > 50:
        status = "Mild cognitive fatigue"
    else:
        status = "Low cognitive performance — further evaluation recommended"

    return score, status


# ──────────────────────────────────────────────────────────────
# 3. AI INTERPRETATION  (mirrors ai_interpretation.py)
# ──────────────────────────────────────────────────────────────

def generate_ai_report(data_dict: dict) -> str:
    """
    Send patient data to Ollama (llama3.2) and return a natural-language
    cognitive health report.  Same prompt template as ai_interpretation.py.
    """
    prompt = f"""
Analyze the following cognitive health data for a patient undergoing cognitive assessment.

--- Sensor Readings ---
Heart Rate: {data_dict.get('HeartRate', 'N/A')} bpm
SpO2: {data_dict.get('SpO2', 'N/A')} %
Motion Level: {data_dict.get('Motion', 'N/A')}

--- Reaction Tests ---
Simple Reaction Time: {data_dict.get('simple_reaction_ms', 'N/A')} ms
Choice Reaction Time: {data_dict.get('choice_reaction_ms', 'N/A')} ms
Choice Accuracy: {data_dict.get('choice_accuracy_percent', 'N/A')} %
Finger Taps (per 5s): {data_dict.get('finger_taps', 'N/A')}

--- Memory Tests ---
Word Recall Score: {data_dict.get('word_recall_score', 'N/A')} / 5
Number Recall Score: {data_dict.get('number_recall_score', 'N/A')} / 6
Stroop Test Accuracy: {data_dict.get('stroop_accuracy_percent', 'N/A')} %

--- Emotion Detection ---
Dominant Emotion: {data_dict.get('dominant_emotion', 'N/A')}

--- ML Prediction ---
Predicted Cognitive Score: {data_dict.get('cognitive_score', 'N/A')} / 100
Status: {data_dict.get('cognitive_status', 'N/A')}

Provide a detailed report with:
1. Cognitive health summary
2. Possible concerns
3. Suggestions for improvement
"""
    try:
        import ollama
        response = ollama.chat(
            model="llama3.2",
            messages=[{"role": "user", "content": prompt}],
        )
        # Handle both old dict-style and new attribute-style access
        try:
            return response["message"]["content"]
        except (TypeError, KeyError):
            return response.message.content
    except ImportError:
        return (
            "⚠️ The `ollama` Python package is not installed.\n\n"
            "Install it with: `pip install ollama`"
        )
    except Exception as e:
        error_msg = str(e)
        if "model" in error_msg.lower() or "not found" in error_msg.lower():
            try:
                import subprocess
                # Attempt to pull the model automatically
                subprocess.run(["ollama", "pull", "llama3.2"], check=True)
                
                # Retry once after pulling
                import ollama
                response = ollama.chat(
                    model="llama3.2",
                    messages=[{"role": "user", "content": prompt}],
                )
                try:
                    return response["message"]["content"]
                except (TypeError, KeyError):
                    return response.message.content
            except Exception as pull_error:
                return (
                    f"⚠️ Model missing and auto-download failed.\n\n"
                    f"Error: {pull_error}\n\n"
                    "**Fix:** Open a terminal and run:\n"
                    "`ollama pull llama3.2`"
                )
        elif "connect" in error_msg.lower() or "refused" in error_msg.lower():
            return (
                f"⚠️ Cannot connect to Ollama service.\n\n"
                f"Error: {e}\n\n"
                "**Fix:** The service is starting up... Please wait a moment and try again."
            )
        else:
            return (
                f"⚠️ AI report generation failed.\n\n"
                f"Error: {e}"
            )


# ──────────────────────────────────────────────────────────────
# 4. EMOTION DETECTION  (mirrors emotion_detection.py)
# ──────────────────────────────────────────────────────────────

# Load cascade once at module level
_FACE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


def detect_emotion_from_frame(frame):
    """
    Run face detection + rule-based emotion classification on a single
    BGR frame.  Same logic as emotion_detection.py.

    Returns
    -------
    (annotated_frame, emotion_label)
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = _FACE_CASCADE.detectMultiScale(gray, 1.3, 5)

    emotion = "neutral"

    for (x, y, w, h) in faces:
        # Same rule-based logic as emotion_detection.py
        if h > 250:
            emotion = "happy"
        elif h < 120:
            emotion = "sad"
        else:
            emotion = "neutral"

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.putText(
        frame,
        emotion,
        (50, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2,
    )

    return frame, emotion


# ──────────────────────────────────────────────────────────────
# 5. VOICE ENGINE  (mirrors pyttsx3 usage in reaction_Test.py / memory_Test.py)
# ──────────────────────────────────────────────────────────────

class VoiceEngine:
    """
    Thread-safe text-to-speech wrapper for Streamlit.
    Runs pyttsx3 in a background thread so it never blocks the UI.
    Calling speak() always interrupts any currently-playing speech
    so voice stays in sync with the user's current action.
    """

    def __init__(self, rate=135, volume=1.0):
        self.rate = rate
        self.volume = volume
        self.enabled = True
        self._lock = threading.Lock()
        self._current_engine = None   # pyttsx3 engine currently speaking
        self._speak_thread = None

    def stop(self):
        """Immediately interrupt any speech in progress."""
        engine = self._current_engine
        if engine is not None:
            try:
                engine.stop()
            except Exception:
                pass

    def speak(self, text: str):
        """Interrupt current speech (if any) and speak new text."""
        if not self.enabled:
            return
        # Signal the running engine to stop, then start fresh
        self.stop()
        t = threading.Thread(target=self._speak_sync, args=(text,), daemon=True)
        self._speak_thread = t
        t.start()

    def _speak_sync(self, text: str):
        """Internal: run pyttsx3 synchronously (called from per-utterance thread)."""
        with self._lock:          # serialize actual playback
            try:
                import pyttsx3
                engine = pyttsx3.init()
                self._current_engine = engine
                engine.setProperty("rate", self.rate)
                engine.setProperty("volume", self.volume)
                engine.say(text)
                engine.runAndWait()
            except Exception:
                pass  # silently fail if audio unavailable
            finally:
                try:
                    if self._current_engine is not None:
                        self._current_engine.stop()
                except Exception:
                    pass
                self._current_engine = None

    def set_enabled(self, enabled: bool):
        self.enabled = enabled
        if not enabled:
            self.stop()


# ──────────────────────────────────────────────────────────────
# 6. OLLAMA SERVICE MANAGEMENT
# ──────────────────────────────────────────────────────────────

def ensure_ollama_running():
    """
    Check if Ollama is responding on 11434. If not, try to start it.
    Returns (True, "Ready") or (True, "Started") or (False, "Error message").
    """
    import socket
    import subprocess

    # 1. Check if port is already open
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(1)
    result = sock.connect_ex(('127.0.0.1', 11434))
    sock.close()

    if result == 0:
        return True, "Ready"

    # 2. If not, try to start it
    try:
        # Launch in background, no console window (on Windows)
        subprocess.Popen(
            ["ollama", "serve"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
        )
        return True, "Started"
    except Exception as e:
        return False, f"Could not auto-start Ollama: {e}"
