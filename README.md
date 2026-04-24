# 🧠 Althera Cognitive Health Monitor

![Althera Hero Image](https://img.shields.io/badge/Status-Active-brightgreen)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-Framework-FF4B4B)
![Ollama](https://img.shields.io/badge/AI-Ollama%20%7C%20Llama%203.2-yellow)
![License](https://img.shields.io/badge/License-MIT-purple)

**Althera** is a comprehensive, real-time cognitive monitoring dashboard designed as an early indicator detection system for Alzheimer's and general cognitive decline. It integrates live physiological sensor data, interactive cognitive tests, facial emotion detection, machine learning, and generative AI to provide a holistic assessment of an individual's cognitive health.

---

## ✨ Features

Althera provides a full end-to-end pipeline for cognitive assessment:

*   **📡 Real-Time Sensor Integration:** Connects seamlessly with ESP8266/Arduino-based sensors over serial to monitor Heart Rate (bpm), SpO2 (%), and Motion levels. Includes a simulated data mode if hardware is unavailable.
*   **⚡ Reaction Tests:** Measures psychomotor speed and decision-making through:
    *   Simple Reaction Test
    *   Choice Reaction Test (Color matching)
    *   Finger Tapping Test (Taps per 5 seconds)
*   **🧩 Memory & Cognitive Tests:** Evaluates short-term memory and executive function:
    *   Word Recall
    *   Number Recall
    *   Stroop Test (Color/Word interference)
*   **😊 Facial Emotion Detection:** Uses your webcam and OpenCV to analyze your facial expressions and emotional state during the assessment.
*   **🤖 ML Cognitive Score Prediction:** A trained Random Forest machine learning model analyzes all aggregated data to predict an overall cognitive score (0-100) and performance status.
*   **🧬 Generative AI Interpretation:** Leverages local LLMs via **Ollama (Llama 3.2)** to generate a detailed, natural-language cognitive health report, highlighting insights, concerns, and suggestions.
*   **🔊 Voice Assistant:** Built-in text-to-speech (TTS) to guide users through the testing process hands-free.

---

## 🛠️ Technology Stack

*   **Frontend & Dashboard:** Streamlit, Plotly
*   **Computer Vision:** OpenCV (`opencv-python`)
*   **Hardware Interface:** PySerial
*   **Machine Learning:** Scikit-learn, Joblib, Pandas, NumPy
*   **Generative AI:** Ollama (Python client)
*   **Voice Engine:** Pyttsx3

---

## 🚀 Getting Started

### Prerequisites

1.  **Python 3.8+** installed on your system.
2.  **Ollama** installed and running locally. You can download it from [ollama.com](https://ollama.com/).
3.  Pull the Llama 3.2 model for Ollama:
    ```bash
    ollama pull llama3.2
    ```

### Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/yourusername/Althera-Minor-Project.git
    cd Althera-Minor-Project
    ```

2.  Install the required Python packages:
    ```bash
    pip install -r requirements.txt
    ```

### Running the Dashboard

Launch the Althera dashboard using Streamlit:

```bash
streamlit run dashboard_app.py
```

The application will automatically open in your default web browser at `http://localhost:8501`.

---

## 📖 Usage Guide

The dashboard guides you through an 8-step workflow:

1.  **Connect Sensor:** Connect your serial sensor device or choose the "Simulated Data" option to proceed without hardware.
2.  **Collect Sensor Data:** The system establishes a 15-second baseline of your heart rate, SpO2, and motion levels.
3.  **Reaction Tests:** Complete the simple reaction, choice reaction, and finger tapping modules.
4.  **Memory Tests:** Complete the word recall, number recall, and Stroop tests.
5.  **Emotion Detection:** Take a photo using your webcam to record your dominant facial emotion.
6.  **ML Prediction:** The dashboard runs the aggregated data through the `cognitive_model.pkl` to produce your score.
7.  **AI Interpretation:** Send your results to Ollama for a personalized generative AI analysis.
8.  **View Results:** Review your comprehensive cognitive health report, complete with charts and AI insights.

---

## 📁 Project Structure

*   `dashboard_app.py`: The main Streamlit dashboard application.
*   `dashboard_bridge.py`: Core logic bridge containing the sensor reader, ML predictor wrapper, AI generator, emotion detector, and voice engine.
*   `dashboard_styles.py`: Custom CSS and Plotly templates for the dark-mode UI.
*   `cognitive_model.pkl`: Pre-trained Random Forest model for score prediction.
*   `requirements.txt`: Python dependency list.
*   *(Various standalone test files for individual modules like `memory_Test.py`, `emotion_detection.py`, etc.)*

---

## ⚠️ Disclaimer

**Althera is an experimental minor project and a prototype.** It is not a certified medical device and should not be used as a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of your physician or other qualified health provider with any questions you may have regarding a medical condition or cognitive health.

---

## 📄 License

This project is open-source and available under the [MIT License](LICENSE).
