# 👨‍💻 Face Recognition Attendance System

![Python](https://img.shields.io/badge/python-3.7%2B-blue)
![OpenCV](https://img.shields.io/badge/OpenCV-4.5%2B-orange)
![Tkinter](https://img.shields.io/badge/GUI-Tkinter-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

A secure attendance tracking system using facial recognition with network restrictions and multi-user authentication.

---

## ✨ Features

- 🔐 Multi-user login with bcrypt hashing
- 🌐 Network location verification (IP/DNS check)
- 📸 Face detection using Haar Cascade
- 🧠 Face recognition via LBPH algorithm
- 📅 Automatic attendance marking
- 📊 Daily attendance reports (CSV export)
- ⚙️ Configurable recognition settings

---

## 🛠️ Tech Stack

### 🖥️ Frontend
- Tkinter (Python GUI)
- PIL (Image processing)

### ⚙️ Backend
- Python 3.7+
- OpenCV (Face detection/recognition)
- Pandas (Data management)
- bcrypt (Password hashing)

---

## 📁 Project Structure

```
├── models/
│ └── haarcascade_frontalface_default.xml
├── TrainingImage/ 
├── StudentDetails/ 
├── Attendance/
├── TrainingImageLabel/
└── main.py
└── README.md
```
---

## ⚙️ Installation

### Prerequisites
- Python 3.7+
- OpenCV (`pip install opencv-python`)
- Additional dependencies:
  ```bash
  pip install pillow pandas bcrypt python-dotenv
  pip install pillow pandas bcrypt python-dotenv
  ```
  
