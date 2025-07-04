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
## 🔐 Authentication

**Secure Login System:**

- User credentials stored using **bcrypt** hashing  
- **JWT** (JSON Web Token) authentication  
- Session management  

### Authentication Routes

| Route       | Method | Description         |
|-------------|--------|---------------------|
| `/login`    | POST   | User login          |
| `/register` | POST   | New user registration |

---

## 👥 User Management

### Admin Functions

- Add/remove users  
- Reset passwords  
- View access logs  

### User Types

- **Admin** – Full access  
- **Staff** – Attendance marking only  
- **Viewer** – Reports only  

---

## 📸 Face Recognition Flow

### 1. Face Detection

- Uses **Haar Cascade** classifier  
- Minimum **120x120** pixel detection  
- Capture **30 samples per student**  

### 2. Model Training

- Uses **LBPH** algorithm  
- Configurable **confidence threshold**

### 3. Attendance Marking

- 3-second **face verification** period  
- **Network location** validation  
- Attendance **exported as CSV**

---

## 🚀 Usage

### Launch the Application

```bash
python main.py
```

## 📬 Contact

For any inquiries or feedback, feel free to reach out:

- **Email:** [sankalpagarwal8057@example.com](mailto:sankalpagarwal8057@example.com)  
- **LinkedIn:** [https://www.linkedin.com/in/sankalp-agarwal-2b61ab253/]((https://www.linkedin.com/in/sankalp-agarwal-2b61ab253/))

---

## 🙏 Acknowledgements

- [OpenCV](https://opencv.org/)
- [Python](https://www.python.org/)
- [Tkinter](https://docs.python.org/3/library/tkinter.html)
