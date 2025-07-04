import tkinter as tk
from tkinter import ttk, messagebox as mess
import os
import cv2
import numpy as np
import pandas as pd
import datetime
import time
import threading
import json
import bcrypt
from dotenv import load_dotenv
from PIL import Image
import webbrowser
import re
import socket
import os
import ipaddress
from tkinter import simpledialog 

# Load environment variables
load_dotenv()

class Config:
    # Path configurations
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    HAARCASCADE_PATH = os.path.join(BASE_DIR, "models", "haarcascade_frontalface_default.xml")
    TRAINING_IMAGE_DIR = os.path.join(BASE_DIR, "TrainingImage")
    STUDENT_DETAILS_PATH = os.path.join(BASE_DIR, "StudentDetails", "StudentDetails.csv")
    ATTENDANCE_DIR = os.path.join(BASE_DIR, "Attendance")
    TRAINER_PATH = os.path.join(BASE_DIR, "TrainingImageLabel", "Trainner.yml")
    
    # Face detection settings
    FACE_DETECTION_SCALE = 1.2
    FACE_DETECTION_NEIGHBORS = 7
    FACE_DETECTION_MIN_SIZE = (120, 120)
    RECOGNITION_CONFIDENCE_THRESHOLD = 50
    REQUIRED_CONSECUTIVE_FRAMES = 3
    
    # Network restrictions
    ALLOWED_NETWORK = {
        'base_ip': '172.16.92.1',
        'subnet_mask': '255.255.252.0',  # Covers 172.16.92.0 - 172.16.95.255
        'allowed_dns': ['192.168.17.40', '172.16.10.10'],
        'fallback_pin': '123456'  # Change this to your secure PIN
    }
    '''ALLOWED_NETWORK = {
        'base_ip': '192.168.31.10',
        'subnet_mask': '255.255.255.0',  # Covers 172.16.92.0 - 172.16.95.255
        'allowed_dns': ['192.168.31.1'],
        'fallback_pin': '123456'  # Change this to your secure PIN
    }'''
def assure_path_exists(path):
    os.makedirs(path, exist_ok=True)

def validate_name(name):
    return all(c.isalpha() or c.isspace() for c in name)

def get_timestamp():
    ts = time.time()
    date = datetime.datetime.fromtimestamp(ts).strftime('%d-%m-%Y')
    time_str = datetime.datetime.fromtimestamp(ts).strftime('%I:%M:%S %p')
    return date, time_str

class AttendanceSystem:
    def __init__(self, window):
        self.window = window
        self.running = True
        self.camera = None
        self.recognizer = None
        self.face_cascade = cv2.CascadeClassifier(Config.HAARCASCADE_PATH)
        self.current_user = None
        self.camera_index = 0
        self.threshold = Config.RECOGNITION_CONFIDENCE_THRESHOLD
        
        # Initialize UI
        self.setup_ui()
        
        # Check configuration
        if not self.check_configuration():
            self.window.after(100, self.cleanup)
            return
        
        # Initialize models after login
        self.load_models()
        self.update_attendance_display()
        
        # Show login window and hide main window
        self.window.withdraw()  # Hide main window initially
        self.setup_login_system()
        
    def setup_ui(self):
        self.window.title("Face Recognition Attendance System")
        self.window.geometry("1280x720")
        self.window.configure(bg='#2d420a')
        
        # Title
        tk.Label(
            self.window,
            text="Face Recognition Based Attendance System",
            fg="white", bg="#2d420a",
            font=('Helvetica', 24, 'bold')
        ).pack(pady=20)
        
        # Main container
        main_frame = tk.Frame(self.window, bg="#2d420a")
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        # Date and Time
        date_frame = tk.Frame(main_frame, bg="#2d420a")
        date_frame.pack(fill=tk.X, pady=5)
        
        date_str = datetime.datetime.now().strftime('%d-%B-%Y')
        tk.Label(
            date_frame, text=date_str,
            fg="#ff61e5", bg="green",
            font=('Helvetica', 12, 'bold')
        ).pack(side=tk.LEFT, padx=10)
        
        self.clock_label = tk.Label(
            date_frame, fg="#ff61e5", bg="green",
            width=20, font=('Helvetica', 12, 'bold')
        )
        self.clock_label.pack(side=tk.LEFT)
        self.update_clock()
        
        # Content frames
        content_frame = tk.Frame(main_frame, bg="#2d420a")
        content_frame.pack(fill=tk.BOTH, expand=True)
        
        # Attendance Frame
        self.attendance_frame = tk.Frame(
            content_frame, bg="#c79cff", bd=2, relief=tk.RAISED
        )
        self.attendance_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.setup_attendance_frame()
        
        # Registration Frame
        self.registration_frame = tk.Frame(
            content_frame, bg="#c79cff", bd=2, relief=tk.RAISED
        )
        self.registration_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.setup_registration_frame()
        
        # Menu
        self.setup_menu()
    
    def setup_attendance_frame(self):
        """Setup the attendance frame UI"""
        # Title
        tk.Label(
            self.attendance_frame, text="For Already Registered", 
            fg="black", bg="#00fcca",
            font=('Helvetica', 14, 'bold')
        ).pack(fill=tk.X, pady=5)
        
        # Status label
        self.attendance_status = tk.Label(
            self.attendance_frame, text="", 
            bg="#c79cff", fg="black",
            font=('Helvetica', 12, 'bold')
        )
        self.attendance_status.pack(fill=tk.X, pady=5)
        
        # Camera Buttons
        btn_frame = tk.Frame(self.attendance_frame, bg="#c79cff")
        btn_frame.pack(fill=tk.X, pady=10)
        
        tk.Button(
            btn_frame, text="Test Camera",
            command=self.test_camera,
            fg="black", bg="#ff9900",
            width=15, height=1,
            font=('Helvetica', 10, 'bold')
        ).pack(side=tk.LEFT, padx=10)
        
        tk.Button(
            btn_frame, text="Take Attendance",
            command=self.track_images_threaded,
            fg="black", bg="#3ffc00",
            width=15, height=1,
            font=('Helvetica', 10, 'bold')
        ).pack(side=tk.LEFT, padx=10)
        
        # Treeview for attendance records
        tree_frame = tk.Frame(self.attendance_frame, bg="#c79cff")
        tree_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        self.tree = ttk.Treeview(
            tree_frame, height=13,
            columns=('name', 'date', 'time')
        )
        self.tree.column('#0', width=82, anchor='center')
        self.tree.column('name', width=130, anchor='center')
        self.tree.column('date', width=133, anchor='center')
        self.tree.column('time', width=133, anchor='center')
        self.tree.heading('#0', text='ID')
        self.tree.heading('name', text='NAME')
        self.tree.heading('date', text='DATE')
        self.tree.heading('time', text='TIME')
        self.tree.pack(fill=tk.BOTH, expand=True)
        
        # Scrollbars
        scroll_y = ttk.Scrollbar(
            tree_frame, orient='vertical',
            command=self.tree.yview
        )
        scroll_y.pack(side=tk.RIGHT, fill=tk.Y)
        
        scroll_x = ttk.Scrollbar(
            self.attendance_frame, orient='horizontal',
            command=self.tree.xview
        )
        scroll_x.pack(fill=tk.X, padx=10)
        
        self.tree.configure(
            yscrollcommand=scroll_y.set,
            xscrollcommand=scroll_x.set
        )
        
        # Quit Button
        tk.Button(
            self.attendance_frame, text="Quit",
            command=self.cleanup,
            fg="black", bg="#eb4600",
            width=35, height=1,
            font=('Helvetica', 12, 'bold')
        ).pack(pady=10)
    
    def setup_registration_frame(self):
        """Setup the registration frame UI"""
        # Title
        tk.Label(
            self.registration_frame, text="For New Registrations", 
            fg="black", bg="#00fcca",
            font=('Helvetica', 14, 'bold')
        ).pack(fill=tk.X, pady=5)
        
        # ID Entry
        id_frame = tk.Frame(self.registration_frame, bg="#c79cff")
        id_frame.pack(fill=tk.X, padx=10, pady=5)
        
        tk.Label(
            id_frame, text="Enter ID:", width=15,
            fg="black", bg="#c79cff",
            font=('Helvetica', 12)
        ).pack(side=tk.LEFT)
        
        self.id_entry = tk.Entry(
            id_frame, width=25,
            fg="black", font=('Helvetica', 12)
        )
        self.id_entry.pack(side=tk.LEFT)
        
        # Name Entry
        name_frame = tk.Frame(self.registration_frame, bg="#c79cff")
        name_frame.pack(fill=tk.X, padx=10, pady=5)
        
        tk.Label(
            name_frame, text="Enter Name:", width=15,
            fg="black", bg="#c79cff",
            font=('Helvetica', 12)
        ).pack(side=tk.LEFT)
        
        self.name_entry = tk.Entry(
            name_frame, width=25,
            fg="black", font=('Helvetica', 12)
        )
        self.name_entry.pack(side=tk.LEFT)
        
        # Status messages
        self.registration_status1 = tk.Label(
            self.registration_frame, 
            text="1) Take Images  >>>  2) Save Profile",
            bg="#c79cff", fg="black",
            font=('Helvetica', 12, 'bold')
        )
        self.registration_status1.pack(fill=tk.X, pady=10)
        
        # Buttons
        tk.Button(
            self.registration_frame, text="Take Images", 
            command=self.take_images_threaded,
            fg="white", bg="#6d00fc",
            width=25, height=1,
            font=('Helvetica', 12, 'bold')
        ).pack(pady=5)
        
        tk.Button(
            self.registration_frame, text="Save Profile", 
            command=self.train_images_threaded,
            fg="white", bg="#6d00fc",
            width=25, height=1,
            font=('Helvetica', 12, 'bold')
        ).pack(pady=5)
        
        # Registration count
        self.registration_status2 = tk.Label(
            self.registration_frame, text="",
            bg="#c79cff", fg="black",
            font=('Helvetica', 12, 'bold')
        )
        self.registration_status2.pack(fill=tk.X, pady=10)
    
    def setup_menu(self):
        menubar = tk.Menu(self.window)
        
        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="Settings", command=self.show_settings)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.cleanup)
        menubar.add_cascade(label="File", menu=file_menu)
        
        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        help_menu.add_command(label="Documentation", command=self.show_documentation)
        help_menu.add_command(label="About", command=self.show_about)
        help_menu.add_command(label="Contact", command=self.contact)
        menubar.add_cascade(label="Help", menu=help_menu)
        
        self.window.config(menu=menubar)
    
    def initialize_directories(self):
        directories = [
            os.path.dirname(Config.HAARCASCADE_PATH),
            Config.TRAINING_IMAGE_DIR,
            os.path.dirname(Config.STUDENT_DETAILS_PATH),
            Config.ATTENDANCE_DIR,
            os.path.dirname(Config.TRAINER_PATH)
        ]
        for directory in directories:
            assure_path_exists(directory)
    def is_in_allowed_network(self, ip):
        """Check if IP is in the allowed subnet"""
        network = ipaddress.IPv4Network(f"{Config.ALLOWED_NETWORK['base_ip']}/{Config.ALLOWED_NETWORK['subnet_mask']}", 
                                    strict=False)
        return ipaddress.IPv4Address(ip) in network

    def verify_network(self):
        """Perform complete network verification"""
        try:
            # Get local IP
            local_ip = socket.gethostbyname(socket.gethostname())
            print(f"Detected local IP: {local_ip}")  # Debugging
            
            # 1. Check subnet
            if not self.is_in_allowed_network(local_ip):
                print(f"IP {local_ip} not in allowed network")
                return False
                
            # 2. Check DNS (Windows/Linux compatible)
            dns_ok = False
            try:
                if os.name == 'nt':  # Windows
                    dns_output = os.popen('ipconfig /all').read()
                else:  # Linux/Mac
                    dns_output = os.popen('nmcli dev show').read()
                
                dns_ok = all(dns in dns_output for dns in Config.ALLOWED_NETWORK['allowed_dns'])
            except:
                dns_ok = False
                
            if not dns_ok:
                print("DNS verification failed")
                return False
                
            return True
            
        except Exception as e:
            print(f"Network verification failed: {str(e)}")
            return False
        
    def fallback_verification(self):
        """Manual verification when automatic checks fail"""
        pin = simpledialog.askstring("Location Verification",
                                    "Network verification failed!\n"
                                    "Enter admin PIN to continue:",
                                    show='*')
        return pin == "YOUR_SECRET_PIN"
        
    def load_models(self):
        try:
            if not os.path.exists(Config.HAARCASCADE_PATH):
                raise Exception("Haar cascade file not found")
                
            self.face_cascade = cv2.CascadeClassifier(Config.HAARCASCADE_PATH)
            if self.face_cascade.empty():
                raise Exception("Failed to load face detection model")
            
            if os.path.exists(Config.TRAINER_PATH):
                self.recognizer = cv2.face.LBPHFaceRecognizer_create()
                self.recognizer.read(Config.TRAINER_PATH)
        except Exception as e:
            mess.showerror("Error", f"Failed to load models: {str(e)}")
    
    def update_clock(self):
        if hasattr(self, 'running') and self.running:
            current_time = time.strftime('%I:%M:%S %p')
            self.clock_label.config(text=current_time)
            self.window.after(1000, self.update_clock)
    
    def take_images_threaded(self):
        threading.Thread(target=self.take_images, daemon=True).start()
    
    def train_images_threaded(self):
        threading.Thread(target=self.train_images, daemon=True).start()
    
    def track_images_threaded(self):
        threading.Thread(target=self.track_images, daemon=True).start()
    
    def get_camera(self, index=None):
        """Helper function to get a working camera"""
        if index is None:
            index = self.camera_index
        for i in [index, *[x for x in [0, 1, 2] if x != index]]:
            cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
            if cap.isOpened():
                return cap
        return None
    
    def test_camera(self):
        """Test camera with improved face detection parameters"""
        try:
            cap = self.get_camera()
            if not cap:
                mess.showerror("Error", "Could not open camera")
                return
            
            # Set higher resolution for better detection
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            
            # Create a window with better properties
            cv2.namedWindow("Camera Test", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Camera Test", 800, 600)
            
            # Improved face detection parameters
            scale_factor = 1.05  # Reduced from 1.2 for better detection
            min_neighbors = 6    # Increased from 5 for better accuracy
            min_size = (80, 80)  # Smaller minimum size for closer faces
            
            start_time = time.time()
            while time.time() - start_time < 15:  # Run for 15 seconds
                ret, frame = cap.read()
                if not ret:
                    continue
                
                # Convert to grayscale and improve contrast
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gray = cv2.equalizeHist(gray)
                
                # Detect faces with optimized parameters
                faces = self.face_cascade.detectMultiScale(
                    gray,
                    scaleFactor=scale_factor,
                    minNeighbors=min_neighbors,
                    minSize=min_size,
                    flags=cv2.CASCADE_SCALE_IMAGE
                )
                
                # Additional validation for detected faces
                valid_faces = []
                for (x, y, w, h) in faces:
                    # Check aspect ratio (typical faces are roughly 1:1 to 1:1.5)
                    aspect_ratio = w / float(h)
                    if 0.7 < aspect_ratio < 1.5:
                        # Check face area isn't too small or too large
                        face_area = w * h
                        frame_area = frame.shape[0] * frame.shape[1]
                        if 0.01 < (face_area / frame_area) < 0.3:
                            valid_faces.append((x, y, w, h))
                
                # Draw rectangles around valid faces only
                for (x, y, w, h) in valid_faces:
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    cv2.putText(frame, "Face", (x, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                # Display count and instructions
                cv2.putText(frame, f"Faces Detected: {len(valid_faces)}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, "Press Q to quit", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(frame, "Ensure good lighting and clear view", (10, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                
                cv2.imshow("Camera Test", frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            cap.release()
            cv2.destroyAllWindows()
        
        except Exception as e:
            mess.showerror("Error", f"Camera test failed: {str(e)}")
            if 'cap' in locals():
                cap.release()
            cv2.destroyAllWindows()

    def take_images(self):
        try:
            Id = self.id_entry.get().strip()
            name = self.name_entry.get().strip()
            
            # Validate inputs
            if not Id or not name:
                mess.showerror("Error", "Both ID and Name are required!")
                return
            
            if not Id.isdigit():
                mess.showerror("Error", "ID must be a number!")
                return
            
            if not validate_name(name):
                mess.showerror("Error", "Name should contain only letters and spaces")
                return
            
            # Check if ID already exists
            if os.path.exists(Config.STUDENT_DETAILS_PATH):
                df = pd.read_csv(Config.STUDENT_DETAILS_PATH)
                if int(Id) in df['ID'].values:
                    mess.showerror("Error", f"ID {Id} already exists!")
                    return
            
            # Create person-specific folder
            person_folder = os.path.join(Config.TRAINING_IMAGE_DIR, f"{name}_{Id}")
            assure_path_exists(person_folder)
            
            # Initialize camera
            self.camera = self.get_camera()
            if not self.camera or not self.camera.isOpened():
                mess.showerror("Error", "Could not open camera. Please check your camera connection.")
                return
            
            # Set camera properties
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            
            sample_num = 0
            face_detected = False
            self.registration_status1.configure(text="Align your face with the camera...")
            self.window.update()
            
            while sample_num < 30:  # Capture 30 samples
                ret, frame = self.camera.read()
                if not ret:
                    continue
                
                # Convert to grayscale and enhance contrast
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gray = cv2.equalizeHist(gray)
                
                # Detect faces with improved parameters
                faces = self.face_cascade.detectMultiScale(
                    gray,
                    scaleFactor=1.3,
                    minNeighbors=5,
                    minSize=(100, 100),
                    flags=cv2.CASCADE_SCALE_IMAGE
                )
                
                if len(faces) > 0:
                    # Get the largest face
                    faces = sorted(faces, key=lambda x: x[2]*x[3], reverse=True)
                    x, y, w, h = faces[0]
                    
                    # Draw rectangle and display sample count
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                    sample_num += 1
                    face_detected = True
                    
                    # Save face image in person's folder
                    img_name = f"{sample_num}.jpg"
                    img_path = os.path.join(person_folder, img_name)
                    cv2.imwrite(img_path, gray[y:y+h, x:x+w])
                    
                    # Display status
                    cv2.putText(frame, f"Samples: {sample_num}/30", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                else:
                    cv2.putText(frame, "No face detected - Move closer", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
                # Show the frame
                cv2.imshow("Capturing Images - Press Q to quit", frame)
                
                # Break if 'q' is pressed
                if cv2.waitKey(200) & 0xFF == ord('q'):
                    break
            
            # Cleanup
            if self.camera:
                self.camera.release()
            cv2.destroyAllWindows()
            
            # Verify results
            if not face_detected:
                mess.showerror("Error", "No face detected during capture!")
                return
            
            if sample_num < 10:  # Minimum samples required
                mess.showerror("Error", f"Only {sample_num} samples captured. Need at least 10.")
                return
            
            # Save student details
            self.save_student_details(Id, name)
            mess.showinfo("Success", f"{sample_num} images captured for {name} (ID: {Id})")
            
        except Exception as e:
            mess.showerror("Error", f"Failed to capture images: {str(e)}")
            if hasattr(self, 'camera') and self.camera:
                self.camera.release()
            cv2.destroyAllWindows()
    
    def save_student_details(self, Id, name):
        """Save student details to CSV file"""
        data = {'ID': [int(Id)], 'Name': [name]}
        new_df = pd.DataFrame(data)
        
        if os.path.exists(Config.STUDENT_DETAILS_PATH):
            existing_df = pd.read_csv(Config.STUDENT_DETAILS_PATH)
            df = pd.concat([existing_df, new_df], ignore_index=True)
        else:
            df = new_df
        
        df.to_csv(Config.STUDENT_DETAILS_PATH, index=False)
        self.registration_status2.configure(text=f"Total Registrations: {len(df)}")
    
    def train_images(self):
        """Train the face recognition model"""
        try:
            if not os.path.exists(Config.TRAINING_IMAGE_DIR) or \
               not os.listdir(Config.TRAINING_IMAGE_DIR):
                raise Exception("No training images found. Please take images first.")
            
            # Add progress feedback
            self.registration_status1.configure(text="Training in progress...")
            self.window.update()
            
            faces, Ids = self.get_images_and_labels(Config.TRAINING_IMAGE_DIR)
            
            if len(faces) == 0:
                raise Exception("No valid faces found in training images. Please retake images.")
            
            # Create recognizer and train
            self.recognizer = cv2.face.LBPHFaceRecognizer_create()
            self.recognizer.train(faces, np.array(Ids))
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(Config.TRAINER_PATH), exist_ok=True)
            
            # Save the model
            self.recognizer.save(Config.TRAINER_PATH)
            
            # Update UI
            self.registration_status1.configure(text="Profile Saved Successfully")
            self.registration_status2.configure(text=f"Total Registrations: {len(set(Ids))}")
            
            mess.showinfo("Success", f"Model trained successfully with {len(faces)} samples!")
            
        except Exception as e:
            mess.showerror("Training Error", f"Failed to train model: {str(e)}\n\n"
                          "Possible solutions:\n"
                          "1. Make sure you've taken enough images (30+)\n"
                          "2. Ensure faces are clearly visible in images\n"
                          "3. Check directory permissions")
    
    def get_images_and_labels(self, path):
        """Load training images and corresponding labels from person-specific folders"""
        faces = []
        Ids = []
        
        # Get all person folders
        person_folders = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
        
        for person_folder in person_folders:
            try:
                # Extract ID from folder name (format: Name_ID)
                Id = int(person_folder.split('_')[-1])
                
                # Get all images in this person's folder
                image_files = [f for f in os.listdir(os.path.join(path, person_folder)) 
                            if f.endswith('.jpg')]
                
                for image_file in image_files:
                    image_path = os.path.join(path, person_folder, image_file)
                    pil_image = Image.open(image_path).convert('L')
                    image_np = np.array(pil_image, 'uint8')
                    faces.append(image_np)
                    Ids.append(Id)
            except Exception as e:
                print(f"Error processing {person_folder}: {str(e)}")
                continue
        
        return faces, Ids
    
    def show_settings(self):
        """Display system settings window"""
        settings_window = tk.Toplevel(self.window)
        settings_window.title("System Settings")
        settings_window.geometry("400x300")
        
        # Add camera index selection
        tk.Label(settings_window, text="Camera Index:").pack(pady=5)
        self.camera_index_var = tk.IntVar(value=self.camera_index)
        tk.Spinbox(settings_window, from_=0, to=2, textvariable=self.camera_index_var).pack()
        
        # Add recognition threshold
        tk.Label(settings_window, text="Recognition Threshold:").pack(pady=5)
        self.threshold_var = tk.IntVar(value=self.threshold)
        tk.Scale(settings_window, from_=0, to=100, variable=self.threshold_var, orient='horizontal').pack()
        
        # Save button
        tk.Button(settings_window, text="Save Settings", 
                 command=self.save_settings).pack(pady=10)
        network_frame = tk.Frame(settings_window, bd=2, relief=tk.GROOVE)
        network_frame.pack(pady=10)
        
        tk.Label(network_frame, 
                text=f"Allowed Network:\n"
                    f"IP: {Config.ALLOWED_NETWORK['base_ip']}\n"
                    f"Subnet: {Config.ALLOWED_NETWORK['subnet_mask']}\n"
                    f"DNS: {', '.join(Config.ALLOWED_NETWORK['allowed_dns'])}",
                justify=tk.LEFT).pack(padx=10, pady=5)
    
    def save_settings(self):
        """Save system settings"""
        self.camera_index = self.camera_index_var.get()
        self.threshold = self.threshold_var.get()
        Config.RECOGNITION_CONFIDENCE_THRESHOLD = self.threshold
        mess.showinfo("Success", "Settings saved successfully")
    
    def check_configuration(self):
        """Verify all required files and settings"""
        errors = []
        
        if not os.path.exists(Config.HAARCASCADE_PATH):
            errors.append("Face detection model file missing")
        
        if errors:
            mess.showerror("Configuration Error", 
                          "The following issues were found:\n\n" + 
                          "\n".join(f"• {error}" for error in errors))
            return False
        return True
    
    def on_login_window_close(self):
        """Handle login window closing attempt"""
        if mess.askyesno("Exit", "Do you want to exit the application?"):
            self.cleanup()
        else:
            # Keep the login window open
            return
    
    def setup_login_system(self):
        """Create login system for multiple users"""
        self.login_window = tk.Toplevel(self.window)
        self.login_window.title("Login")
        self.login_window.geometry("300x200")
        
        # Make the login window modal
        self.login_window.grab_set()  # Prevents interaction with other windows
        self.login_window.protocol("WM_DELETE_WINDOW", self.on_login_window_close)
        
        tk.Label(self.login_window, text="Username:").pack(pady=5)
        self.username_entry = tk.Entry(self.login_window)
        self.username_entry.pack(pady=5)
        
        tk.Label(self.login_window, text="Password:").pack(pady=5)
        self.password_entry = tk.Entry(self.login_window, show="*")
        self.password_entry.pack(pady=5)
        
        tk.Button(self.login_window, text="Login", 
                command=self.authenticate_user).pack(pady=10)
        tk.Button(self.login_window, text="Register New User",
                command=self.register_user).pack(pady=5)
    
    def authenticate_user(self):
        """Authenticate existing user"""
        username = self.username_entry.get()
        password = self.password_entry.get()
        
        if not os.path.exists("users.json"):
            mess.showerror("Error", "No users registered yet")
            return
        
        with open("users.json", "r") as f:
            users = json.load(f)
        
        if username in users and bcrypt.checkpw(password.encode(), users[username].encode()):
            self.current_user = username
            self.login_window.destroy()
            self.window.deiconify()  # Show the main window
            mess.showinfo("Success", f"Welcome {username}!")
        else:
            mess.showerror("Error", "Invalid username or password")
    
    def register_user(self):
        """Register new user"""
        username = self.username_entry.get()
        password = self.password_entry.get()
        
        if not username or not password:
            mess.showerror("Error", "Username and password required")
            return
        
        users = {}
        if os.path.exists("users.json"):
            with open("users.json", "r") as f:
                users = json.load(f)
        
        if username in users:
            mess.showerror("Error", "Username already exists")
            return
        
        users[username] = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
        
        with open("users.json", "w") as f:
            json.dump(users, f)
        
        mess.showinfo("Success", "User registered successfully")
    
    def track_images(self):
        """Track faces and mark attendance with extended duration"""
        try:
            # Network verification FIRST before anything else
            if not self.verify_network():
                mess.showerror("Access Denied", 
                            "Attendance can only be marked from authorized devices!\n\n"
                            "Required network: 172.16.92.0/22\n"
                            "Contact admin if this is incorrect.")
                return
                
            if not self.recognizer or not os.path.exists(Config.TRAINER_PATH):
                raise Exception("No trained model found. Please train first.")
            
            if not os.path.exists(Config.STUDENT_DETAILS_PATH):
                raise Exception("Student details not found")
            
            df = pd.read_csv(Config.STUDENT_DETAILS_PATH)
            
            # Initialize camera
            cam = self.get_camera()
            if not cam or not cam.isOpened():
                raise Exception("Could not open camera")
            
            cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            
            # Create named window for attendance
            cv2.namedWindow('Attendance System', cv2.WINDOW_NORMAL)
            
            # Get screen dimensions
            screen_width = self.window.winfo_screenwidth()
            screen_height = self.window.winfo_screenheight()
            
            # Set window size (adjust as needed)
            window_width = 800
            window_height = 600
            
            # Calculate position to center the window
            position_x = int((screen_width - window_width) / 2)
            position_y = int((screen_height - window_height) / 2)
            
            # Set window position and size
            cv2.resizeWindow('Attendance System', window_width, window_height)
            cv2.moveWindow('Attendance System', position_x, position_y)
            
            # Variables for attendance marking
            attendance_marked = False
            marked_id = None
            marked_name = None
            face_detected_time = None
            min_detection_duration = 3  # Minimum 3 seconds to show the face
            
            while True:  # Run until face detected or user quits
                ret, frame = cam.read()
                if not ret:
                    continue
                
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gray = cv2.equalizeHist(gray)  # Improve contrast
                
                faces = self.face_cascade.detectMultiScale(
                    gray,
                    scaleFactor=1.2,
                    minNeighbors=7,
                    minSize=(120, 120),
                    flags=cv2.CASCADE_SCALE_IMAGE
                )
                
                current_time = time.time()
                
                if len(faces) > 0:
                    # Get the largest face
                    faces = sorted(faces, key=lambda x: x[2]*x[3], reverse=True)
                    x, y, w, h = faces[0]
                    
                    # Start timer when face is first detected
                    if face_detected_time is None:
                        face_detected_time = current_time
                    
                    Id, confidence = self.recognizer.predict(gray[y:y+h, x:x+w])
                    
                    if confidence < Config.RECOGNITION_CONFIDENCE_THRESHOLD:  # Good recognition
                        student = df[df['ID'] == Id]
                        if not student.empty:
                            name = student['Name'].values[0]
                            
                            # Draw green rectangle for recognized face
                            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                            cv2.putText(frame, name, (x, y-10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                            
                            # Calculate how long the face has been detected
                            detection_duration = current_time - face_detected_time
                            
                            # Show remaining time (if less than min_detection_duration)
                            if detection_duration < min_detection_duration:
                                remaining = min_detection_duration - detection_duration
                                cv2.putText(frame, f"Processing: {remaining:.1f}s", 
                                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 
                                        0.7, (0, 255, 255), 2)
                            
                            # Mark attendance only after minimum duration
                            if detection_duration >= min_detection_duration and not attendance_marked:
                                date, time_str = get_timestamp()
                                self.mark_attendance(Id, name, date, time_str)
                                attendance_marked = True
                                marked_id = Id
                                marked_name = name
                    else:
                        # Draw red rectangle for unknown face
                        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
                        cv2.putText(frame, "Unknown", (x, y-10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                        face_detected_time = None  # Reset timer for unknown faces
                else:
                    face_detected_time = None  # Reset timer when no face detected
                
                # Show the frame with larger text
                cv2.putText(frame, "Attendance System - Press Q to quit", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                cv2.imshow('Attendance System', frame)
                
                # Break if attendance is marked or 'q' is pressed
                if attendance_marked or (cv2.waitKey(1) & 0xFF == ord('q')):
                    break
            
            # Cleanup
            cam.release()
            cv2.destroyAllWindows()
            
            # Update UI with attendance result
            self.update_attendance_display()
            
            # Show result in main UI
            if attendance_marked:
                self.attendance_status.config(
                    text=f"{marked_name} (ID: {marked_id}) marked present!",
                    fg="green"
                )
            else:
                self.attendance_status.config(
                    text="Attendance not marked",
                    fg="red"
                )
            
        except Exception as e:
            mess.showerror("Error", f"Attendance tracking failed: {str(e)}")
            if 'cam' in locals():
                cam.release()
            cv2.destroyAllWindows()
    
    def mark_attendance(self, Id, name, date, time_str):
        if not self.verify_network():
            mess.showerror("Access Denied", 
                        "Attendance can only be marked from authorized devices!\n\n"
                        "Required network: 172.16.92.0/22\n"
                        "Contact admin if this is incorrect.")
            return
        attendance_file = os.path.join(Config.ATTENDANCE_DIR, f"Attendance_{date}.csv")
        
        if os.path.exists(attendance_file):
            existing_df = pd.read_csv(attendance_file)
            if Id in existing_df["ID"].values:
                return
        
        new_entry = pd.DataFrame({
            'ID': [Id],
            'Name': [name],
            'Date': [date],
            'Time': [time_str]
        })
        
        if os.path.exists(attendance_file):
            existing_df = pd.read_csv(attendance_file)
            updated_df = pd.concat([existing_df, new_entry], ignore_index=True)
        else:
            updated_df = new_entry
        
        updated_df.to_csv(attendance_file, index=False)
    
    def update_attendance_display(self):
        date = datetime.datetime.now().strftime('%d-%m-%Y')
        attendance_file = os.path.join(Config.ATTENDANCE_DIR, f"Attendance_{date}.csv")
        
        for item in self.tree.get_children():
            self.tree.delete(item)
        
        if os.path.exists(attendance_file):
            df = pd.read_csv(attendance_file)
            for _, row in df.iterrows():
                self.tree.insert(
                    "",
                    'end',
                    text=row['ID'],
                    values=(row['Name'], row['Date'], row['Time']),
                )
    
    def show_documentation(self):
        """Show documentation in a web browser"""
        docs = """
        <html>
        <body>
        <h1>Face Recognition Attendance System Documentation</h1>
        
        <h2>1. Registration Process</h2>
        <ol>
            <li>Enter a numeric ID for the student</li>
            <li>Enter the student's name (letters and spaces only)</li>
            <li>Click "Take Images" to capture 30 face samples</li>
            <li>Click "Save Profile" to train the recognition model</li>
        </ol>
        
        <h2>2. Taking Attendance</h2>
        <ol>
            <li>Click "Take Attendance" to start face recognition</li>
            <li>The system will automatically mark attendance for recognized faces</li>
            <li>Results will be displayed in the main window</li>
        </ol>
        </body>
        </html>
        """
        
        # Create temporary HTML file
        temp_file = os.path.join(Config.BASE_DIR, "documentation.html")
        with open(temp_file, "w") as f:
            f.write(docs)
        
        # Open in default browser
        webbrowser.open(f"file://{temp_file}")
    
    def show_about(self):
        """Show about information"""
        about_text = """
        Face Recognition Based Attendance System
        
        Version: 2.1
        Developed by: Your Name
        
        Features:
        - Accurate face detection and recognition
        - Automatic attendance marking
        - Real-time status updates
        """
        mess.showinfo("About", about_text)
    
    def contact(self):
        """Show contact information"""
        mess.showinfo("Contact", 
                     "For support, please contact:\n\n"
                     "Email: sankalpagarwal@gmail.com\n"
                     "Phone: +91 7060795512\n\n"
                     "GitHub: github.com/sankalp/attendance-system")
    
    def cleanup(self):
        """Clean up resources before exiting"""
        self.running = False
        if hasattr(self, 'camera') and self.camera:
            self.camera.release()
        if hasattr(self, 'login_window') and self.login_window:
            self.login_window.destroy()
        cv2.destroyAllWindows()
        self.window.destroy()

if __name__ == "__main__":
    # Create required directories
    assure_path_exists("models")
    assure_path_exists("TrainingImage")
    assure_path_exists("StudentDetails")
    assure_path_exists("Attendance")
    assure_path_exists("TrainingImageLabel")
    
    root = tk.Tk()
    app = AttendanceSystem(root)
    root.protocol("WM_DELETE_WINDOW", app.cleanup)
    root.mainloop()