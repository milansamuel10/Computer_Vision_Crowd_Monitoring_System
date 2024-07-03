import os
import sys
import io
import subprocess
import datetime
import sqlite3
import bcrypt
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from detection import ObjectDetector
# rtsp://admin:admin1234@192.168.68.98:554/h264Preview_01_main

# Color Scheme
BACKGROUND_COLOR = "#D3E3FC"  # lavender
TEXT_COLOR = "#333333"  # dark Gray
BUTTON_COLOR = "#007BFF"  # blue
ERROR_COLOR = "#FFA500"  # orange

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Crowd Detection & Monitoring System - Main Window')
        self.setFixedSize(800,600)
        self.setStyleSheet(f"background-color: {BACKGROUND_COLOR};")
        self.video_path = None      #store the video path
        self.selected_fps = None    #store the selected fps
        self.selected_targetWidth = None     #store the target width
        self.selected_targetHeight = None   #store the target height
        self.generate_heatmap = False   #generate heatmap is set to off by default
        self.detection_threshold = None
        self.confidence_level = None
        self.results_filename = None

        # Add widgets and layout for the main window
        self.create_main_window()

    def create_main_window(self):
        # Add widgets and layout for the main window
        layout = QVBoxLayout()
            
        # Add QLineEdit for detection threshold
        threshold_label = QLabel('Enter Detection Threshold (integer above 0):', self)
        threshold_label.setStyleSheet(f"color: {TEXT_COLOR};")
        threshold_label.setFont(QFont('Arial', 15))
        layout.addWidget(threshold_label)
    
        self.threshold_entry = QLineEdit(self)
        self.threshold_entry.setPlaceholderText('Enter Threshold')
        self.threshold_entry.setStyleSheet(f"color: {TEXT_COLOR}; background-color: {BACKGROUND_COLOR}; border: 2px solid {BUTTON_COLOR}; border-radius: 5px;")
        self.threshold_entry.setFont(QFont('Arial', 12))
        self.threshold_entry.setMaximumWidth(150)
        layout.addWidget(self.threshold_entry)
    
        # Add the dropdown menu for selecting confidence levels
        confidence_label = QLabel('Select Confidence Level:', self)
        confidence_label.setStyleSheet(f"color: {TEXT_COLOR};")
        confidence_label.setFont(QFont('Arial', 15))
        layout.addWidget(confidence_label)

        self.confidence_combo = QComboBox(self)
        self.confidence_combo.setStyleSheet(f"color: {TEXT_COLOR}; background-color: {BACKGROUND_COLOR}; border: 2px solid {BUTTON_COLOR}; border-radius: 5px;")
        self.confidence_combo.setFont(QFont('Arial', 12))
        self.confidence_combo.setMaximumWidth(150)
        self.confidence_combo.activated.connect(self.set_confidence_level)
        
        # Add confidence levels in increments of 10 from 0 to 100
        for i in range(0, 101, 10):
            self.confidence_combo.addItem(f"{i}%")
            
        layout.addWidget(self.confidence_combo)
        
        # Add QLineEdit for results filename
        results_label = QLabel('Enter Results Filename:', self)
        results_label.setStyleSheet(f"color: {TEXT_COLOR};")
        results_label.setFont(QFont('Arial', 15))
        layout.addWidget(results_label)

        self.results_entry = QLineEdit(self)
        self.results_entry.setPlaceholderText('Enter Filename')
        self.results_entry.setStyleSheet(f"color: {TEXT_COLOR}; background-color: {BACKGROUND_COLOR}; border: 2px solid {BUTTON_COLOR}; border-radius: 5px;")
        self.results_entry.setFont(QFont('Arial', 12))
        self.results_entry.setMaximumWidth(300)
        layout.addWidget(self.results_entry)
        
        #dropdown menu for selecting the FPS
        fps_label = QLabel('Select FPS:', self)
        fps_label.setStyleSheet(f"color: {TEXT_COLOR};")
        fps_label.setFont(QFont('Arial', 15))

        self.fps_combo = QComboBox(self)
        self.fps_combo.setPlaceholderText('Set FPS')
        self.fps_combo.addItem('5 FPS')
        self.fps_combo.addItem('10 FPS')
        self.fps_combo.addItem('15 FPS')
        self.fps_combo.addItem('20 FPS')
        self.fps_combo.addItem('25 FPS')
        self.fps_combo.addItem('30 FPS')
        self.fps_combo.setStyleSheet(f"color: {TEXT_COLOR}; background-color: {BACKGROUND_COLOR}; border: 2px solid {BUTTON_COLOR}; border-radius: 5px;")
        self.fps_combo.setFont(QFont('Arial', 12))
        self.fps_combo.setMaximumWidth(100)
        self.fps_combo.activated[str].connect(self.set_fps)

        layout.addStretch(1)
        
        fps_layout = QVBoxLayout()
        fps_layout.addWidget(fps_label)
        fps_layout.addWidget(self.fps_combo)
        fps_layout.addStretch(1)
        layout.addLayout(fps_layout)
        
        layout.addStretch(1)
        
        # Dropdown menu for selecting the resolution
        resolution_label = QLabel('Select Resolution:', self)
        resolution_label.setStyleSheet(f"color: {TEXT_COLOR};")
        resolution_label.setFont(QFont('Arial', 15))

        self.resolution_combo = QComboBox(self)
        self.resolution_combo.setPlaceholderText('Set Resolution')
        self.resolution_combo.addItem('1920x1080')
        self.resolution_combo.addItem('1280x720')
        self.resolution_combo.addItem('800x600')
        self.resolution_combo.addItem('640x480')
        self.resolution_combo.setStyleSheet(f"color: {TEXT_COLOR}; background-color: {BACKGROUND_COLOR}; border: 2px solid {BUTTON_COLOR}; border-radius: 5px;")
        self.resolution_combo.setFont(QFont('Arial', 12))
        self.resolution_combo.setMaximumWidth(130)
        self.resolution_combo.activated[str].connect(self.set_resolution)
        layout.addWidget(resolution_label)
        layout.addWidget(self.resolution_combo)
        
        # Add checkbox for generating heatmap
        self.heatmap_checkbox = QCheckBox('Generate Heatmap (WARNING: Processing time may increase)', self)
        self.heatmap_checkbox.setStyleSheet(f"color: {TEXT_COLOR};")
        self.heatmap_checkbox.setFont(QFont('Arial', 15))
        self.heatmap_checkbox.stateChanged.connect(self.set_heatmap_generation)
        layout.addWidget(self.heatmap_checkbox)
        
        #Button to use default values
        defaultVal_btn = QPushButton('Use default values', self)
        defaultVal_btn.setStyleSheet(f"color: white; background-color: #FFA600; border: 2px solid {BUTTON_COLOR}; border-radius: 5px;")
        defaultVal_btn.setFont(QFont('Arial', 15))
        defaultVal_btn.setMaximumWidth(300)        
        defaultVal_btn.setCursor(QCursor(Qt.PointingHandCursor))
        defaultVal_btn.clicked.connect(self.setDefault_values)
        layout.addWidget(defaultVal_btn)
        
        # Add QLabel for displaying selected video filename or "No video selected" text
        self.video_label = QLabel('No video selected', self)
        self.video_label.setStyleSheet(f"color: {TEXT_COLOR};")
        self.video_label.setFont(QFont('Arial', 12))
        layout.addWidget(self.video_label)
        
        # Button for uploading video
        upload_btn = QPushButton('Upload Video', self)
        upload_btn.setStyleSheet(f"color: white; background-color: {BUTTON_COLOR}; border: 2px solid {BUTTON_COLOR}; border-radius: 5px;")
        upload_btn.setFont(QFont('Arial', 15))
        upload_btn.setMaximumWidth(300)        
        upload_btn.setCursor(QCursor(Qt.PointingHandCursor))
        upload_btn.clicked.connect(self.upload_video)
        layout.addWidget(upload_btn)
        
        # Button for entering RTSP path
        rtsp_input_btn = QPushButton('Enter RTSP Path', self)
        rtsp_input_btn.setStyleSheet(f"color: white; background-color: {BUTTON_COLOR}; border: 2px solid {BUTTON_COLOR}; border-radius: 5px;")
        rtsp_input_btn.setFont(QFont('Arial', 15))
        rtsp_input_btn.setMaximumWidth(300)
        rtsp_input_btn.setCursor(QCursor(Qt.PointingHandCursor))
        rtsp_input_btn.clicked.connect(self.get_rtsp_path)
        layout.addWidget(rtsp_input_btn)
        
        # Button to begin detection
        detect_btn = QPushButton('Begin Detection', self)
        detect_btn.setStyleSheet(f"color: white; background-color: #28A745; border: 2px solid {BUTTON_COLOR}; border-radius: 5px;")
        detect_btn.setFont(QFont('Arial', 15))
        detect_btn.setMaximumWidth(300)
        detect_btn.setCursor(QCursor(Qt.PointingHandCursor))
        detect_btn.clicked.connect(self.set_results_filename)
        detect_btn.clicked.connect(self.begin_detection)  # Connect to begin_detection method
        layout.addWidget(detect_btn)
        

        self.setLayout(layout)
        
    #function to set values as default
    def setDefault_values(self):
        self.selected_fps = 15    #store the selected fps
        self.selected_targetWidth = 1920     #store the target width
        self.selected_targetHeight = 1080   #store the target height
        self.generate_heatmap = False   #generate heatmap is set to off by default
        self.confidence_level = 0.5
        self.detection_threshold = 47
        #update the UI elements
        self.fps_combo.setCurrentIndex(2)
        self.confidence_combo.setCurrentIndex(int(self.confidence_level * 10))
        self.resolution_combo.setCurrentIndex(0)
        self.heatmap_checkbox.setChecked(self.generate_heatmap)
        self.threshold_entry.setText("47")

    def upload_video(self):
        # Open file dialog for selecting video file
        file_dialog = QFileDialog()
        video_path, _ = file_dialog.getOpenFileName(self, 'Select Video File', '', 'Video Files (*.mp4 *.avi)')
        if video_path:
            self.video_path = video_path
            print("Selected video:", video_path)
            self.video_label.setText("Selected video: " + os.path.basename(video_path))
            
    def get_rtsp_path(self):
        rtsp_path, ok = QInputDialog.getText(self, 'Enter RTSP Path', 'RTSP Path:')
        if ok:
         self.video_path = rtsp_path
         print("RTSP Path:", self.video_path)
         self.video_label.setText("RTSP Link: " + rtsp_path)
         
    #function to set the resolution after user selects an option from the dropdown
    def set_resolution(self, resolution_str):
        width, height = resolution_str.split('x')
        self.selected_targetWidth = int(width)
        self.selected_targetHeight = int(height)
        print("Selected Resolution:", self.selected_targetWidth, "x", self.selected_targetHeight)
        
     #function to set the fps after the user selects an option from the dropdown   
    def set_fps(self, fps_str):
        self.selected_fps = int(fps_str.split()[0])
        
     #function to enable/disable heatmap generation   
    def set_heatmap_generation(self, state):
        if state == Qt.Checked:
            self.generate_heatmap = True
        else:
            self.generate_heatmap = False
            
     #function to set the threshold       
    def set_threshold(self):
        threshold_text = self.threshold_entry.text()
        print("Threshold: " + threshold_text)
        try:
            threshold_value = int(threshold_text)
            if threshold_value <= 0:
                raise ValueError("Threshold must be above 0")
            self.detection_threshold = threshold_value
        except ValueError as e:
            QMessageBox.critical(self, 'Error', str(e))
            self.detection_threshold = None
            
    #function to set the confidence level
    def set_confidence_level(self):
        confidence_text = self.confidence_combo.currentText()  # Get the currently selected item from the combo box
        confidence_value = float(confidence_text.strip('%')) / 100  # Convert the selected item to a float percentage value
        self.confidence_level = confidence_value
            
    #function to set results filename
    def set_results_filename(self):
        filename_text = self.results_entry.text().strip()
        if filename_text:
            self.results_filename = str(filename_text)
        else:
            QMessageBox.critical(self, 'Error', 'Please enter a filename.')
            self.results_filename = None

    #function to clear the fields after processing is done 
    def clear_fields(self):
        # Clear video path
        self.video_path = None
        
        # Clear detection threshold field
        self.threshold_entry.clear()
        self.detection_threshold = None
        
        # Clear confidence level field
        self.confidence_entry.clear()
        self.confidence_level = None
        
        # Clear results filename field
        self.results_entry.clear()
        self.results_filename = None
        
        # Clear FPS selection
        self.selected_fps = None
        
        # Clear resolution selection
        self.selected_targetWidth = None
        self.selected_targetHeight = None
        
        # Clear heatmap generation checkbox
        self.generate_heatmap = False
        
    def begin_detection(self):      
        if self.video_path:
            self.set_threshold()
            if self.selected_fps is None:
                QMessageBox.critical(self, 'Error', 'Please select FPS.')    #need to fix because it doesnt reset after a video is done processing
                return
            
            if self.selected_targetWidth is None or self.selected_targetHeight is None:
                QMessageBox.critical(self, 'Error', 'Please select a resolution.')
                return
            
            print("Confidence is set to: ", self.confidence_level)
            
            # Get the current date
            current_date = datetime.datetime.now().strftime("%d-%m-%Y")
            video_basename = os.path.splitext(os.path.basename(self.video_path))[0]
            results_dir = "results"
            os.makedirs(results_dir, exist_ok=True)
            output_dir = os.path.join(results_dir, self.results_filename)
            os.makedirs(output_dir, exist_ok=True)
            filename_format = f'{output_dir}/{video_basename}_{current_date}'
            heatmap_filename = f"{filename_format}_heatmap.mp4",
            annotated_filename = f"{filename_format}_annotated.mp4",
            summary_filename = f"{filename_format}_summary.txt",
            total_objects_plot_filename = f"{filename_format}_objects_per_frame.png",
            threshold_plot_filename = f"{filename_format}_threshold_exceedance.png",
            movement_plot_filename = f"{filename_format}_movement_directions.png"
            
            # Provide the required parameters when creating an instance of ObjectDetector
            print(self.generate_heatmap)
            detector = ObjectDetector(
                model_path='models/head.pt',  
                detection_threshold=self.detection_threshold,      # Adjust the detection threshold as needed
                video_path=self.video_path,   # Use the selected video path
                results_filename=self.results_filename,
                target_width=self.selected_targetWidth,             # Adjust the target width for detection
                target_height=self.selected_targetHeight,            # Adjust the target height for detection
                fps=self.selected_fps,         # Adjust the frames per second (fps) for the video
                confidence=self.confidence_level,               # Adjust the confidence threshold for detection
                generate_heatmap=self.generate_heatmap,      #depends on if the user toggled generate heatmap or not
                heatmap_filename = f"{filename_format}_heatmap.mp4",
                annotated_filename = f"{filename_format}_annotated.mp4",
                summary_filename = f"{filename_format}_summary.txt",
                total_objects_plot_filename = f"{filename_format}_objects_per_frame.png",
                threshold_plot_filename = f"{filename_format}_threshold_exceedance.png",
                movement_plot_filename = f"{filename_format}_movement_directions.png"
            )
            #detector_output = 
            detector.detect_objects_in_video(self.video_path)  # start the detection process
            QMessageBox.information(self, 'Notification', 'Video processing is complete.')
            self.clear_fields()     #clear all fields after processing
        else:
            QMessageBox.critical(self, 'Error', 'Please upload a video first.')
            
class SignUpPage(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Crowd Detection & Monitoring System -- Sign Up')
        self.setFixedSize(800,600)
        self.setStyleSheet(f"background-color: {BACKGROUND_COLOR};")
        self.setMaximumSize(QSize(800,600))     #disable maximizing 

        self.font1 = QFont('Helvetica', 25)
        self.font2 = QFont('Arial', 17)
        self.font3 = QFont('Arial', 13)
        self.font4 = QFont('Arial', 13)

        self.conn = sqlite3.connect('registeredUsers.db')
        self.cursor = self.conn.cursor()
        self.login_page = None 

        self.cursor.execute('''
                            CREATE TABLE IF NOT EXISTS users (
                                username TEXT NOT NULL,
                                password TEXT NOT NULL)''')

        self.create_widgets()

    def create_widgets(self):
        image_path = '1.png'  # path for the picture on login screen
        main_layout = QHBoxLayout()  # Use QHBoxLayout for horizontal arrangement

        image_label = QLabel(self)
        pixmap = QPixmap(image_path)
        image_label.setPixmap(pixmap)
        image_label.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        main_layout.addWidget(image_label)

        signup_fields_layout = QVBoxLayout()  # Vertical layout for signup fields
        signup_fields_layout.setAlignment(Qt.AlignCenter)
        
        signup_label = QLabel('Sign up', self)
        signup_label.setStyleSheet(f"color: {TEXT_COLOR}; font-size: 25pt;")
        signup_fields_layout.addWidget(signup_label)
        signup_label.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        signup_label.setFixedHeight(signup_label.sizeHint().height())
        signup_fields_layout.addSpacing(50)

        # Sign-up Inputs and Button
        self.username_textbox_signup = QLineEdit(self)
        self.username_textbox_signup.setPlaceholderText('Username')
        self.username_textbox_signup.setStyleSheet(
            f"color: {TEXT_COLOR}; background-color: {BACKGROUND_COLOR}; border: 2px solid {BUTTON_COLOR}; border-radius: 5px; padding: 5px;")
        self.username_textbox_signup.setMaximumWidth(150)
        signup_fields_layout.addWidget(self.username_textbox_signup)
        signup_fields_layout.addSpacing(10)

        self.password_entry_signup = QLineEdit(self)
        self.password_entry_signup.setPlaceholderText('Password')
        self.password_entry_signup.setStyleSheet(
            f"color: {TEXT_COLOR}; background-color: {BACKGROUND_COLOR}; border: 2px solid {BUTTON_COLOR}; border-radius: 5px; padding: 5px;")
        self.password_entry_signup.setEchoMode(QLineEdit.Password)
        self.password_entry_signup.setMaximumWidth(150)
        signup_fields_layout.addWidget(self.password_entry_signup)
        signup_fields_layout.addSpacing(10)

        self.confirm_password_entry_signup = QLineEdit(self)
        self.confirm_password_entry_signup.setPlaceholderText('Confirm Password')
        self.confirm_password_entry_signup.setStyleSheet(
            f"color: {TEXT_COLOR}; background-color: {BACKGROUND_COLOR}; border: 2px solid {BUTTON_COLOR}; border-radius: 5px; padding: 5px;")
        self.confirm_password_entry_signup.setEchoMode(QLineEdit.Password)
        self.confirm_password_entry_signup.setMaximumWidth(150)
        signup_fields_layout.addWidget(self.confirm_password_entry_signup)
        signup_fields_layout.addSpacing(10)

        signup_btn = QPushButton('Sign up', self)
        signup_btn.setStyleSheet(
            f"color: {BACKGROUND_COLOR}; background-color: {BUTTON_COLOR}; border: 2px solid {BUTTON_COLOR}; border-radius: 5px; padding: 5px;")
        signup_btn.clicked.connect(self.register_user)
        signup_btn.setMaximumWidth(150)
        signup_fields_layout.addWidget(signup_btn)
        signup_fields_layout.addSpacing(75)
        
        # Add signup fields layout to main layout
        main_layout.addLayout(signup_fields_layout)
        
        login_linklabel = QLabel('Already have an account?', self)
        login_linklabel.setStyleSheet(f"color: {TEXT_COLOR}; font-size: 12pt;")
        signup_fields_layout.addWidget(login_linklabel)
        login_linklabel.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        login_linklabel.setFixedHeight(login_linklabel.sizeHint().height())
        signup_fields_layout.addSpacing(10)
        
        switch_toLogin_btn = QPushButton('Click here to login', self)
        switch_toLogin_btn.setStyleSheet(
            f"color: {BACKGROUND_COLOR}; background-color: {BUTTON_COLOR}; border: 2px solid {BUTTON_COLOR}; border-radius: 5px; padding: 5px;")
        switch_toLogin_btn.clicked.connect(self.switch_to_loginPage)
        switch_toLogin_btn.setMaximumWidth(250)
        signup_fields_layout.addWidget(switch_toLogin_btn)
        signup_fields_layout.addSpacing(10)

        self.setLayout(main_layout)

    def switch_to_main_window(self):
        self.clear_layout()
        self.hide()
        self.main_window = MainWindow()
        self.main_window.show()
        
    def switch_to_loginPage(self):
        self.clear_layout()
        self.hide()
        self.login_page = loginWindowPage(self.cursor)
        self.login_page.show()

    def register_user(self):
        username = self.username_textbox_signup.text()
        password = self.password_entry_signup.text()
        confirm_password = self.confirm_password_entry_signup.text()

        if username == '' or password == '' or confirm_password == '':
            QMessageBox.critical(self, 'Error', 'Please enter all fields.')
            return

        if password != confirm_password:
            QMessageBox.critical(self, 'Error', 'Passwords do not match.')
            return

        self.cursor.execute('SELECT username FROM users WHERE username=?', [username])
        if self.cursor.fetchone() is not None:
            QMessageBox.critical(self, 'Error', 'Username already exists.')
        else:
            encoded_password = password.encode('utf-8')
            hashed_password = bcrypt.hashpw(encoded_password, bcrypt.gensalt())
            print(hashed_password)
            self.cursor.execute('INSERT INTO users VALUES (?,?)', [username, hashed_password])
            self.conn.commit()
            QMessageBox.information(self, 'Success', 'Account has been created!')

    def clear_layout(self):
            for i in reversed(range(self.layout().count())):
                widget = self.layout().itemAt(i).widget()
                if widget is not None:
                    widget.setParent(None)
                    
#class for the loginWindow
class loginWindowPage(QWidget):
    def __init__(self, cursor):
        super().__init__()
        self.cursor = cursor
        self.setWindowTitle('Crowd Detection & Monitoring System -- Login In')
        self.setFixedSize(800,600)
        self.setStyleSheet(f"background-color: {BACKGROUND_COLOR};")
        self.setMaximumSize(QSize(800,600))     #disable maximizing 

        self.create_login_window()

    def create_login_window(self):
        image_path = '1.png'  # path for the picture on login screen
        main_layout = QHBoxLayout()  # Use QHBoxLayout for horizontal arrangement

        image_label = QLabel(self)
        pixmap = QPixmap(image_path)
        image_label.setPixmap(pixmap)
        image_label.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        main_layout.addWidget(image_label)

        login_fields_layout = QVBoxLayout()  # Vertical layout for login fields
        login_fields_layout.setAlignment(Qt.AlignCenter)
        
        login_label = QLabel('Log In', self)
        login_label.setStyleSheet(f"color: {TEXT_COLOR}; font-size: 25pt;")
        login_fields_layout.addWidget(login_label)
        login_label.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        login_label.setFixedHeight(login_label.sizeHint().height())
        login_fields_layout.addSpacing(50)

        # Login Inputs and Button
        self.username_textbox = QLineEdit(self)
        self.username_textbox.setPlaceholderText('Username')
        self.username_textbox.setStyleSheet(
            f"color: {TEXT_COLOR}; background-color: {BACKGROUND_COLOR}; border: 2px solid {BUTTON_COLOR}; border-radius: 5px; padding: 5px;")
        self.username_textbox.setMaximumWidth(150)
        login_fields_layout.addWidget(self.username_textbox)
        login_fields_layout.addSpacing(10)

        self.password_entry = QLineEdit(self)
        self.password_entry.setPlaceholderText('Password')
        self.password_entry.setStyleSheet(
            f"color: {TEXT_COLOR}; background-color: {BACKGROUND_COLOR}; border: 2px solid {BUTTON_COLOR}; border-radius: 5px; padding: 5px;")
        self.password_entry.setEchoMode(QLineEdit.Password)
        self.password_entry.setMaximumWidth(150)
        login_fields_layout.addWidget(self.password_entry)
        login_fields_layout.addSpacing(10)

        login_btn = QPushButton('Login', self)
        login_btn.setStyleSheet(
            f"color: {BACKGROUND_COLOR}; background-color: {BUTTON_COLOR}; border: 2px solid {BUTTON_COLOR}; border-radius: 5px; padding: 5px;")
        login_btn.clicked.connect(self.login)
        login_btn.setMaximumWidth(150)
        login_fields_layout.addWidget(login_btn)
        login_fields_layout.addSpacing(75)
        
        # Add login fields layout to main layout
        main_layout.addLayout(login_fields_layout)
        
        signup_linklabel = QLabel('Don\'t have an account?', self)
        signup_linklabel.setStyleSheet(f"color: {TEXT_COLOR}; font-size: 12pt;")
        login_fields_layout.addWidget(signup_linklabel)
        signup_linklabel.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        signup_linklabel.setFixedHeight(signup_linklabel.sizeHint().height())
        login_fields_layout.addSpacing(10)
        
        switch_toSignup_btn = QPushButton('Click here to sign up', self)
        switch_toSignup_btn.setStyleSheet(
            f"color: {BACKGROUND_COLOR}; background-color: {BUTTON_COLOR}; border: 2px solid {BUTTON_COLOR}; border-radius: 5px; padding: 5px;")
        switch_toSignup_btn.clicked.connect(self.switch_to_signupPage)
        switch_toSignup_btn.setMaximumWidth(250)
        login_fields_layout.addWidget(switch_toSignup_btn)
        login_fields_layout.addSpacing(10)

        self.setLayout(main_layout)
        
    def login(self):
        username = self.username_textbox.text()
        password = self.password_entry.text()
        if username != '' and password != '':
            self.cursor.execute('SELECT password FROM users WHERE username=?', [username])
            result = self.cursor.fetchone()
            if result:
                if bcrypt.checkpw(password.encode('utf-8'), result[0]):
                    QMessageBox.information(self, 'Success', 'Logged in successfully!')
                    self.switch_to_main_window()  # Switch to main window upon successful login
                else:
                    QMessageBox.critical(self, 'Error', 'Password is incorrect.')
            else:
                QMessageBox.critical(self, 'Error', 'Username is incorrect.')
        else:
            QMessageBox.critical(self, 'Error', 'Please input valid entries.')
            
    def switch_to_main_window(self):
        self.clear_layout()
        self.hide()
        self.main_window = MainWindow()
        self.main_window.show()
        
    def switch_to_signupPage(self):
        self.clear_layout()
        self.hide()
        self.signup_page = SignUpPage()
        self.signup_page.show()
        
    def clear_layout(self): 
        for i in reversed(range(self.layout().count())):
            widget = self.layout().itemAt(i).widget()
            if widget is not None:
                widget.setParent(None)

def main():
    app = QApplication(sys.argv)
    window = SignUpPage()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
