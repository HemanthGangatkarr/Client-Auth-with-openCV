import tkinter as tk
from tkinter import messagebox
import cv2
import face_recognition
import numpy as np
import csv
from datetime import datetime
import threading

# Initialize Tkinter
root = tk.Tk()
root.title("Face Recognition Attendance System - Sapthagiri College of Engineering")

# Global variables
video_capture = cv2.VideoCapture(0)
is_running = False

# Function to perform face recognition and save recognized images
def run_face_recognition():
    global is_running

    # Load known face encodings and names
    jobs_image = face_recognition.load_image_file("images/Steve_Jobs.jpg")
    jobs_encoding = face_recognition.face_encodings(jobs_image)[0]

    abhi_image = face_recognition.load_image_file("images/abhi.jpg")
    abhi_encoding = face_recognition.face_encodings(abhi_image)[0]

    hemanth_image = face_recognition.load_image_file("images/hemy.jpg")
    hemanth_encoding = face_recognition.face_encodings(hemanth_image)[0]

    ratan_tata_image = face_recognition.load_image_file("images/ratan tata.jpg")
    ratan_tata_encoding = face_recognition.face_encodings(ratan_tata_image)[0]

    sadmona_image = face_recognition.load_image_file("images/mona lisa.jpg")
    sadmona_encoding = face_recognition.face_encodings(sadmona_image)[0]

    tesla_image = face_recognition.load_image_file("images/tesla.jpeg")
    tesla_encoding = face_recognition.face_encodings(tesla_image)[0]


    known_face_encoding = [
        jobs_encoding,
        ratan_tata_encoding,
        sadmona_encoding,
        tesla_encoding,
        abhi_encoding,
        hemanth_encoding
    ]

    known_faces_names = [
        "jobs",
        "ratan tata",
        "mona lisa",
        "tesla",
        "abhi",
        "Hemanth"
    ]

    students = known_faces_names.copy()

    now = datetime.now()
    current_date = now.strftime("%Y-%m-%d")
    csv_filename = current_date + '.csv'
    f = open(csv_filename, 'w+', newline='')
    lnwriter = csv.writer(f)

    while is_running:
        _, frame = video_capture.read()
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
        face_names = []

        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_face_encoding, face_encoding)
            name = ""
            face_distance = face_recognition.face_distance(known_face_encoding, face_encoding)
            best_match_index = np.argmin(face_distance)
            if matches[best_match_index]:
                name = known_faces_names[best_match_index]

            face_names.append(name)

            if name in known_faces_names:
                font = cv2.FONT_HERSHEY_SIMPLEX
                bottomLeftCornerOfText = (10, 100)
                fontScale = 0.75
                fontColor = (255, 0, 0)
                thickness = 3
                lineType = 2

                cv2.putText(frame, name + ' marked Present'
                                          ' ready for next student',
                            bottomLeftCornerOfText,
                            font,
                            fontScale,
                            fontColor,
                            thickness,
                            lineType)

                if name in students:
                    students.remove(name)
                    print(students)
                    current_time = now.strftime("%H-%M-%S")
                    lnwriter.writerow([name,"was present ","on :",current_time])

        cv2.imshow("attendance system", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()
    f.close()
    messagebox.showinfo("Info", f"Face recognition stopped. Recognized images saved to {csv_filename}")

# Function to start the face recognition process
def start_face_recognition():
    global is_running
    if not is_running:
        is_running = True
        threading.Thread(target=run_face_recognition).start()
    else:
        messagebox.showwarning("Warning", "Face recognition is already running.")

# Function to stop the face recognition process
def stop_face_recognition():
    global is_running
    if is_running:
        is_running = False
    else:
        messagebox.showwarning("Warning", "Face recognition is not running.")

# Add labels for project details
project_title_label = tk.Label(root, text="Project Title: Authentication through Biometrics")
project_title_label.pack(padx=20, pady=(20, 0))

project_title_label = tk.Label(root, text="Current use case: Student Identification")
project_title_label.pack(padx=20, pady=(20, 0))

team_members_label = tk.Label(root, text="Presented by: Hemanth")
team_members_label.pack(padx=20, pady=5)

college_label = tk.Label(root, text="Sapthagiri College of Engineering")
college_label.pack(padx=20, pady=5)

# Button to start face recognition
start_button = tk.Button(root, text="Start Face Recognition", command=start_face_recognition)
start_button.pack(padx=20, pady=10)

# Button to stop face recognition
stop_button = tk.Button(root, text="Stop Face Recognition", command=stop_face_recognition)
stop_button.pack(padx=20, pady=10)

# Function to handle window closing
def on_closing():
    if messagebox.askokcancel("Quit", "Do you want to quit?"):
        root.destroy()
        if is_running:
            stop_face_recognition()

# Configure window closing behavior
root.protocol("WM_DELETE_WINDOW", on_closing)

# Run the Tkinter main loop
root.mainloop()