import cv2
import RPi.GPIO as GPIO
import time
import numpy as np

# Load the cascade classifier for face detection
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Initialize the camera
cap = cv2.VideoCapture(0)

# Set the camera resolution to 640x480
cap.set(3, 1280)
cap.set(4, 720)

# Initialize GPIO 14 as an output pin
GPIO.setwarnings(False)
GPIO.setmode(GPIO.BCM)
GPIO.setup(14, GPIO.OUT)

# Initialize a timer variable
timer_start = None

# Create a window to show the camera view
cv2.namedWindow('Camera View', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Camera View', 711, 400)

# Create a window to display the GPIO output and GPIO state
cv2.namedWindow('GPIO Output', cv2.WINDOW_NORMAL)
cv2.resizeWindow('GPIO Output', 200, 200)
gpio_frame = np.zeros((200, 200, 3), dtype=np.uint8)
gpio_state = ''

def on_gpio_output_resize(event, x, y, flags, param):
    global gpio_frame
    if event == cv2.EVENT_WINDOW_RESIZED:
        width = cv2.getWindowImageRect('GPIO Output')[2]
        height = cv2.getWindowImageRect('GPIO Output')[3]
        gpio_frame = np.zeros((height, width, 3), dtype=np.uint8)

def update_gpio_frame():
    global gpio_frame, gpio_state
    if GPIO.input(14) == GPIO.HIGH:
        # Set the GPIO output window to green if the GPIO output is high
        gpio_frame[:, :, :] = (0, 255, 0)
        gpio_state = 'HIGH'
    else:
        # Set the GPIO output window to red if the GPIO output is low
        gpio_frame[:, :, :] = (0, 0, 255)
        gpio_state = 'LOW'
        
    # Add the GPIO state text to the GPIO output window
    text_size, _ = cv2.getTextSize(gpio_state, cv2.FONT_HERSHEY_SIMPLEX, 2, 2)
    text_x = int((gpio_frame.shape[1] - text_size[0]) / 2)
    text_y = int((gpio_frame.shape[0] + text_size[1]) / 2)
    cv2.putText(gpio_frame, gpio_state, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2)

cv2.setMouseCallback('GPIO Output', on_gpio_output_resize)
update_gpio_frame()
cv2.imshow('GPIO Output', gpio_frame)

while True:
    # Read the camera input
    ret, frame = cap.read()
   
    # Convert the input to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the input
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) == 0:
        # If no faces are detected, start the timer
        if timer_start is None:
            timer_start = time.time()
        else:
            # If the timer has been running for more than 5 seconds, set the GPIO output to high
            if time.time() - timer_start > 3:
                GPIO.output(14, GPIO.HIGH)
    else:
        # If faces are detected, stop the timer and set the GPIO output to low
        timer_start = None
        GPIO.output(14, GPIO.LOW)

        # Draw a rectangle around the faces
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Display the camera view in the first window
    cv2.imshow('Camera View', frame)

    # Update the GPIO output window
    update_gpio_frame()
    cv2.imshow('GPIO Output', gpio_frame)

    # Update the GPIO state window
    gpio_state_frame = np.zeros((200, 200, 3), dtype=np.uint8)
    text_size, _ = cv2.getTextSize(gpio_state, cv2.FONT_HERSHEY_SIMPLEX, 12, 12)
    
    # Exit the program if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and GPIO resources and close all windows
cap.release()
GPIO.cleanup()
cv2.destroyAllWindows()