import cv2

from utils.npsocket import SocketNumpyArray

cap = cv2.VideoCapture(0)
sock_sender = SocketNumpyArray()

sock_sender.initialize_sender('192.168.1.11', 8888)

while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, (620, 480))
    sock_sender.send_numpy_array(frame)