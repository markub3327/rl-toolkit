import cv2

from utils.npsocket import SocketNumpyArray

sock_receiver = SocketNumpyArray()
sock_receiver.initalize_receiver(8888)

while True:
    frame = sock_receiver.receive_array()
    frame = cv2.applyColorMap(frame, cv2.COLORMAP_JET)

    # Display
    cv2.imshow('frame', frame)
    cv2.waitKey(1)