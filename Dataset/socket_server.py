import socket
import cv2
import pickle
import struct
import test
import torch
import torchvision.transforms as transforms
from model import ResNet
import main
from PIL import Image
from io import BytesIO
import time


ip = '192.168.35.19'  # ip 주소
port = 9999  # port 번호

# device = torch.device('cuda')
# classes = ['glass', 'plastic', 'metal']
# trans_1 = transforms.Compose(
#     [transforms.Resize((256, 256)), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
# trans_2 = transforms.Compose(
#     [transforms.ToPILImage(), transforms.Resize((256, 256)), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
# model = ResNet(classes)
# model.load_state_dict(torch.load('./' + main.MODELNAME, map_location=device))
# model.eval()
# model.cuda()
# print("모델 로딩 끝")

classes = ['glass', 'metal', 'plastic']

device = torch.device('cuda:0')
model = ResNet(classes).to(device)
model.load_state_dict(torch.load('model/model3', map_location=device))
model.eval()
print("모델 로딩 끝")

test_transformations = transforms.Compose([transforms.Resize((256, 256)),
                                               # transforms.RandomHorizontalFlip(),
                                               transforms.ToTensor()
                                               # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                                               ])


s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)  # 소켓 객체를 생성
s.bind((ip, port))  # 바인드(bind) : 소켓에 주소, 프로토콜, 포트를 할당
s.listen(10)  # 연결 수신 대기 상태(리스닝 수(동시 접속) 설정)
print('클라이언트 연결 대기')

# 연결 수락(클라이언트 소켓 주소를 반환)
conn, addr = s.accept()
print(addr)  # 클라이언트 주소 출력
print("클라이언트와 연결 완료")

data = b""  # 수신한 데이터를 넣을 변수
payload_size = struct.calcsize(">L")

# while True:
    # 프레임 수신
while len(data) < payload_size:
    data += conn.recv(4096)
packed_msg_size = data[:payload_size]
data = data[payload_size:]
msg_size = struct.unpack(">L", packed_msg_size)[0]
while len(data) < msg_size:
    data += conn.recv(4096)
frame_data = data[:msg_size]
data = data[msg_size:]
print("Frame Size : {}".format(msg_size))  # 프레임 크기 출력

# 역직렬화(de-serialization) : 직렬화된 파일이나 바이트를 원래의 객체로 복원하는 것
frame = pickle.loads(frame_data, fix_imports=True, encoding="bytes")# 직렬화되어 있는 binary file로 부터 객체로 역직렬화


# cv2.imdecode : 3.3748 초
time_cv = time.time()
frame_cv = cv2.imdecode(frame, cv2.IMREAD_COLOR)  # 프레임 디코딩
# frame : ndarray
# percent, predict = test.predict_cuda(frame)
frame_pil = Image.fromarray(frame_cv)
frame_pil = test_transformations(frame_pil)
confidence, predict = test.socket(device, model, classes, frame_pil)
print("opencv : ", time.time() - time_cv)


# # io.bytesio : 3.5282 초
# time_pil = time.time()
# bytes_io = bytearray(frame)
# frame_pil = Image.open(BytesIO(bytes_io))
# frame_pil = trans_1(frame_pil)
# percent, predict = test.predict(device, model, classes, frame_pil)
# print("pil : ", time.time() - time_pil)


conn.sendall(predict.encode())
print(confidence, predict)
# 영상 출력
cv2.imshow('TCP_Frame_Socket', frame_cv)

# 1초 마다 키 입력 상태를 받음
key = cv2.waitKey()
if key == ord('q'):  # q를 입력하면 종료
    # break
    exit()