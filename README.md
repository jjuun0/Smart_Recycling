# 2021.04. ~ 2021.05.12  
- 현재까지 garbage_classification 기술 적용해봤다.  
- 팀원들과 코드 공유를 위해 깃허브를 추가함.     
- 나의 네이버 블로그에 그동안 어떤 모델을 사용했는지 적어둠. 
  - https://blog.naver.com/ggghhh97/222337946341  
  - 모델별로 정확도, 학습도중 loss 값을 그래프로 표현하여 비교해볼 수 있다.  
  - 그중 resnet50, googlenet 이 성능이 좋아서 둘을 학습 데이터중 랜덤으로 뽑아서 테스트 해본 결과 4%, 5% 에러율 가졌다.  
  - 내 데이터셋(mydata)을 가지고 평가해봤는데 둘다 에러율이 높아서 다른 모델을 사용해보던지, 다른 기술을 찾아봐야함.  
  
- 해야할 것  
  - 이런 기술을 윈도우상에서 모델을 돌려봤는데, 후에 라즈베리파이에서 돌려야한다. 이 방법을 찾아봐야함.  
  - 현재 모터가 180도만 회전이 가능해 360도 돌리는 모터를 찾아보고 구매해야함.  
  - 라즈베리파이 쿨러케이스도 구매해야함.  


# 2021.05.17  
- 문제 : 카메라로 촬영한 사진이 제대로 성능이 안나온다.  
  - 데이터셋을 다른것을 사용해보자.  
  - 학습시킬때 이미지를 정규화를 시켜보자. -> 성능 좋아짐 resnet50 사용시
  - object localization 을 이용해서 사진속에서 물체만 감지해 잘라낸 사진으로 학습시킨다.  
  - object detection 을 이용해보자.  
- 윈도우 말고 라즈베리파이에서 돌리는방법
  -  flask 에서 모델 배포 : 캠으로 사진을 찍어서 서버로 전송하고, 서버에서 사진을 받아 모델이 예측한 결과 값을 뿌려준다.
  -  edge TPU 
  -  aws torchserve(?)

# 2021.05.20  
- coral usb 로 gpu 처럼 돌릴수 있는 방법이 있으나 추천하지 않음.  
- 데이터 셋을 더 많이 구축해야한다.  
  - augmentation (ex. cutmix,,)   
- attention 기법 사용해봐도 됨.  
- Q. 데이터 셋 이미지에서 물체를 detection 한 결과를 bounding box 로 그려서 이미지를 crop 해 다시 데이터 셋을 만들고 학습을 시키면 더 똑똑해지지 않을까요?  
- A. 아니다. 오히려 negative 데이터가 많아야지 모델이 더 똑똑해지는데, crop 한다면 negative 값 데이터가 사라지므로 성능은 떨어질 것이다.  
- 플라스크 (서버 - 클라이언트) 방법을 사용해 라즈베리파이에서 캠으로 사진을 찍어 서버로 데이터를 보내, 처리를 하자.  

# 2021.05.21  
- 라즈베리파이에서 pytorch 설치후, 윈도우에서 학습한 모델을 가져와, cpu 로 모델을 돌려보니 이미지 한장당 12초 정도 처리하는데 시간이 걸린다.  
  ![image](https://user-images.githubusercontent.com/66052461/119105686-89fed600-ba58-11eb-9868-cff6e5878f4e.png)  

# 2021.05.25 
- 소켓 통신 방법으로 사진을 클라이언트에서 서버로 전송하고, 서버에서 받은 사진으로 모델에 돌려서 예측한 값을 클라이언트에 전송.  
- 서버 : 노트북, 클라이언트 : 라즈베리파이  
- 9초 정도 걸린다. (위의 cpu 방법보다는 빠르다.)
- 서버 -> socket_server.py  
![image](https://user-images.githubusercontent.com/66052461/119500307-9cec1000-bda2-11eb-8f49-ebe946599da8.png)  
- 클라이언트 -> client.py  
![image](https://user-images.githubusercontent.com/66052461/119500437-c60ca080-bda2-11eb-95b0-4634a33bf180.png)

# 2021.05.27  
- 서버 코드 수정  
  - (수정전) : 클라이언트와 연결후에 모델 로드와 클라이언트에서 보낸 이미지를 인풋으로 넣음 -> 9초  
  - (수정후) : 클라이언트 연결전에 모델 로드 우선적으로 한다, 다음에 클라이언트 연결을 통해 이미지를 받고 연산할 수 있게끔 수정 -> 3초 대로 줄어듦  
- 연산 속도 처리 : 이미지 디코딩 방법  
  - cv2.imdecode() VS io.BytesIO()  
    - cv2 가 조금 더 빠르다  
    - cv2.imdecode() : 3.29 초  
    ![opencv_open](https://user-images.githubusercontent.com/66052461/119762923-8d2b1380-bee9-11eb-9d7e-c9eb817a37a0.PNG)  
    - io.BytesIO() : 3.62 초  
    ![io bytesio_open](https://user-images.githubusercontent.com/66052461/119762919-8b615000-bee9-11eb-8f7b-14c2db5ecc15.PNG)  
    
# 2021.07.13 
- demo_focal.py
- pytorch 프레임워크로 Custom Dataset 을 구축후에 Dataloader 를 통하여 모델을 학습하고 평가함
- tensorboard 를 통해 모델, 학습하는데 train loss, acc 와 val loss, acc 도 시각화 하여 사진으로도 저장
- sot 모델로 mot 개발중임 : 사용자가 selectroi 로 객체를 선택하고 트래킹하는 방식
  - 에러 : 객체가 서로 가까우면 BBox 가 업데이트 되면서 물체가 같아져버림
