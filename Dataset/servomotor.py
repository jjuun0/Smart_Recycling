from gpiozero import Servo
from time import sleep

servo = Servo(18)
servo.mid()
sleep(1)

def infinite():
    while True:
        servo.min()
        sleep(1)
        servo.mid()
        sleep(1)
        servo.max()
        sleep(1)
    
def setPos(class_name):
    print(class_name)
    if class_name == 'glass':
        servo.min()
        sleep(1)
        print(class_name)
    elif class_name == 'metal':
        servo.mid()
        sleep(1)
        print(class_name)
    elif class_name == 'plastic':
        servo.max()
        sleep(1)
        print(class_name)
        
if __name__ == '__main__':
    setPos('plastic')
    setPos('metal')
    
    
