from pydobot import Dobot

device = Dobot(port='/dev/ttyUSB0')  # แก้ port ให้ตรงเครื่องคุณ

# list method ทั้งหมดที่เกี่ยวกับ alarm
methods = [m for m in dir(device) if 'alarm' in m.lower()]
print(methods)