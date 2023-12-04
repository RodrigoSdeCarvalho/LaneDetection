''' =====================================================
Add include path to the root of this project, if not already added.
'''
import sys, os
found = False
for p in sys.path:
    if "sdav" in p and len(p.split("sdav/")) == 1:
        found = True
        break
if (not found):
    for p in sys.path:
        if "/sdav" in p:
            sys.path.append(p.split("sdav/")[0]+"sdav/")
            break
''' ===================================================== '''

from smartdata.data_model import Data_Model
import socket
import struct

class SmartData_Support:
    _MCAST_GRP = '224.1.1.1'
    _MCAST_PORT = 5000
    _MULTICAST_TTL = 2

    def __init__(self, name, transform_into_sd):
        self._port = self._MCAST_PORT + Data_Model.data_model[name][0] + 1
        print(self._port)
        self._socket = self.__create_socket()
        self._transform_into_sd = transform_into_sd
        self._log_path = os.getcwd().split("sdav/")[0] + "sdav/external/lane_detection/logs/"

    def __create_socket(self):
        soc = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
        soc.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        soc.bind((self._MCAST_GRP, self._port))
        mreq = struct.pack("4sl", socket.inet_aton(self._MCAST_GRP), socket.INADDR_ANY)
        soc.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)
        return soc

    def __send_socket(self, byte_array_value):
        self._socket.sendto(byte_array_value,(self._MCAST_GRP, self._port))

    def __read_socket(self, size):
        return self._socket.recv(size)

    def write_data(self, data):
        self.__send_socket(self._transform_into_sd(data))

    def read_data(self, size=65536):
        return self.__read_socket(size)

    @staticmethod
    def return_semantic_empty_byte_array(data): # used when no transducer is available
        return b'\xFF\xFF\xFF\xFF'

    @staticmethod
    def data_is_empty(data):
        return data[:4] == b'\xFF\xFF\xFF\xFF'


class Waypoint():
    def __init__(self, x=0, y=0, z=0, speed=0, heading=0, yaw_rate=0, accel=0, width=0, length=0, height=0) -> None:
        self._x = x
        self._y = y
        self._z = z
        self._speed = speed
        self._heading = heading
        self._yaw_rate = yaw_rate
        self._accel = accel
        self._width = width
        self._length = length
        self._height = height

    @staticmethod
    def to_bytes(w):
        byte_array = bytes()
        byte_array += struct.pack('1d', w._x)
        byte_array += struct.pack('1d', w._y)
        byte_array += struct.pack('1d', w._z)
        byte_array += struct.pack('1f', w._speed)
        byte_array += struct.pack('1f', w._heading)
        byte_array += struct.pack('1f', w._yaw_rate)
        byte_array += struct.pack('1f', w._accel)
        byte_array += struct.pack('1f', w._width)
        byte_array += struct.pack('1f', w._length)
        byte_array += struct.pack('1f', w._height)

        return byte_array

    @staticmethod
    def from_byte_array_to_waypoint(bytes):
        x           = struct.unpack('1d', bytes[  : 8])
        y           = struct.unpack('1d', bytes[ 8:16])
        z           = struct.unpack('1d', bytes[16:24])
        speed       = struct.unpack('1f', bytes[24:28])
        heading     = struct.unpack('1f', bytes[28:32])
        yaw_rate    = struct.unpack('1f', bytes[32:36])
        accel       = struct.unpack('1f', bytes[36:40])
        width       = struct.unpack('1f', bytes[40:44])
        length      = struct.unpack('1f', bytes[44:48])
        height      = struct.unpack('1f', bytes[48:52])

        return Waypoint(x, y, z, speed, heading, yaw_rate, accel, width, length, height)

    def __str__(self):
        return "Waypoint: x="+str(self._x)+" y="+str(self._y)+" z="+str(self._z)+" speed="+str(self._speed)+" heading="+str(self._heading)+" yaw_rate="+str(self._yaw_rate)+" accel="+str(self._accel)+" width="+str(self._width)+" length="+str(self._length)+" height="+str(self._height)

class Control:

    def __init__(self, on):
        self._on = on

    def from_control_to_bytes(self):
        return struct.pack('1B', self._on)

    @staticmethod
    def from_byte_array_to_control(byte):
        return Control(struct.unpack('1B',byte))
