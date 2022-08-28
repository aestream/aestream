from sqlite3 import Timestamp
from metavision_core.event_io import EventsIterator, DatWriter
from metavision_sdk_core import PeriodicFrameGenerationAlgorithm
from metavision_sdk_ui import EventLoop, BaseWindow, Window, UIAction, UIKeyEvent
from concurrent.futures import ThreadPoolExecutor

import sys, os
import time
import numpy as np
import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from lsm9ds1_rjg import Driver, I2CTransport, SPITransport


last_val = None # if imu don't record data, it takes the last value recorded


f = open("IMUnoise.txt",'w+') 
g = open("CAMevents.txt", 'w+')
h = open("IMUCAM.txt", "w+")


def function_process_data(imu_data, cam_data):

    if len(imu_data) == 0: # in this way, we consider also delays or bottleneck from imu reader (buffer)
        imu_data = last_val
    else:
        last_val = imu_data

    # idea - create two streams and stamp values of imu and cam on files

    for data in imu_data:

        # _, acc, gyro = data
        temp, acc, gyro = data
        
        # get timestamp
        ct = datetime.datetime.now()
        ts = ct.timestamp()
        
        np.savetxt(f,np.array([ts]+[temp]+acc+gyro) [None,:],fmt='%i %i %i %i %i %i %i %i', newline='\n')


    for eventi in cam_data:
        np.savetxt(g,eventi,fmt='%i %i %i %i', newline='\n')

    # now, we take the timestamp correspondence and stamp it one per line (t x y p ax ay ax gx gy gz)

    for data, eventi in imu_data, cam_data:
        if eventi.t == data.ts: #TODO correct this function
            np.savetxt(h, eventi+np.array(acc+gyro) [None,:],fmt='%i %i %i %i %i %i %i %i %i %i', newline='\n')
            #now, convert it into aedat4





########## VIDEO ##########

class CamExecute:


    def main():
        """ Main """

        mv_iterator = EventsIterator(input_path='', delta_t=1000)

        height, width = mv_iterator.get_size()  # Camera Geometry (640x480)
     
        # Process events
        for evs in mv_iterator:

            eventi = np.c_[evs['t'], evs['x'], evs['y'], evs['p']]
            # return evs in tuple (one event per line - we can find events-imu correspondence thanks to timestamp t)
            return eventi    




######### IMU (model LSM9DS1) #########

class ImuExecute:

    def __init__(self):
        self.driver = self._create_i2c_driver()
        self.driver.configure()

    @staticmethod
    def _create_i2c_driver() -> Driver:
        return Driver(
            I2CTransport(1, I2CTransport.I2C_AG_ADDRESS),
            I2CTransport(1, I2CTransport.I2C_MAG_ADDRESS))

    @staticmethod
    def _create_spi_driver() -> Driver:
        return Driver(
            SPITransport(0, False),
            SPITransport(1, True))

    def main(self):

        data = []

        try:
            start = time.time()
            
            while time.time() - start < 0.001: 
                ag_data_ready = self.driver.read_ag_status().accelerometer_data_available

                if ag_data_ready:
                    data.append(self.read_ag())
                else:
                    time.sleep(0.00001)

        finally:
            self.driver.close()
            return data

    def read_ag(self):        
        return self.driver.read_ag_data()




######### main #########

if __name__ == '__main__':

    while True:
        with ThreadPoolExecutor(max_workers=2) as executor:
            imu = executor.submit(ImuExecute().main) # main imu
            cam = executor.submit(CamExecute().main) # main cam

        imu_data = imu.result() 
        cam_data = cam.result()

        function_process_data(imu_data, cam_data)




