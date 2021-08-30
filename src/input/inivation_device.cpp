#include "inivation_device.hpp"

/* This function has been taken from the example list of the libcaer 
library <https://github.com/inivation/libcaer> developed by the iniVation AG. 

Copyright (c) 2014-2018, Luca Longinotti, iniVation AG
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice,
   this list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in the
   documentation and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
*/ 

int list_devices(){
	caerDeviceDiscoveryResult discovered;
	ssize_t result = caerDeviceDiscover(CAER_DEVICE_DISCOVER_ALL, &discovered);

	if (result < 0) {
		// Error!
		return (EXIT_FAILURE);
	}

	printf("\n");
	printf("Device discovery: found %zi devices.\n", result);
	printf("\n");

	for (size_t i = 0; i < (size_t) result; i++) {
		switch (discovered[i].deviceType) {
			case CAER_DEVICE_DVS128:
				printf("- DVS128\n");
				printf("  - USB busNum:devAddr: %d:%d\n", discovered[i].deviceInfo.dvs128Info.deviceUSBBusNumber,
					discovered[i].deviceInfo.dvs128Info.deviceUSBDeviceAddress);
				printf("  - Device can be opened: %d\n", !discovered[i].deviceErrorOpen);
				if (!discovered[i].deviceErrorOpen) {
					printf("  - USB serial number: %s\n", discovered[i].deviceInfo.dvs128Info.deviceSerialNumber);
					printf("  - Device needs firmware update: %d\n", discovered[i].deviceErrorVersion);

					if (!discovered[i].deviceErrorVersion) {
						printf("  - Timestamp Master: %d\n", discovered[i].deviceInfo.dvs128Info.deviceIsMaster);
						printf("  - Firmware Version: %d\n", discovered[i].deviceInfo.dvs128Info.firmwareVersion);
						printf("  - DVS Size X: %d\n", discovered[i].deviceInfo.dvs128Info.dvsSizeX);
						printf("  - DVS Size Y: %d\n", discovered[i].deviceInfo.dvs128Info.dvsSizeY);
					}
				}
				printf("\n");
				break;

			case CAER_DEVICE_EDVS:
				// If a serial device exists, it means it could be opened. Firmware version is not checked.
				printf("- EDVS-4337\n");
				printf("  - COM port: '%s'.\n", discovered[i].deviceInfo.edvsInfo.serialPortName);
				printf("  - Baud rate: %d.\n", discovered[i].deviceInfo.edvsInfo.serialBaudRate);
				printf("  - Timestamp Master: %d\n", discovered[i].deviceInfo.edvsInfo.deviceIsMaster);
				printf("  - DVS Size X: %d\n", discovered[i].deviceInfo.edvsInfo.dvsSizeX);
				printf("  - DVS Size Y: %d\n", discovered[i].deviceInfo.edvsInfo.dvsSizeY);
				printf("\n");
				break;

			case CAER_DEVICE_DAVIS:
			case CAER_DEVICE_DAVIS_FX2:
			case CAER_DEVICE_DAVIS_FX3:
			case CAER_DEVICE_DAVIS_RPI:
				printf("- DAVIS (type %u)\n", discovered[i].deviceType);
				printf("  - USB busNum:devAddr: %d:%d\n", discovered[i].deviceInfo.davisInfo.deviceUSBBusNumber,
					discovered[i].deviceInfo.davisInfo.deviceUSBDeviceAddress);
				printf("  - Device can be opened: %d\n", !discovered[i].deviceErrorOpen);
				if (!discovered[i].deviceErrorOpen) {
					printf("  - USB serial number: %s\n", discovered[i].deviceInfo.davisInfo.deviceSerialNumber);
					printf("  - Device needs firmware update: %d\n", discovered[i].deviceErrorVersion);

					if (!discovered[i].deviceErrorVersion) {
						printf("  - Timestamp Master: %d\n", discovered[i].deviceInfo.davisInfo.deviceIsMaster);
						printf("  - Firmware Version: %d\n", discovered[i].deviceInfo.davisInfo.firmwareVersion);
						printf("  - Logic Version: %d\n", discovered[i].deviceInfo.davisInfo.logicVersion);
						printf("  - Chip ID: %d\n", discovered[i].deviceInfo.davisInfo.chipID);
						printf("  - DVS Size X: %d\n", discovered[i].deviceInfo.davisInfo.dvsSizeX);
						printf("  - DVS Size Y: %d\n", discovered[i].deviceInfo.davisInfo.dvsSizeY);
						printf("  - DVS Statistics: %d\n", discovered[i].deviceInfo.davisInfo.dvsHasStatistics);
						printf("  - DVS ROI Filter: %d\n", discovered[i].deviceInfo.davisInfo.dvsHasROIFilter);
						printf("  - DVS Pixel Filter: %d\n", discovered[i].deviceInfo.davisInfo.dvsHasPixelFilter);
						printf("  - DVS Skip Filter: %d\n", discovered[i].deviceInfo.davisInfo.dvsHasSkipFilter);
						printf(
							"  - DVS Polarity Filter: %d\n", discovered[i].deviceInfo.davisInfo.dvsHasPolarityFilter);
						printf("  - DVS BA/Refr Filter: %d\n",
							discovered[i].deviceInfo.davisInfo.dvsHasBackgroundActivityFilter);
						printf("  - APS Size X: %d\n", discovered[i].deviceInfo.davisInfo.apsSizeX);
						printf("  - APS Size Y: %d\n", discovered[i].deviceInfo.davisInfo.apsSizeY);
						printf("  - APS Color Filter: %d\n", discovered[i].deviceInfo.davisInfo.apsColorFilter);
						printf("  - APS Global Shutter: %d\n", discovered[i].deviceInfo.davisInfo.apsHasGlobalShutter);
						printf(
							"  - External IO Generator: %d\n", discovered[i].deviceInfo.davisInfo.extInputHasGenerator);
						printf("  - Multiplexer Statistics: %d\n", discovered[i].deviceInfo.davisInfo.muxHasStatistics);
					}
				}
				printf("\n");
				break;

			case CAER_DEVICE_DYNAPSE:
				printf("- Dynap-SE\n");
				printf("  - USB busNum:devAddr: %d:%d\n", discovered[i].deviceInfo.dynapseInfo.deviceUSBBusNumber,
					discovered[i].deviceInfo.dynapseInfo.deviceUSBDeviceAddress);
				printf("  - Device can be opened: %d\n", !discovered[i].deviceErrorOpen);
				if (!discovered[i].deviceErrorOpen) {
					printf("  - USB serial number: %s\n", discovered[i].deviceInfo.dynapseInfo.deviceSerialNumber);
					printf("  - Device needs firmware update: %d\n", discovered[i].deviceErrorVersion);

					if (!discovered[i].deviceErrorVersion) {
						printf("  - Timestamp Master: %d\n", discovered[i].deviceInfo.dynapseInfo.deviceIsMaster);
						printf("  - Logic Version: %d\n", discovered[i].deviceInfo.dynapseInfo.logicVersion);
						printf("  - Logic Clock Frequency: %d\n", discovered[i].deviceInfo.dynapseInfo.logicClock);
						printf("  - Chip ID: %d\n", discovered[i].deviceInfo.dynapseInfo.chipID);
						printf("  - AER Statistics: %d\n", discovered[i].deviceInfo.dynapseInfo.aerHasStatistics);
						printf(
							"  - Multiplexer Statistics: %d\n", discovered[i].deviceInfo.dynapseInfo.muxHasStatistics);
					}
				}
				printf("\n");
				break;

			case CAER_DEVICE_DVXPLORER:
				printf("- DVXPLORER\n");
				printf("  - USB busNum:devAddr: %d:%d\n", discovered[i].deviceInfo.dvXplorerInfo.deviceUSBBusNumber, discovered[i].deviceInfo.dvXplorerInfo.deviceUSBDeviceAddress);
				printf("  - Device can be opened: %d\n", !discovered[i].deviceErrorOpen);
				if (!discovered[i].deviceErrorOpen) {
					printf("  - USB serial number: %s\n", discovered[i].deviceInfo.dvXplorerInfo.deviceSerialNumber);
					printf("  - Device needs firmware update: %d\n", discovered[i].deviceErrorVersion);

					if (!discovered[i].deviceErrorVersion) {
						printf("  - Timestamp Master: %d\n", discovered[i].deviceInfo.dvXplorerInfo.deviceIsMaster);
						printf("  - Firmware Version: %d\n", discovered[i].deviceInfo.dvXplorerInfo.firmwareVersion);
						printf("  - Logic Version: %d\n", discovered[i].deviceInfo.dvXplorerInfo.logicVersion);
						printf("  - Chip ID: %d\n", discovered[i].deviceInfo.dvXplorerInfo.chipID);
						printf("  - DVS Size X: %d\n", discovered[i].deviceInfo.dvXplorerInfo.dvsSizeX);
						printf("  - DVS Size Y: %d\n", discovered[i].deviceInfo.dvXplorerInfo.dvsSizeY);
						printf("  - DVS Statistics: %d\n", discovered[i].deviceInfo.dvXplorerInfo.dvsHasStatistics);
						printf("  - External IO Generator: %d\n", discovered[i].deviceInfo.dvXplorerInfo.extInputHasGenerator);
						printf("  - Multiplexer Statistics: %d\n", discovered[i].deviceInfo.dvXplorerInfo.muxHasStatistics);
					}
				}
				printf("\n");
				break;

			default:
				printf("- Unknown device type!\n");
				printf("\n");
				break;
		}
	}

	// Done!
	free(discovered);
	return (EXIT_SUCCESS);
}
