#include "dvs2tensor.hpp"
#include <string>
#include <iostream>


int main(int argc, char *argv[]) {
	std::string command = "Nonset";
	uint32_t interval, buffer_size, nr_events;
	
	int id = 1;
	int exitcode;

	if (argc < 4) {
        fprintf(stderr,"usage: ./stream2tensor <max_packet_time_μs> <buffer size> <number_packets_2_process>\n");
        exit(1);
    }

	interval = strtol(argv[1], NULL, 0);
	buffer_size = strtol(argv[2], NULL, 0);
	nr_events = strtol(argv[3], NULL, 0);

	DVSDataConv dvsdata(interval, buffer_size); 

	dvsdata.connect2camera(id);
	dvsdata.startdatastream();

	for(int i=0; i<nr_events; i++){
		auto tensor = dvsdata.update();
	}

	exitcode = dvsdata.stopdatastream();
}