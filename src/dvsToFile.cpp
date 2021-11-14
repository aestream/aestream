#include "dvsToFile.hpp"

template <typename T>
void DVSToFile(Generator<T>& input_generator, std::string filename){
    std::fstream fileOutput;
    fileOutput.open(filename, std::fstream::app);

    for (AEDAT::PolarityEvent event : input_generator) {
        if (event.valid == true){
            fileOutput << "DVS " << event.timestamp << " " << event.x << " " << event.y << " " << event.polarity << std::endl;
        }
    }

    fileOutput.close(); 
}

template void DVSToFile<AEDAT::PolarityEvent>(Generator<AEDAT::PolarityEvent>&, std::string);