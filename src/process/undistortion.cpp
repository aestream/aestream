
#include "./undistortion.hpp"
using namespace std;

void print_lut(int width, int height, map lut[]){
    for (int x = 0; x < width; x++) {
        for (int y = 0; y < height; y++) {  
            printf("(%i, %i): ", x, y);
            printf("%i --> ", lut[x*height+y].np);
            printf("(%i, %i)\t\t", lut[x*height+y].p[0].x, lut[x*height+y].p[0].y);
            printf("(%i, %i)\n", lut[x*height+y].p[1].x, lut[x*height+y].p[1].y);
        }
    }
}

void count_stuff(int width, int height, map lut[]){

    int count_0 = 0;
    int count_1 = 0;
    int count_2 = 0;

    
    for(int idx=0; idx < width*height; idx++){
        switch(lut[idx].np){
            case 0:
                count_0 += 1;
                break;
            case 1:
                count_1 += 1;
                break;
            case 2:
                count_2 += 1;
                break;

        }
    }

    printf("%i with 0 maps\n", count_0);
    printf("%i with 1 maps\n", count_1);
    printf("%i with 2 maps\n", count_2);
}

void get_empty_lut(int width, int height, map lut[]){
    

    for (int x = 0; x < width; x++) {
        for (int y = 0; y < height; y++) {           
            lut[x*height+y].np = 0;
            lut[x*height+y].p[0].x = -1;            
            lut[x*height+y].p[0].y = -1;         
            lut[x*height+y].p[1].x = -1;      
            lut[x*height+y].p[1].y = -1;
        }
    }
}


void load_lut(const std::string & fname, int width, int height, map lut[]){

    
    get_empty_lut(width, height, lut);   

    // File pointer
    std::ifstream infile(fname);  

    vector<string> row;
    string line, val, temp;  

    while (std::getline(infile, line))
    {
        stringstream s(line);
  
        int tok_ix = 0;
        int np = 0;
        int lutix;
        bool done = false;
        while (getline(s, val, ',')) 
        {
            switch(tok_ix) {
                case 0:
                    lutix = stoi(val);
                    break;
                case 1:
                    lut[lutix].p[0].x= stoi(val);
                    if (lut[lutix].p[0].x >= 0){
                        np += 1;
                    }
                    break;
                case 2:
                    lut[lutix].p[0].y= stoi(val);
                    lut[lutix].np= np;
                    break;
                case 3:
                    lut[lutix].p[1].x= stoi(val);
                    if (lut[lutix].p[1].x >= 0){
                        np += 1;
                    }
                    break;
                case 4:
                    lut[lutix].p[1].y= stoi(val);
                    lut[lutix].np= np;
                    break;
                case 5:
                    done = true;
                    break;
            }
            tok_ix += 1;
        }
    }
    
    
}


#define MAX_W 1280
#define MAX_H 720

Generator<AEDAT::PolarityEvent>
undistortion_event_generator(Generator<AEDAT::PolarityEvent> &input_generator,
                       const std::string &filename, const std::uint16_t width, const std::uint16_t height) {
 
    map lut[MAX_W*MAX_H];

    uint16_t new_x;
    uint16_t new_y;
    
    int count = 0;
    load_lut(filename, width, height, lut);                

    for (auto event : input_generator) {
        // co_yield event;
        for(int pixix = 0; pixix < lut[event.x*height+event.y].np; pixix++){
            new_x = lut[event.x*height+event.y].p[pixix].x;
            new_y = lut[event.x*height+event.y].p[pixix].y;
            event.x = new_x;
            event.y = new_y;        
            co_yield event;
        } 
    }          

}