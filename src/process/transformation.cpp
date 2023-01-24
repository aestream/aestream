
#include "./transformation.hpp"
using namespace std;


#define MAX_W 1280
#define MAX_H 720

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

void get_one2one_lut(int width, int height, map lut[]){
    

    for (int x = 0; x < width; x++) {
        for (int y = 0; y < height; y++) {           
            lut[x*height+y].np = 1;
            lut[x*height+y].p[0].x = x;            
            lut[x*height+y].p[0].y = y;         
            lut[x*height+y].p[1].x = -1;      
            lut[x*height+y].p[1].y = -1;
        }
    }
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

void load_undistortion_lut(const std::string & fname, int width, int height, map lut[]){

    
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


  trans from_string_to_trans(std::string requested_trans){

    trans transformation = no_trans;

    if(requested_trans.compare("no_trans") == 0){
        transformation = no_trans;
    }
    if(requested_trans.compare("rot_90") == 0){
        transformation = rot_90;
    }
    if(requested_trans.compare("rot_180") == 0){
        transformation = rot_180;
    }
    if(requested_trans.compare("rot_270") == 0){
        transformation = rot_270;
    }
    if(requested_trans.compare("flip_ud") == 0){
        transformation = flip_ud;
    }
    if(requested_trans.compare("flip_lr") == 0){
        transformation = flip_lr;
    }

    return transformation;

  }

void update_lut_element(map lut[], map aux[], int ix_lut, int ix_aux){  
    for (int pixix = 0; pixix < MAX_PIXIX; pixix++) {
        aux[ix_aux].np = lut[ix_lut].np;
        aux[ix_aux].p[pixix].x = lut[ix_lut].p[pixix].x;            
        aux[ix_aux].p[pixix].y = lut[ix_lut].p[pixix].y;  
    }
}



void trans_lut(std::uint16_t * i_width, std::uint16_t * i_height, map lut[], map aux[], trans trans_type, uint8_t s_sample){

    if(trans_type != no_trans){

        uint16_t o_width = *i_width;
        uint16_t o_height= *i_height;

        uint16_t x = 0;
        uint16_t y = 0;
        uint16_t new_x = 0;
        uint16_t new_y = 0;
        uint16_t new_h = 0;
        for(uint64_t i = 0; i < o_width*o_height; i++) {
            x = i/o_height;
            y = i%o_height;
            switch(trans_type) {
                case rot_90:
                    new_x = (o_height-1-y);
                    new_y = x;
                    new_h = o_width;
                    break;
                case rot_180:
                    new_x = (o_width-x-1);
                    new_y = (o_height-y-1);
                    new_h = o_height;
                    break;
                case rot_270:
                    new_x = y;
                    new_y = (o_width-1-x);
                    new_h = o_width;
                    break;
                case flip_lr:
                    new_x = (o_width-x-1);
                    new_y = y;
                    new_h = o_height;
                    break;
                case flip_ud:
                    new_x = x;
                    new_y = (o_height-y-1);
                    new_h = o_height;
                    break;
            }
            update_lut_element(lut, aux, x*o_height+y, new_x*new_h+new_y);
        }

        if (trans_type == rot_90 || trans_type == rot_270){
            *i_width = o_height;
            *i_height = o_width;
        }

        memcpy(lut, aux, MAX_W*MAX_H*sizeof(map)); 
    }
    
}



Generator<AEDAT::PolarityEvent>
transformation_event_generator(Generator<AEDAT::PolarityEvent> &input_generator,
                       const std::string &undistortion_filename, trans transformation, 
                       std::uint16_t width, std::uint16_t height, uint8_t t_sample, uint8_t s_sample) {
 
    map lut[MAX_W*MAX_H];
    map aux[MAX_W*MAX_H];
    
    get_one2one_lut(width, height, lut);    
    get_empty_lut(width, height, aux);
    
    if(undistortion_filename.length() > 0){
        load_undistortion_lut(undistortion_filename, width, height, lut);
    }
    trans_lut(&width, &height, lut, aux, transformation, s_sample);

    uint16_t new_x;
    uint16_t new_y;
    uint64_t count;
    for (auto event : input_generator) {
        // co_yield event;
        for(int pixix = 0; pixix < lut[event.x*height+event.y].np; pixix++){
            new_x = lut[event.x*height+event.y].p[pixix].x;
            new_y = lut[event.x*height+event.y].p[pixix].y;
            event.x = new_x;
            event.y = new_y;
            if(count%t_sample == 0){
                co_yield event;
            }
            count+= 1;
        }
    }          

}