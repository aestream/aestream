#include <tuple>
#include <vector>

#include "./transformation.hpp"

typedef std::vector<std::vector<std::tuple<int, int>>> lut;

inline int coordinates_to_index(AER::Event event, size_t height) {
    return event.x * height + event.y;
}

/* Function: create_new_lut
 * ---------------------
 * Fills a LUT, an array of type lutmap, with naive mapping from (x, y) -> (x, y)
 *
 * width: expected image width
 * height: expected image height
 *
 */
lut create_new_lut(int width, int height){    
    lut lut = {};
    for (int x = 0; x < width; x++) {
        for (int y = 0; y < height; y++) {
            std::vector<std::tuple<int, int>> mapping {{x, y}};
            lut.push_back(mapping);
        }
    }
    return lut;
} 

/* Function: load_undistortion_lut
 * ---------------------
 * Fills a LUT with content from a *.csv file
 *
 * fname: name of the *.csv file (can be "")
 * width: expected image width
 * height: expected image height
 * lut: pointer towards allocated LUT
 *
 */
// void load_undistortion_lut(const std::string & fname, int width, int height, lutmap lut[]){
    
//     // File pointer
//     std::ifstream infile(fname);  

//     vector<string> row;
//     string line, val, temp;  

//     while (std::getline(infile, line))
//     {
//         stringstream s(line);
  
//         int col_idx = 0;
//         int lut_idx;
//         uint8_t np = 0;
//         bool done = false;
//         while (getline(s, val, ',')) 
//         {
//             switch(col_idx) {
//                 case 0:
//                     lut_idx = stoi(val);
//                     np = 0;
//                     break;
//                 case 1:
//                     lut[lut_idx].x[0]= stoi(val);
//                     if (lut[lut_idx].x[0] >= 0){
//                         np += 1;
//                     }
//                     break;
//                 case 2:
//                     lut[lut_idx].y[0]= stoi(val);
//                     lut[lut_idx].np= np;
//                     break;
//                 case 3:
//                     lut[lut_idx].x[1]= stoi(val);
//                     if (lut[lut_idx].x[1] >= 0){
//                         np += 1;
//                     }
//                     break;
//                 case 4:
//                     lut[lut_idx].y[1]= stoi(val);
//                     lut[lut_idx].np= np;
//                     break;
//                 case 5:
//                     done = true;
//                     break;
//             }
//             col_idx += 1;
//         }
//     }
    
// }

/* Function: from_string_to_trans
 * ---------------------
 * Maps string to element of type 'trans'
 *
 * requested_trans: name of requested transformation
 *
 * returns: a value from enum 'trans'
 *
 */
trans from_string_to_trans(const std::string &requested_trans) {
    trans transformation = no_trans;

    if(requested_trans.compare("no_trans") == 0){
        transformation = no_trans;
    } else if(requested_trans.compare("rot_90") == 0){
        transformation = rot_90;
    } else if(requested_trans.compare("rot_180") == 0){
        transformation = rot_180;
    } else if(requested_trans.compare("rot_270") == 0){
        transformation = rot_270;
    } else if(requested_trans.compare("flip_ud") == 0){
        transformation = flip_ud;
    } else if(requested_trans.compare("flip_lr") == 0){
        transformation = flip_lr;
    } else {
        throw std::runtime_error("Unknown transformation type: " + requested_trans);
    }

    return transformation;
}


/* Function: trans_lut
 * ---------------------
 * Maps string to element of type 'trans'
 *
 * requested_trans: name of requested transformation
 *
 */
// void trans_lut(uint16_t max_dim, uint16_t i_width, uint16_t i_height, lut &lut, trans trans_type, uint8_t s_sample){

//     uint16_t o_width = i_width;
//     uint16_t o_height= i_height;
    
//     /* Define an auxiliary, empty,  LUT*/
//     lutmap * aux = (lutmap*)malloc(max_dim*max_dim*sizeof(lutmap));
//     get_new_lut(o_width, o_height, aux, true);  

//     uint16_t x, y, new_x, new_y;
//     bool cross;

//     /* Fill auxiliary LUT with re-organized content from main LUT */
//     for(uint64_t i = 0; i < o_width*o_height; i++) {

//         x = i/o_height;
//         y = i%o_height;        

//         /* Only process 1 out of s_sample*s_sample events*/
//         if(x%s_sample==0 && y%s_sample==0){
//             /* Map (x, y) to (new_x, new_y) + determine if cross=true/false*/
//             switch(trans_type) {
//                 case rot_90:
//                     new_x = o_width-1-x;
//                     new_y = y;
//                     cross = true;
//                     break;
//                 case rot_180:
//                     new_x = o_width-1-x;
//                     new_y = o_height-1-y;
//                     cross = false;
//                     break;
//                 case rot_270:
//                     new_x = x;
//                     new_y = o_height-1-y;
//                     cross = true;
//                     break;
//                 case flip_lr:
//                     new_x = o_width-1-x;
//                     new_y = y;
//                     cross = false;
//                     break;
//                 case flip_ud:
//                     new_x = x;
//                     new_y = o_height-1-y;
//                     cross = false;
//                     break;
//                 default:
//                     new_x = x;
//                     new_y = y;
//                     cross = false;
//                     break;
//             }

//             /* Update auxiliary LUT with current element of main LUT*/
//             for (int pixix = 0; pixix < MAX_PIXIX; pixix++) {
//                 if(cross){
//                     aux[new_x*o_width+new_y].np = lut[i].np;
//                     aux[new_x*o_width+new_y].x[pixix] = lut[i].y[pixix]/s_sample;            
//                     aux[new_x*o_width+new_y].y[pixix] = lut[i].x[pixix]/s_sample; 
//                 } else {
//                     aux[new_x*o_height+new_y].np = lut[i].np;
//                     aux[new_x*o_height+new_y].x[pixix] = lut[i].x[pixix]/s_sample;            
//                     aux[new_x*o_height+new_y].y[pixix] = lut[i].y[pixix]/s_sample;  
//                 }
//             }
//         }
//     }

//     /* Copy auxiliary LUT into main LUT */
//     memcpy(lut, aux, max_dim*max_dim*sizeof(lutmap)); 

//     /* Change weight <--> height if needed (depends on requested transformation) */
//     if (cross){
//         *i_width = o_height;
//         *i_height = o_width;
//     }    
// }


Generator<AER::Event>
transformation_event_generator(Generator<AER::Event> &input_generator,
                       const std::string &undistortion_filename, trans transformation, 
                       uint16_t width, uint16_t height, uint8_t t_sample, uint8_t s_sample) {

    lut lut;

    /* Fill the main LUT with meaningful data: with or without undistortion */
    if(undistortion_filename.length() > 0){        
        // load_undistortion_lut(undistortion_filename, width, height, lut);
        std::runtime_error("Cannot import LUT files yet");
    } else {
        lut = create_new_lut(width, height);
    }
    
    // /* Transform LUT according to user's request (including spatial sampling) */
    // // trans_lut(max_dim, &width, &height, lut, transformation, s_sample);

    uint64_t count;

    // /* Get events from 'input_generator' */
    for (auto event : input_generator) {
        for (auto [x, y] : lut[coordinates_to_index(event, height)]) {
            AER::Event new_event = {event.timestamp, x, y, event.polarity};
            
            /* Yield 1 out of t_sample events (temporal sampling)*/
            if (count % t_sample == 0) {
                co_yield event;
            }
            count+= 1;
        }
    }   
}