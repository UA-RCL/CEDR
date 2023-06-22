// gaussian_blur and sobel_gx_operator includes repetitive code of convolution. That code can be declared as function.

#include <iostream>
#include <cstdio>
#include <cstddef>
#include <fstream>
#include <math.h>
#include <iomanip> 
#include "dash.h"
#include <vector>
#include <algorithm>

#error "Lane detection needs updates, the ZIP API has been changed"

#define SIGMA 0.75

#define PI 3.141593

// Header files for image read/write
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"

/*double blurred_img_canny[HEIGTH][WIDTH];
double gx_canny_img[HEIGTH][WIDTH];
double gy_canny_img[HEIGTH][WIDTH];
double sobel_canny_img[HEIGTH][WIDTH];
double angle[HEIGTH][WIDTH];
int mask[HEIGTH][WIDTH];
double lanes[HEIGTH][WIDTH];*/

struct Line{
    int x1, y1, x2, y2;

    Line(int x1, int y1, int x2, int y2): x1(x1), y1(y1), x2(x2), y2(y2){};
};

struct Point{
    int x, y;

    Point(int x, int y): x(x), y(y){};
};

// Triangle area operation
// Input: 
// x1, y1 --> first vertice's coordinates
// x2, y2 --> second vertice's coordinates
// x3, y3 --> third vertice's coordinates
// Output:
// return_array --> computed area of the rectangle
double area(int x1, int y1, int x2, int y2, int x3, int y3){
    return std::abs((x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)) / 2);
}


// Grayscale conversion operation
// Input: 
// img --> 3D image that is going to be grayscaled
// height --> image height
// width --> image width
// depth --> image depth(# of color channels)
// Output:
// return_array --> 2D grayscaled image
void grayscale_conversion(double *img, int height, int width, int depth, double *return_array){
    double r, g, b, gray;
    for(int i = 0; i < height; i++){
        for(int j = 0; j < width; j++){
                r = (double)img[(i * width + j) * depth + 0];
                g = (double)img[(i * width + j) * depth + 1];
                b = (double)img[(i * width + j) * depth + 2];

                r *= 0.299;
                g *= 0.587;
                b *= 0.144;

                gray = (r + g + b);
                return_array[i * width + j] = gray;
        }
    }
}


// Gaussian blur operation
// Input:
// img --> image that is going to blurred
// kernel_size --> kernel size for filter, filter[kernel_size][kernel_size] is created.
// height --> image height
// width --> image width
// Output: return_array --> blurred_img
void gaussian_blur(double *img, int kernel_size, int height, int width, double *return_array){
    double r, g, p, s, e;
    double sum = 0;
    double gaussian_kernel[kernel_size][kernel_size];
    double flattened_kernel[kernel_size * kernel_size];


    // Computation of gaussian filter
    for(int x = 0; x < kernel_size; x++){
        for(int y = 0; y < kernel_size; y++){
            r = std::pow(x, 2) + std::pow(y, 2);
            s = 2 * std::pow(SIGMA, 2);
            s *= PI;
            p = -(r / s);
            e = std::exp(p);
            g = e / s;
            gaussian_kernel[x][y] = g;
            sum += g;
        }
    }

    for(int i = 0; i < kernel_size; i++){
        for(int j = 0; j < kernel_size; j++){
            gaussian_kernel[i][j] /= sum;
        }
    }

    // Assigning original pixel values to newly created frame
    for(int i = 0; i < height; i++){
        for(int j = 0; j < width; j++){
                return_array[i * width + j] = img[i * width + j];
        }
    }

    // Convolution operation with DASH_ZIP
    // There are 2 other methods at the end of the file that utilizes DASH_GEMM.
    int index = 0;
    for(int i = 0; i < kernel_size; i++){
        for(int j = 0; j < kernel_size; j++){
            flattened_kernel[index++] = gaussian_kernel[i][j];
        }
    }
    int x, y;
    double temp_img_holder[kernel_size * kernel_size];
    double temp_out_holder[kernel_size * kernel_size];

    for(int i = 0; i < height; i++){
        for(int j = 0; j < width ; j++){
            index = 0;
            for(int k = 0; k < kernel_size; k++){
                for(int m = 0; m < kernel_size; m++){
                    x = i + k;
                    y = j + m ;
                    temp_img_holder[index++] = img[x * width + y];
                }
            }
            DASH_ZIP(temp_img_holder, flattened_kernel, temp_out_holder, kernel_size * kernel_size, ZIP_MULT);
            double sum = 0;
            for(int s = 0; s < kernel_size * kernel_size; s++){
                sum += temp_out_holder[s];
            }
            return_array[i * width + j] = sum;
        }
    }
    
}

// Sobel filter operation
// Input:
// img --> image that is going to processed by sobel filter
// filter --> sobel filter 
// kernel_size --> kernel size for filter, filter[kernel_size][kernel_size] is created.
// height --> image height
// width --> image width
// Output: return_array --> filtered_img
void sobel_g_operator(double *img, double *filter, int kernel_size, int height, int width, double *return_array){
    
    // Declaring some variables
    double temp_img_holder[kernel_size * kernel_size];
    double temp_out_holder[kernel_size * kernel_size];
    double acc = 0;
    int x, y;
    int index = 0;

    // Convolution operation utilizing DASH_ZIP
    for(int i = 0; i < height; i++){
        for(int j = 0; j < width ; j++){
            index = 0;
            for(int k = 0; k < kernel_size; k++){
                for(int m = 0; m < kernel_size; m++){
                    x = i + k;
                    y = j + m ;
                    temp_img_holder[index++] = img[x * width + y];
                }
            }
            DASH_ZIP(temp_img_holder, filter, temp_out_holder, kernel_size * kernel_size, ZIP_MULT);
            double sum = 0;
            for(int s = 0; s < kernel_size * kernel_size; s++){
                sum += temp_out_holder[s];
            }
            return_array[i * width + j] = sum;
        }
    }
}

// Sobel filter operation
// Input:
// gx --> previously computed gx frame
// gy --> previously computed gy frame
// height --> image height
// width --> image width
// lower_threshold --> lower bound of omitting frames
// upper_threshold --> upper bound of omitting frames
// Output: return_array --> edge detected frame
void sobel(double *gx, double *gy, int height, int width, int lower_threshold, int upper_threshold, double *return_array){

    // Computing the resultant sobel frame using generated gx and gy frames. Equation is taken from the original method.
    for(int i = 0; i < height; i++){
        for(int j = 0; j < width; j++){
            return_array[i * width + j] = std::sqrt(std::pow(gx[i * width + j], 2) + std::pow(gy[i * width + j], 2));
        }
    }

    double *angle = (double *) malloc (sizeof (double) * (height * width));
    // Computing angles to decide which pixels are going to be omitted later in the next step.
    for(int i = 0; i < height; i++){
        for(int j = 0; j < width; j++){
            angle[i * width + j] = std::atan2(gy[i * width + j], gx[i * width + j]) * 180 / PI;
            if(angle[i * width + j] < 0) angle[i * width + j] += 180;
        }
    }

    // Checking nearby frames according to the associated angle, omitting if necessary.
    for(int i = 0; i < height; i++){
        for(int j = 0; j < width; j++){
            if(i == 0 || j == 0 || i == height - 1 || j == width - 1){
                return_array[i * width + j] = 0;
            }
            else{
                if(angle[i * width + j] == 0){
                    if((return_array[i * width + j] < return_array[i * width + j - 1]) || (return_array[i * width + j] < return_array[i * width + j + 1])) 
                        return_array[i * width + j] = 0;
                }
                if(angle[i * width + j] == 90){
                    if((return_array[i * width + j] < return_array[(i - 1) * width + j]) || (return_array[i * width + j] < return_array[(i + 1) * width + j])) 
                        return_array[i * width + j] = 0;
                }
                if(angle[i * width + j] == 45){
                    if((return_array[i * width + j] < return_array[(i - 1) * width + j + 1]) || (return_array[i * width + j] < return_array[(i + 1) * width + j - 1])) 
                        return_array[i * width + j] = 0;
                }
                if(angle[i * width + j] == 135){
                    if((return_array[i * width + j] < return_array[(i - 1) * width +  j - 1]) || (return_array[i * width + j] < return_array[(i + 1) * width + j + 1])) 
                        return_array[i * width + j] = 0;
                }
            }
                
                
            if(return_array[i * width + j] < lower_threshold)  return_array[i * width + j] = 0;
            else if (return_array[i * width + j] > lower_threshold && return_array[i * width + j] < upper_threshold){
                if((return_array[(i - 1) * width + j] > upper_threshold) || (return_array[(i + 1) * width + j] > upper_threshold) || 
                    (return_array[i * width + j - 1] > upper_threshold) || (return_array[i * width + j + 1] > upper_threshold))continue;
                else return_array[i * width + j] = 0;
            }
        }
    }
}

// Region of interest operation
// Input:
// height --> image height
// width --> image width
// Output: return_array --> edge detected frame
void roi(int height, int width, double *return_array){
    int vertices[4][2] = {{3 * width / 4, 3 * height / 5} ,{width / 4, 3 * height/5} , {40, height}, {width - 40, height}};
    double A, A1, A2, A3;
    double *mask = (double *) malloc (sizeof (double) * (height * width));


    // Computation of mask. Trapezoid shaped mask which covers lanes is calculated.
    // Next few lines compute the trapezoid shaped mask. To achieve this, I firstly created rectangle part of the mask.
    // Then two triangular shape remains. A point creates three triangles with vertices of the triangle. 
    // If point is in the triangle, these three triangle's area must add up to the the original triangle's area.
    for(int i = 0; i < height; i++){
        for(int j = 0; j < width; j++){
                if(i >= vertices[1][1] && i <= vertices[2][1] && j >= vertices[1][0] && j <= vertices[0][0]) mask[i * width + j] = 255;
                else if(i > vertices[1][1] && i < vertices[2][1] && j > vertices[2][0] && j < vertices[1][0]) {
                    A = area(vertices[1][0], vertices[1][1], vertices[2][0], vertices[2][1], vertices[1][0], vertices[2][1]);
                    A1 = area(j, i, vertices[1][0], vertices[1][1], vertices[2][0], vertices[2][1]);
                    A2 = area(j, i, vertices[2][0], vertices[2][1], vertices[1][0], vertices[2][1]);
                    A3 = area(j, i, vertices[1][0], vertices[1][1], vertices[1][0], vertices[2][1]);
                    if(A1 + A2 + A3 == A) mask[i * width + j] = 255;
                }
                else if(i > vertices[0][1] && i < vertices[3][1] && j > vertices[0][0] && j < vertices[3][0]) {
                    A = area(vertices[0][0], vertices[0][1], vertices[3][0], vertices[3][1], vertices[0][0], vertices[2][1]);
                    A1 = area(j, i, vertices[0][0], vertices[0][1], vertices[3][0], vertices[3][1]);
                    A2 = area(j, i, vertices[3][0], vertices[3][1], vertices[0][0], vertices[2][1]);
                    A3 = area(j, i, vertices[0][0], vertices[0][1], vertices[0][0], vertices[2][1]);
                    if(A1 + A2 + A3 == A) mask[i * width + j] = 255;
                }
                else mask[i * width + j] = 0;
        }
    }

    // Original frame is updated according to mask. 
    for(int i = 0; i < height; i++){
        for(int j = 0; j < width; j++){
                if(mask[i * width + j] == 0)return_array[i * width + j] = 0;
        }
    }
}

// Hough transform operation
// Input:
// img --> edge detected frame
// height --> image height
// width --> image width
// rho --> the resolution of the parameter r in pixels*
// theta --> the resolution of the parameter θ in radians*
// threshold --> the minimum number of intersections to "*detect*" a line*
// line_length --> the minimum number of points that can form a line*
// line_gap --> the maximum gap between two points to be considered in the same line.
// Output: return_array --> detected lines in the edge detected frame
// * denotes explanations taken from OpenCV
std::vector<Line> hough_transform(double *img, int height, int width, float rho, float theta, int threshold, int line_length, int line_gap){
    
    // This is probabilistic Hough transform, different from classical Hough transform. 
    // Details can be found on the following paper: https://www.researchgate.net/publication/222317034_Robust_Detection_of_Lines_Using_the_Progressive_Probabilistic_Hough_Transform
    // Paper is not open-access. Hence, I do not know all details about the algorithm.
    // I will explain classical Hough transform and will mention differences between probabilistic and classical implementation.
    // First only pixels that lies on a edge are stored in a different array and only these pixels are included in calculation process.
    // Classical Hough transform moves coordinates from cartesian plane to polar plane.
    // In polar plane rho(r) denotes distance of point from the origin and theta denotes the angle wrt to origin of the line that goes from origin to point.
    // Interval of theta is predefined. Mostly (0, 180) or (-90, 90) is chosen. In this implementation (0, 180) is chosen.
    // To calculate rho following equation is used: (x_coord * cos(theta) + y_coord * sin(theta)) (This operation is done for the whole theta interval)
    // An holder array is created for each (rho, theta) pair. Then, for each occurence of (rho, theta) count is updated with one.
    // Lastly, with finding local maximas lines are computed.

    // Difference between probabilistic and classical Hough transform is that probabilistic version uses less pixels for calculation and more accurate.
    // For more information: https://en.wikipedia.org/wiki/Hough_transform#:~:text=The%20Hough%20transform%20is%20a,shapes%20by%20a%20voting%20procedure.
    // (For probabilistic one, please only refer to above paper. There are some confusing materials on the internet but they do not match the behaviour.)
    std::vector<Line>lines;

    float irho = 1./ rho;
    int numangle = std::round(PI / theta);
    int numrho = std::round(((width + height) * 2 + 1) / rho);

    int * accum = new int[numangle * numrho];
    int * mask = new int[height * width];

    for( int i = 0; i < numangle * numrho; i++){
	    *(accum + i) = 0;
    }

    double cos_thetas[numangle];
    double sin_thetas[numangle];
    for(int t = 0; t < numangle; t++){
        cos_thetas[t] = std::cos(t * (PI /180));
        sin_thetas[t] = std::sin(t * (PI /180));
    }

    std::vector<Point> nzloc;

    for(int y = 0; y < height; y++){
        for(int x = 0; x < width; x++){
            if(img[y * width + x] != 0.0){
                *(mask + y* width + x) = 1;
                Point pt{x, y};
                nzloc.push_back(pt); 
            }else{
                *(mask + y* width + x) = 0;
            }
        }
    }

    int count = (int)nzloc.size();

    for(; count > 0; count--){
        int idx = rand()%(count-0 + 1) + 0;
        int max_val = threshold-1, max_n = 0;
        Point point = nzloc[idx];
        Point p = {0, 0};
        std::vector<Point>line_end;
        line_end.push_back(p);
        line_end.push_back(p);
        float a, b;
        int i = point.y, j = point.x, k, x0, y0, dx0, dy0, xflag;
        int good_line;
        const int shift = 16;

        nzloc[idx] = nzloc[count-1];

        if(*(mask + i * width + j) == 0)continue;

        for( int n = 0; n < numangle; n++){
            int r = std::round(j * cos_thetas[n] + i * sin_thetas[n]);
            r += (numrho - 1) / 2;
            *(accum + (n * numrho + r)) += 1;
            int val = *(accum + (n * numrho + r));
            if( max_val < val )
            {
                max_val = val;
                max_n = n;
            }
        }

        if(max_val < threshold) continue;

        a = -sin_thetas[max_n];
        b = cos_thetas[max_n];
        x0= j;
        y0= i;

        if(fabs(a) > fabs(b)){
            xflag = 1;
            dx0 = a > 0 ? 1 : -1;
            dy0 = std::round( b*(1 << shift)/fabs(a) );
            y0 = (y0 << shift) + (1 << (shift-1));
        }else{
            xflag = 0;
            dy0 = b > 0 ? 1 : -1;
            dx0 = std::round( a*(1 << shift)/fabs(b) );
            x0 = (x0 << shift) + (1 << (shift-1));
        }


        for( k = 0; k < 2; k++ ){
            int gap = 0, x = x0, y = y0, dx = dx0, dy = dy0;

            if( k > 0 )
                dx = -dx, dy = -dy;

            for( ;; x += dx, y += dy )
            {
                int * mdata;
                int i1, j1;

                if( xflag )
                {
                    j1 = x;
                    i1 = y >> shift;
                }
                else
                {
                    j1 = x >> shift;
                    i1 = y;
                }

                if( j1 < 0 || j1 >= width || i1 < 0 || i1 >= height )
                    break;

                mdata = mask + i1*width + j1;

                if( *mdata )
                {
                    gap = 0;
                    line_end[k].y = i1;
                    line_end[k].x = j1;
                }
                else if( ++gap > line_gap )
                    break;
            }
        }

        good_line = std::abs(line_end[1].x - line_end[0].x) >= line_length ||
                    std::abs(line_end[1].y - line_end[0].y) >= line_length;


        for( k = 0; k < 2; k++ )
        {
            int x = x0, y = y0, dx = dx0, dy = dy0;

            if( k > 0 )
                dx = -dx, dy = -dy;

            for( ;; x += dx, y += dy )
            {
                int* mdata;
                int i1, j1;

                if( xflag )
                {
                    j1 = x;
                    i1 = y >> shift;
                }
                else
                {
                    j1 = x >> shift;
                    i1 = y;
                }

                
                if( j1 < 0 || j1 >= width || i1 < 0 || i1 >= height )
                    break;

                mdata = mask + i1*width + j1;

                if( *mdata )
                {
                    if( good_line )
                    {
                        for( int n = 0; n < numangle; n++)
                        {
                            int r = std::round( j1 * cos_thetas[n] + i1 * sin_thetas[n]);
                            r += (numrho - 1) / 2;
                            *(accum + (n * numrho + r)) -= 1;
                        }
                    }
                    *mdata = 0;
                }

                if( i1 == line_end[k].y && j1 == line_end[k].x )
                    break;
            }
        }
        
        if(good_line)
        {
            Line l{line_end[0].x, line_end[0].y, line_end[1].x, line_end[1].y};
            lines.push_back(l);
        }
        
    }
    return lines;
}

//Helper for line
void seg_intersect(int* A, int *B, int *C, int *D, int *return_array){
    // Line AB
    int a1 = B[1] - A[1];
    int b1 = A[0] - B[0];
    int c1 = a1 * (A[0]) + b1 * (A[1]);

    // Line CD
    int a2 = D[1] - C[1];
    int b2 = C[0] - D[0];
    int c2 = a2 * (C[0]) + b2 * (C[1]);

    int determinant = a1 *b2 - a2 * b1;
    int x,y;
    if(determinant == 0){
        return_array[0] = -1;
        return_array[1] = -1;
    }else{
        x = (b2 * c1 - b1 * c2) / determinant;
        y = (a1 * c2 - a2 * c1) / determinant;
        return_array[0] = x;
        return_array[1] = y;
    }

}

// Method to draw a circle around a center point
// Input:
// x1, y1, x2, y2 --> coordinates of points that line goes through
// width --> image width
// dimension --> image dimension(1 --> grayscale, 3 --> RGB)
// color --> fill color(0 --> blue, 1 --> green, 2 --> red)
// Output: return_array --> line drawn image
void draw_line(int x1, int y1, int x2, int y2, int width, int dimension, int color, double *return_array){
    int dx = x2 - x1;
    int dy = y2 - y1;
    int yi = 1;
    if(dy < 0){
        yi = -1;
        dy = -dy;
    }

    int D = 2 * dy - dx;
    int y = y1;

            
    for(int x = x1; x < x2; x++){
    if(dimension == 3){
        if(color == 0){
            return_array[(y * width + x) * 3 + 0] = 255;
            return_array[(y * width + x) * 3 + 1] = 0;
            return_array[(y * width + x) * 3 + 2] = 0;

        }else if(color == 1){
            return_array[(y * width + x) * 3 + 0] = 0;
            return_array[(y * width + x) * 3 + 1] = 255;
            return_array[(y * width + x) * 3 + 2] = 0;
        }else if(color == 2){

            return_array[(y * width + x) * 3 + 0] = 0;
            return_array[(y * width + x) * 3 + 1] = 0;
            return_array[(y * width + x) * 3 + 2] = 255;
        }
        
    }else{
        return_array[y * width + x] = 255;
    }
    if(D > 0){
        y = y + yi;
        D = D + (2 * (dy -dx));
    }else{
        D = D + 2 * dy;
        }
    }
}

// Method to draw a circle around a center point
// Input:
// center --> center point coordinates(center[0] --> x, center[1] -->y)
// r --> radius of the circle
// color --> fill color(0 --> blue, 1 --> green, 2 --> red)
// height --> image height
// width --> image width
// depth --> image depth(color channel)
// Output: return_array --> circle drawn image
void draw_circle(int *center, int r, int color, int height, int width, int depth, double *return_array){
    int r_sq = r * r;
    int x_diff, y_diff;
    for(int y = 0; y < height; y++){
        for(int x = 0; x < width; x++){
            x_diff = x - center[0];
            y_diff = y - center[1];
            if((x_diff * x_diff + y_diff * y_diff) <= (r_sq)){
                if(color == 0){
                    return_array[(y * width + x) * depth + 0] = 255;
                    return_array[(y * width + x) * depth + 1] = 0;
                    return_array[(y * width + x) * depth + 2] = 0;
                }else if(color == 1){
                    return_array[(y * width + x) * depth + 0] = 0;
                    return_array[(y * width + x) * depth + 1] = 255;
                    return_array[(y * width + x) * depth + 2] = 0;
                }else if(color == 2){
                    return_array[(y * width + x) * depth + 0] = 0;
                    return_array[(y * width + x) * depth + 1] = 0;
                    return_array[(y * width + x) * depth + 2] = 255;
                }
            }
            
        }
    }
}

// Helper method for drawing line
// Input:
// avg --> current average
// sample --> current sample
// Output: avg --> new average
int moving_avg(int avg, int sample){
    int n = 20;
    if(avg == 0)return sample;
    avg -= avg / n;
    avg += sample / n;
    return avg;
}

// Drawing lanes on a black image
// Input:
// lines --> previously detected lines
// width --> image width
// Output: return_array --> frame with detected lanes
void draw_lane(std::vector<Line> lines, int height, int width, double * return_array, int *avg_left_return, int *avg_right_return){
    int avg_left[4] = {0, 0, 0, 0};
    int avg_right[4] = {0, 0, 0, 0};
    int y;
    int largest_left_line_size = 0,largest_right_line_size = 0;
    int largest_left_line[4] = {0, 0, 0, 0};
    int largest_right_line[4] = {0, 0, 0, 0};

    double slope;
    for(auto item: lines){
        int x1 = item.x1, y1 = item.y1, x2 = item.x2, y2 = item.y2;
        int size = std::hypot(item.x2 - item.x1, item.y2 - item.y1);
        if(x2 != x1) slope = ((float(y2)-y1)/(float(x2)-x1));
        else continue;

        if(slope > 0.5){
            if(size > largest_right_line_size) {
                largest_right_line[0] = x1;
                largest_right_line[1] = y1;
                largest_right_line[2] = x2;
                largest_right_line[3] = y2;
            }
            draw_line(x1, y1, x2, y2, width, 1, -1, return_array);
        }
        else if(slope < -0.5){
            if(size > largest_left_line_size) {
                largest_left_line[0] = x1;
                largest_left_line[1] = y1;
                largest_left_line[2] = x2;
                largest_left_line[3] = y2;
            }
            draw_line(x1, y1, x2, y2, width, 1, -1, return_array);
        }
    }

    int up_line_point1[2] = {0, (int) ((2.0/3.0) * height)};
    int up_line_point2[2] = {width, (int) ((2.0/3.0) * height)};
    int down_line_point1[2] = {0, height};
    int down_line_point2[2] = {width, height};

    int p3[2] = {largest_left_line[0], largest_left_line[1]};
    int p4[2] = {largest_left_line[2], largest_left_line[3]};

    int up_left_point[2];
    int down_left_point[2];
    seg_intersect(up_line_point1, up_line_point2, p3, p4, up_left_point);
    seg_intersect(down_line_point1, down_line_point2, p3, p4, down_left_point);


    if(up_left_point[0] == -1 || down_left_point[0] == -1){
        draw_line(avg_left[0], avg_left[1], avg_left[2], avg_left[3], width, 1, -1,  return_array);
        draw_line(avg_right[0], avg_right[1], avg_right[2], avg_right[3], width, 1, -1, return_array);

    }

    avg_left[0] = moving_avg(avg_left[0], up_left_point[0]);
    avg_left[1] = moving_avg(avg_left[1], up_left_point[1]);
    avg_left[2] = moving_avg(avg_left[2], down_left_point[0]);
    avg_left[3] = moving_avg(avg_left[3], down_left_point[1]);
    draw_line(avg_left[0], avg_left[1], avg_left[2], avg_left[3], width, 1, -1, return_array);

    int p5[2] = {largest_right_line[0], largest_right_line[1]};
    int p6[2] = {largest_right_line[2], largest_right_line[3]};
    int up_right_point[2];
    int down_right_point[2];
    seg_intersect(up_line_point1, up_line_point2, p5, p6, up_right_point);
    seg_intersect(down_line_point1, down_line_point2, p5, p6, down_right_point);

    avg_right[0] = moving_avg(avg_right[0], up_right_point[0]);
    avg_right[1] = moving_avg(avg_right[1], up_right_point[1]);
    avg_right[2] = moving_avg(avg_right[2], down_right_point[0]);
    avg_right[3] = moving_avg(avg_right[3], down_right_point[1]);
    draw_line(avg_right[0], avg_right[1], avg_right[2], avg_right[3], width, 1, -1, return_array);

    for(int i = 0; i < 4; i++){
        avg_left_return[i] = avg_left[i];
        avg_right_return[i] = avg_right[i];
    }

}

// Saving image to output
// Input:
// img --> image that is going to be saved
// name --> file name for saved image
void save_img(double*img, int height, int width, std::string name){
    //std::string file_name = name + ".txt";
    std::string file_name = name + ".png";
    //std::ofstream file;
    //file.open(file_name);

    //std::cout << std::setprecision(5) << f << '\n';


    //for(int i = 0; i < height; i++){
    //    for(int j = 0; j < width; j++){
    //        file << (int) img[i * width + j] << "\n";
    //    }
    //}

    //file.close();
    uint8_t* out_img = (uint8_t *) malloc (sizeof (uint8_t) * (height * width));
    for (int i = 0; i<(height * width); i++){
	out_img[i] = (uint8_t)img[i];
    }
    stbi_write_png(file_name.c_str(), width, height, 1, out_img, width);
    free(out_img);
}

int main(){
    // Reading image to a 3D array
    // ---------------------------
    //std::fstream f("image.txt", std::ios_base::in);

    // Next few lines reads height, width and depth values of the input frame.
    int element;
    int index = 0;
    //f >> element;
    int height;// = element;
    //f >> element;
    int width;// = element;
    //f >> element;
    int depth;// = element;

    char imagepath[100] = "image.png";
    uint8_t* rgb_imag = stbi_load(imagepath, &width, &height, &depth, 3);
    if (rgb_imag == NULL){
	std::cout << "Failed to open image " << imagepath << std::endl;
        return -1;
    }
    int size = height * width;


    // Allocating memory for frame holder arrays.
    double *temp_arr = (double *) malloc (sizeof (double) * (size * 3));
    double *img = (double *) malloc (sizeof (double) * (size * 3));
    double *grayscale_img = (double *) malloc (sizeof (double) * size);
    double *blurred_img = (double *) malloc (sizeof (double) * size);
    double *blurred_img_before_canny = (double *) malloc (sizeof (double) * size);
    double *gx_canny = (double *) malloc (sizeof (double) * size);
    double *gy_canny = (double *) malloc (sizeof (double) * size);
    double *sobel_img = (double *) malloc (sizeof (double) * size);
    double *lanes = (double *) malloc (sizeof (double) * size);


    // Putting frame pixel values to the holder array.
/*    while(f >> element){
        temp_arr[index++] = element;
    }

    index = 3;
    for(int i = 0; i < height; i++){
        for(int j = 0; j < width; j++){
            for(int k = 0; k < depth; k++){
                img[(i * width + j) * depth + k] = temp_arr[index++];
            }
        }
    }
*/
    for (int i = 0; i<size*3; i++){
	img[i] = (double) rgb_imag[i];
    }

    // Convertion to grayscale. [HEIGHT][WIDTH][DEPTH] --> [HEIGHT][DEPTH]
    // Doing in main to only deal with 2 dimensional frame from now on.
    // Flattening 2D frame to 1D array for better handling with methods.
    grayscale_conversion(img, height, width, depth, grayscale_img);

    save_img(grayscale_img, height, width, "grayscale_img");

    // First gaussian blur operation. Kernel size is chosen as 11. 
    gaussian_blur(grayscale_img, 11, height, width, blurred_img);

    save_img(blurred_img, height, width, "blurred_img");

    // Second gaussian blur operation. Kernel size is chosen as 5.
    // Applying two consecutive gaussian blur looks a bit odd but this is a requirement of canny edge detection algorithm.
    gaussian_blur(blurred_img, 5, height, width, blurred_img_before_canny);

    save_img(blurred_img_before_canny, height, width, "blurred_img_before_canny");
    
    // First operation of sobel. Detects lines in the x direction.
    double flattened_gx[3 * 3];
    double gx[3][3] = {{-1.0, 0.0, 1.0},{-2.0, 0.0, 2.0},{-1.0, 0.0, 1.0}};
    index = 0;
    for(int i = 0; i < 3; i++){
        for(int j = 0; j < 3; j++){
            flattened_gx[index++] = gx[i][j];
        }
    }

    sobel_g_operator(blurred_img_before_canny, flattened_gx, 3, height, width, gx_canny);

    save_img(gx_canny, height, width, "gx_canny");

    // Second operation of sobel. Detects lines in the y direction.
    double flattened_gy[3 * 3];
    double gy[3][3] = {{1.0, 2.0, 1.0},{0.0, 0.0, 0.0},{-1.0, -2.0, -1.0}};
    index = 0;
    for(int i = 0; i < 3; i++){
        for(int j = 0; j < 3; j++){
            flattened_gy[index++] = gy[i][j];
        }
    }

    sobel_g_operator(blurred_img_before_canny, flattened_gy, 3, height, width, gy_canny);
    
    save_img(gy_canny, height, width, "gy_canny");

    // Merge of gx and gy frames to generate edge marked frame.
    int lower_threshold = 40;
    int upper_threshold = 50;
    sobel(gx_canny, gy_canny, height, width, 40, 50, sobel_img);

    save_img(sobel_img, height, width, "sobel_img");

    roi(height, width, sobel_img);

    save_img(sobel_img, height, width, "roi");

    // Declaring some variables for hough transform
    float rho = 1; 
    float theta = PI/180;
    int threshold = 30;
    int line_length = 30;
    int line_gap = 200;

    // Hough transform operation.
    std::vector<Line> lines = hough_transform(sobel_img, height, width, rho, theta, threshold, line_length, line_gap);

    //printf("Lines: %d\n", lines.size());

    int avg_left[4];
    int avg_right[4];

    // Drawing lanes
    draw_lane(lines, height, width, lanes, avg_left, avg_right);

    save_img(lanes, height, width, "lanes");

    int reference_line_1[2] = {0, (int)(0.8 * height)};
    int reference_line_2[2] = {width, (int)(0.8 * height)};
    int p1[2];
    int p2[2];
    int p3[2] = {0, reference_line_2[1]};
    int p4[2] = {width, reference_line_2[1]};

    int upper_point[2] = {avg_left[0], avg_left[1]};
    int lower_point[2] = {avg_left[2], avg_left[3]};
    seg_intersect(upper_point, lower_point, reference_line_1, reference_line_2, p1);

    upper_point[0] = avg_right[0]; upper_point[1] = avg_right[1];
    lower_point[0] = avg_right[2]; lower_point[1] = avg_right[3];
    seg_intersect(upper_point, lower_point, reference_line_1, reference_line_2, p2);

    int car_point[2] = {((p3[0] + p4[0]) / 2), ((p3[1] + p4[1]) / 2)};
    int lane_point[2] = {((p1[0] + p2[0]) / 2), ((p1[1] + p2[1]) / 2)};

    for(int m = -5; m < 5; m++){
        draw_line(p1[0], p1[1] + m, p2[0], p2[1] + m, width, 3, 2, img);
    }

    draw_circle(car_point, 15, 0, height, width, depth, img);
    draw_circle(lane_point, 10, 1, height, width, depth, img);   

    for (int i = 0; i<(height * width * 3); i++){
        rgb_imag[i] = (uint8_t)img[i];
    }

    stbi_write_png("redline.png", width, height, 3, rgb_imag, width*3);
/*
    std::ofstream file;
    file.open("redline.txt");

    for(int i = 0; i < height; i++){
        for(int j = 0; j < width; j++){
            for(int d = 0; d < 3; d++){
                file << img[(i * width + j) * depth + d] << "\n";
            }
        }
    }

    file.close();
*/
    return 0;
}






// ALTERNATIVES FOR DASH_ZIP METHOD
// Second way of doing this operation with DASH_GEMM
    /*int index = 0;
    for(int i = 0; i < kernel_size; i++){
        for(int j = 0; j < kernel_size; j++){
            flattened_kernel[index++] = gaussian_kernel[j][i];
        }
    }

    double acc = 0;
    int x, y;
    double temp_img_holder[kernel_size * kernel_size];
    double temp_out_holder[kernel_size * kernel_size];
    double temp_img_holder_im[kernel_size * kernel_size];
    double temp_out_holder_im[kernel_size * kernel_size];
    double flattened_kernel_im[kernel_size * kernel_size];

    for(int i = 0; i < kernel_size * kernel_size; i++){
        temp_img_holder_im[i] = 0;
        temp_out_holder_im[i] = 0;
        flattened_kernel_im[i] = 0;
    }

    for(int i = 0; i < height; i++){
        for(int j = 0; j < width ; j++){
            acc = 0;
            index = 0;
            for(int k = 0; k < kernel_size; k++){
                for(int m = 0; m < kernel_size; m++){
                    x = i + k;
                    y = j + m ;
                    temp_img_holder[index++] = img[x * width + y];
                }
            }
            DASH_GEMM(temp_img_holder, temp_img_holder_im, flattened_kernel, flattened_kernel_im, 
                        temp_out_holder, temp_out_holder_im, kernel_size, kernel_size, kernel_size);
            
            double sum = temp_out_holder[0];
            for(int s = 1; s < kernel_size; s++){
                sum += temp_out_holder[s * kernel_size + s];
            }
            return_array[i * width + j] = sum;

        }
    }*/

    // Third way of doing this operation with DASH_GEMM
    /*int index = 0;
    double acc = 0;
    int x, y;
    double flattened_kernel_s[kernel_size];
    double temp_img_holder[kernel_size];
    double temp_out_holder[1];
    double temp_img_holder_im[kernel_size];
    double temp_out_holder_im[1];
    double flattened_kernel_im[kernel_size];

    for(int i = 0; i < kernel_size; i++){
        temp_img_holder_im[i] = 0;
        flattened_kernel_im[i] = 0;
    }

    int count = 0;
    for(int i = 0; i < height; i++){
        for(int j = 0; j < width ; j++){
            acc = 0;
            sum = 0;
            for(int k = 0; k < kernel_size; k++){
                index = 0;
                for(int m = 0; m < kernel_size; m++){
                    x = i + k;
                    y = j + m ;
                    acc += img[x * width + y] * gaussian_kernel[k][m];
                    flattened_kernel_s[index++] = gaussian_kernel[k][m];
                    temp_img_holder[--index] = img[x * width + y];
                    index++;
                    DASH_GEMM(temp_img_holder, temp_img_holder_im, flattened_kernel_s, flattened_kernel_im, 
                        temp_out_holder, temp_out_holder_im, 1, kernel_size, 1);
                    count ++;
                }
                sum += temp_out_holder[0];
            }            
            return_array[i * width + j] = sum;
        }
    }*/
