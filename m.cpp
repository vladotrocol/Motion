#include "opencv2/opencv.hpp"
#include <iostream>
#include <stdio.h>

using namespace cv;
using namespace std;

float** kx;
float** ky;

float** padd_border(float** src, int h, int w){
        //init the resulting matrix with 0's floats
       float** r = new float*[h+2];
        for(int i = 0; i < h+2; ++i){
            r[i] = new float[w+2];
         }

        //Copy the corners of the image
        r[0][0] = src[0][0]; //north-west
        r[0][w+1] = src[0][w-1]; //north-east
        r[h+1][0] = src[h-1][0]; //south-west
        r[h+1][w+1] = src[h-1][w-1]; //south-east

        //Horizontal borders
        for(int it=0;it<w;it++){
                r[0][it+1] = src[0][it]; //north border
                r[h+1][it+1] = src[h-1][it]; //south border
        }

        //Vertical borders
        for(int it=0;it<h;it++){
                r[it+1][0] = src[it][0]; //east border
                r[it+1][w+1] = src[it][w-1]; //west border
        }

        //Fill centre of matrix
        for(int i=0;i<h;i++){
                for(int j=0;j<w;j++){
                        r[i+1][j+1] = src[i][j];
                }
        }

        return r;
};

//Convolve a matrix with a kernel
float** convolution(float** src, float** ker, int height, int width){

        //Padding the border
        float** psrc = padd_border(src, height, width);
        
        //Height and width after padding
        unsigned int h = psrc->rows;
        unsigned int w = psrc->cols;

        //Kernel radius (used as offsets for parsing image block by block)
        unsigned int rkh = (unsigned int)((ker->rows-1)*0.5f); //horizontal
        unsigned int rkw = (unsigned int)((ker->cols-1)*0.5f); //vertical

        //Resulting matrix
        Mat* r = new Mat( Mat::zeros(h, w, CV_8U) );
        
        int sum; //Total convolved sum for each pixel
        
        //needed for shifting and normalizing
        int max = -32766; 
        int min = 32766;

        //for all values in source
        for(unsigned int i=1;i<h-rkh-1;i++){
                for(unsigned int j=1;j<w-rkw-1;j++){
                        sum=0.0; //reset for each pixels/block
                        //each kernel-sized block
                        for(unsigned int k=i-rkh;k<i+rkh+1;k++){
                                for(unsigned int l=j-rkw;l<j+rkw+1;l++){
                                        sum += (float)psrc->at<uchar>(k, l)*ker->at<float>(k-i+rkh, l-j+rkw);
                                }
                        }
                        //Compute final pixel value
                        r->at<uchar>(i,j) = sum;
                        
                        //find min&max
                        if(sum>max){
                                max = sum;
                        }
                        else if(sum<min){
                                min = sum;
                        }
                }
        }
        //Return the normalized result
        return normalize(shift(r, min), max, min);
};


void printMat(float** src, int h, int w){
    for(int i=0;i<h;i++){
        for(int j=0;j<w;j++){
            printf("%f ", src[i][j]);
        }
        printf("\n");
    }
    printf("\n\n");
};


void init_kernel_y(){
    ky = new float*[3];
    for(int i = 0; i < 3; ++i){
        ky[i] = new float[3];
    }
    ky[0][0] = 1.0f;
    ky[0][1] = 2.0f;
    ky[0][2] = 1.0f;

    ky[1][0] = 0.0f;
    ky[1][1] = 0.0f;
    ky[1][2] = 0.0f;

    ky[2][0] = -1.0f;
    ky[2][1] = -2.0f;
    ky[2][2] = -1.0f;

};

void init_kernel_x(){
    kx = new float*[3];
    for(int i = 0; i < 3; ++i){
        kx[i] = new float[3];
    }

    kx[0][0] = 1.0f;
    kx[0][1] = 0.0f;
    kx[0][2] = -1.0f;

    kx[1][0] = 2.0f;
    kx[1][1] = 0.0f;
    kx[1][2] = -2.0f;

    kx[2][0] = 1.0f;
    kx[2][1] = 0.0f;
    kx[2][2] = -1.0f;
};

int main(int, char**)
{
    init_kernel_x();
    init_kernel_y();
    printMat(padd_border(kx, 3, 3), 5, 5);
    printMat(padd_border(ky, 3, 3), 5, 5);
    // VideoCapture cap(0); // open the default camera
    // if(!cap.isOpened())  // check if we succeeded
    //     return -1;
    // namedWindow("Camera",1);
    // for(;;)
    // {
    //     Mat frame;
    //     cap >> frame; // get a new frame from camera
    //     imshow("Camera", frame);
    //     if(waitKey(30) >= 0) break;
    // }
    // the camera will be deinitialized automatically in VideoCapture destructor
    return 0;
}