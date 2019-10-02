/*
 * Apriltags - matlab wrapper / Michael Pantic, 2016, mpantic@student.ethz.ch
 *
 * Usage: Compile in matlab using mex:
 *
 * mex mex_apriltags.cpp -I/usr/include/eigen3 -Iethz_apriltag2/include -L/usr/local/lib/ -Lethz_apriltag2/build/devel/lib/ -lopencv_core.so -lopencv_imgproc.so -lopencv_highgui.so -lethz_apriltag2.so
 * Note: Paths should be customized:
 *          * /usr/include/eigen3 to the include dir of eigen3
 *
 *  Once compiled, use in the following way:
 *      mex_apriltags('/home/mpantic/GIT/3dv/mex/apriltags.bmp','25h9')
 *
 *      - First param is image name (currently only .bmp works..whatever)
 *      - Second param is tag family
 *
 *  Outputs one matrix with one row per detected tag:
 *        [ x_pixel, y_pixel, tag_id, tag_obsCode, tag_code, tag_hammingDistance]
 *
 *
 *  And writes a "debug.bmp" in the current folder, drawing the detected tags
 *  with their tag_id.
 *
 *
 * Example:
 *  A=mex_apriltags('/home/mpantic/GIT/3dv/mex/Apriltagrobots_overlay.bmp','36h11');
 *
 * Compile:
 *  mex mex_apriltags.cpp -Iethz_apriltag2/include/ 
 */

#include "mex.h"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "apriltags/TagDetector.h"
#include "apriltags/Tag16h5.h"
#include "apriltags/Tag25h7.h"
#include "apriltags/Tag25h9.h"
#include "apriltags/Tag36h9.h"
#include "apriltags/Tag36h11.h"



void mexFunction(int nlhs, mxArray *plhs[], int nrhs,
     const mxArray *prhs[])
{
  // num of right hand / left hand params
  nrhs = 2; // 1 right hand (file name) tag_name
  nlhs = 1; // 2 left hand (array with x/y/tag id),
  const mxArray* imgArray = prhs[0];
  //char* imgData = (char*)mxGetData(imgArray);
  std::string tag_family = mxArrayToString(prhs[1]);


  // parse input settings (tagfamily)
  AprilTags::TagCodes m_tagCodes(AprilTags::tagCodes36h11);

  if (tag_family=="16h5") {
      m_tagCodes = AprilTags::tagCodes16h5;
    } else if (tag_family=="25h7") {
      m_tagCodes = AprilTags::tagCodes25h7;
    } else if (tag_family=="25h9") {
      m_tagCodes = AprilTags::tagCodes25h9;
    } else if (tag_family=="36h9") {
      m_tagCodes = AprilTags::tagCodes36h9;
    } else if (tag_family=="36h11") {
      m_tagCodes = AprilTags::tagCodes36h11;
    } else {
      // set error code to 100, exit
      mexPrintf("Tag family not recognized (Possible Vals: 16h5/25h7/25h9/36h9/36h11\n");
      return;
    }

  // set output argument to used tag family
  plhs[1] = mxCreateString(tag_family.c_str());

  // read image and convert to grayscale
  cv::Mat image(mxGetM(imgArray), mxGetN(imgArray),CV_8U,mxGetData(imgArray));
  //cv::Mat image_gray;
//   // imgData
//   image = cv::imread(file_name.c_str());
//   cv::cvtColor(image, image_gray, CV_BGR2GRAY);
//   if(!image.data)
//   {
//     // in case of error, return code 101
//     mexPrintf("Could not load file %s. Only bmp working at the moment!\n",file_name.c_str());
//     return;
//  }



  //set up tagdetector
   std::vector<AprilTags::TagDetection> detections;
  try{
    AprilTags::TagDetector m_tagDetector(m_tagCodes);

     // get tags
     detections = m_tagDetector.extractTags(image);
     mexPrintf("Detected %i tags.\n", detections.size());
  }
  catch(...)
  {
      mexPrintf("Something went wrong in apriltags. Restart matlab :-( \n");
  }
  //convert to matlab array
  int cols = 6;
  int rows = detections.size();

  plhs[0] = mxCreateDoubleMatrix(rows,cols,mxREAL);
  double *mxData = mxGetPr(plhs[0]);

  for(int i=0; i<rows; i++)
  {
    *(mxData+0*rows+i) = detections[i].cxy.first;
    *(mxData+1*rows+i) = detections[i].cxy.second;
    *(mxData+2*rows+i) = detections[i].id;
    *(mxData+3*rows+i) = detections[i].obsCode;
    *(mxData+4*rows+i) = detections[i].code;
    *(mxData+5*rows+i) = detections[i].hammingDistance;
    detections[i].draw(image);


  }

   //write debug output
   //cv::imwrite("debug.bmp", image);

}

