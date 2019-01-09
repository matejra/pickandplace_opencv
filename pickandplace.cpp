#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <cmath>
#include <iostream>

using namespace cv;
using namespace std;

// Constants for image processing
int sigma = 1.2;
int ksize = 5;
int erosion_size = 2; 
double min_contour_area = 220;
double max_contour_area = 400;
float center_proximity = 7.0;
double contour_area = 0;
double holes_threshold = 100;
double object_threshold = 190;
int max_rectangle_index = 0;
bool working_table_limits_found = 0;
int n_expected_objects = 100;
int surrounding_rectangle_size = 40;
int sum_current_rectangle = 0;
int avg_current_rectangle = 0;
int lowThreshold;
int const max_lowThreshold = 200;
int ratio = 3;
int kernel_size = 3;
char* window_name = "Edge Map";
static const std::string OPENCV_WINDOW = "Color image";
Rect working_table, bounding_patch;
Mat im_working_table_gray, image, im_orig, im_gray, im_bw_canny, im_gray_edges, dst, im_bw_holes_, im_bw_canny_;
;
vector<Mat> im_objectholepatch;

static double angle(cv::Point pt1, cv::Point pt2, cv::Point pt0)
{
	double dx1 = pt1.x - pt0.x;
	double dy1 = pt1.y - pt0.y;
	double dx2 = pt2.x - pt0.x;
	double dy2 = pt2.y - pt0.y;
	return (dx1*dx2 + dy1*dy2)/sqrt((dx1*dx1 + dy1*dy1)*(dx2*dx2 + dy2*dy2) + 1e-10);
}

void setLabel(Mat& im, const string label, vector<Point>& contour)
{
	int fontface = FONT_HERSHEY_SIMPLEX;
	double scale = 0.4;
	int thickness = 1;
	int baseline = 0;

	Size text = getTextSize(label, fontface, scale, thickness, &baseline);
	Rect r = boundingRect(contour);

	Point pt(r.x + ((r.width - text.width) / 2), r.y + ((r.height + text.height) / 2));
	rectangle(im, pt + Point(0, baseline), pt + Point(text.width, -text.height), CV_RGB(255,255,255), CV_FILLED);
	putText(im, label, pt, fontface, scale, CV_RGB(0,0,0), thickness, 8);
}

void CannyThreshold(int, void*)
{
  /// Reduce noise with a kernel 3x3
  blur( im_gray, im_bw_canny_, Size(3,3) );

  /// Canny detector
  Canny( im_bw_canny_, im_bw_canny_, lowThreshold, lowThreshold*ratio, kernel_size );

  /// Using Canny's output as a mask, we display our result
  dst = Scalar::all(0);

  im_gray.copyTo( dst, im_bw_canny_);
  imshow( window_name, dst );
 }


int main(int argc, char** argv)
{
 ///////// Begin image processing /////////
    char* imageName = argv[1];
    image = imread( imageName, 1 );
    im_orig = image;
    cvtColor(image, im_gray, COLOR_BGR2GRAY);
    Mat channel[3];
    split(image, channel);
    Mat bw, im_blue;
    im_blue = channel[0];
    GaussianBlur(im_blue, im_blue, Size(ksize, ksize), sigma, sigma);
	  Canny(im_gray, im_bw_canny, 100, 300, 5);
    GaussianBlur(im_gray, im_gray, Size(ksize, ksize), sigma, sigma);
    
    /*Ptr<CLAHE> clahe = createCLAHE();
    clahe->setClipLimit(2);
    clahe->setTilesGridSize(Size(20,20));

    clahe->apply(im_gray,im_gray);
    imshow("Clahe",im_gray);

    equalizeHist( im_gray, im_gray );*/

    

    ///////// Getting working table limits
    // shape detection based on https://github.com/bsdnoobz/opencv-code/blob/master/shape-detect.cpp
    // should be done just once - to find the table
    vector< vector <Point> > contours_table;
    vector<Point> approx;
    double largest_rectangle_area = 0;


    if (working_table_limits_found == 0)
    {
      findContours(im_bw_canny, contours_table, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE); // Find the rectangles in the image
      for (int i = 0; i < contours_table.size(); i ++)                         // iterate through each contour.
      { 
        approxPolyDP(Mat(contours_table[i]), approx, arcLength(Mat(contours_table[i]), true)*0.02, true);

        if (fabs(contourArea(contours_table[i])) < 100 || !isContourConvex(approx))
              continue;

        if (approx.size() >= 4 && approx.size() <= 6)
        {
          // Number of vertices of polygonal curve
          int vtc = approx.size();

          // Get the cosines of all corners
          vector<double> cos;
          for (int j = 2; j < vtc+1; j++)
            cos.push_back(angle(approx[j%vtc], approx[j-2], approx[j-1]));

          // Sort ascending the cosine values
          sort(cos.begin(), cos.end());

          // Get the lowest and the highest cosine
          double mincos = cos.front();
          double maxcos = cos.back();

          // Use the degrees obtained above and the number of vertices
          // to determine the shape of the contour
          if (vtc == 4 && mincos >= -0.1 && maxcos <= 0.3)
          {
            setLabel(im_bw_canny, "RECT", contours_table[i]);
            // Find the greatest rectangle
            Rect r = boundingRect(contours_table[i]);
            double rectangle_area = r.height*r.width; 
            if (rectangle_area>largest_rectangle_area) //find the largest rectangle
            {
              largest_rectangle_area = rectangle_area;
              max_rectangle_index = i;
              working_table=boundingRect(contours_table[i]);
            }
          }
        }
      }

      // If the working table is found make a new image with just the working table
      
      int table_limit_area = (im_gray.cols/2)*(im_gray.rows/3);
      if (largest_rectangle_area > table_limit_area) { 
        working_table = Rect(0,0,im_gray.cols,im_gray.rows) & working_table;
        im_working_table_gray = im_gray(working_table).clone();
        //imshow("Working table", im_working_table_gray);
        working_table_limits_found = 1;
        cout << "Working table found \n";
      }
    }


    // If the working table is found work with coordinates of working table, otherwise whole image
    Mat im_gray_, im_bw_objects;
    if (working_table_limits_found == 1)
    {
      im_gray_ = im_gray(working_table).clone();
      image = image(working_table).clone();
    } else {
      im_gray_ = im_gray;
    }
    //imshow("Color without label", image);


    ///////// End working table limits


 //////// Find the holes centers
    // adaptiveThreshold(im_gray_, im_bw_holes_, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY, 25, 5);
    threshold(im_gray_, im_bw_holes_, holes_threshold, 255.0, THRESH_OTSU);
    // Opening - erosion followed by dilation for noise (glare) removal
    Mat element = getStructuringElement( MORPH_ELLIPSE, 
                       Size( 2*erosion_size + 1, 2*erosion_size+1 ),
                       Point( erosion_size, erosion_size ));
    
    erode( im_bw_holes_, im_bw_holes_, element );
    dilate( im_bw_holes_, im_bw_holes_, element );


    //Canny(im_gray_, im_bw_canny_, 100, 105, 3);
    vector< vector <Point> > contours_holes;
    vector<Point> approx_holes;

    dst.create( im_gray.size(), im_gray.type() );
    namedWindow( window_name, CV_WINDOW_AUTOSIZE );
    createTrackbar( "Min Threshold:", window_name, &lowThreshold, max_lowThreshold, CannyThreshold );
    CannyThreshold(0, 0);
    waitKey(0);
    Size kernelSize (5,5);
    element = getStructuringElement (MORPH_RECT, kernelSize);
    morphologyEx( dst, dst, MORPH_CLOSE, element );

    im_gray_ = im_blue;

    findContours(im_bw_holes_, contours_holes, CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE); // Find the circles in the image based on Canny edge d.
    vector<Point2f>center_holes;
    vector<float>radius_vect_holes;
    vector<Point2f>true_object_center;
    vector<int>surrounding_avg_brightness_level;
    center_holes.reserve(contours_holes.size());
    radius_vect_holes.reserve(contours_holes.size());
    true_object_center.reserve(n_expected_objects);
    surrounding_avg_brightness_level.reserve(contours_holes.size());
    vector <bool> objects_flag (contours_holes.size());
      for (int i = 0; i < contours_holes.size(); i ++)                         // iterate through each contour.
      { 
        sum_current_rectangle = 0;
        avg_current_rectangle = 0;
        approxPolyDP(Mat(contours_holes[i]), approx_holes, arcLength(Mat(contours_holes[i]), true)*0.02, true);

        if (fabs(contourArea(contours_holes[i])) < 100 || !isContourConvex(approx_holes))
              continue;

        if (approx_holes.size() > 6) // Detects just circles
        {
          // Detect and label circles
          double area = contourArea(contours_holes[i]);
          Rect r = boundingRect(contours_holes[i]);
          int radius = r.width / 2;

          if (abs(1 - ((double)r.width / r.height)) <= 0.3 &&
              abs(1 - (area / (CV_PI * pow(radius, 2)))) <= 0.3)
            {
              //setLabel(im_bw_holes_, "CIR", contours_holes[i]);
              minEnclosingCircle( (Mat)contours_holes[i], center_holes[i], radius_vect_holes[i] );

               // begin adaptive thr around centers
              int curr_x_coord, curr_y_coord;
              int min_x_coord = 10000, min_y_coord = 10000, max_x_coord = 0, max_y_coord = 0;
              for (int j = 0; j < contours_holes[i].size(); j++)
              {
                curr_x_coord = contours_holes[i][j].x;
                curr_y_coord = contours_holes[i][j].y;
                if (curr_x_coord < min_x_coord) 
                  min_x_coord = curr_x_coord;
                if (curr_y_coord < min_y_coord) 
                  min_y_coord = curr_y_coord;
                if (curr_x_coord > max_x_coord) 
                  max_x_coord = curr_x_coord;
                if (curr_y_coord > max_y_coord) 
                  max_y_coord = curr_y_coord;                
              }
              vector <int> contours_total_gray_level;
              int n_total = (max_x_coord - min_x_coord) * (max_y_coord - min_y_coord);
              contours_total_gray_level.reserve(n_total);
              for (int j = min_x_coord; j < max_x_coord; j++)
              {
                for (int k = min_y_coord; k < max_y_coord; k++)
                {
                  if ((pow((j - center_holes[i].x),2) + pow((k-center_holes[i].y),2)) < pow(radius_vect_holes[i],2))
                  {
                    //cout << j << ", " << k << ", L " << int(im_gray_.at<unsigned char>(k,j)) << "\n" ;
                    contours_total_gray_level.push_back(int(im_gray_.at<unsigned char>(k,j)));
                  }
                }
              }
              // avg_gray_level = sum(contours_total_gray_level);
              int sum_x = cv::sum( contours_total_gray_level )[0];
              int avg_contour = sum_x / contours_total_gray_level.size();
              cout << "AVG area " << avg_contour << " ";

              Rect r(int(center_holes[i].x)-20,int(center_holes[i].y)-20,40,40);
              bounding_patch = Rect(0,0,im_blue.cols,im_blue.rows) & r;
              im_objectholepatch.push_back(im_blue(bounding_patch).clone());

              rectangle(image, r, Scalar(255,0,0));

              for (int k = 0; k < 2; k++)
              {
                for(int m = 0; m < surrounding_rectangle_size; m++)
                {
                  if (k == 0)
                  {
                    sum_current_rectangle += int(im_gray_.at<unsigned char>( (center_holes[i].y-surrounding_rectangle_size/2),
                    (center_holes[i].x+(surrounding_rectangle_size/2 - m)) ) );
                    if (m < (surrounding_rectangle_size-2))
                    {
                      sum_current_rectangle += int(im_gray_.at<unsigned char>( (center_holes[i].y+(surrounding_rectangle_size/2 - m - 1)),
                      (center_holes[i].x-surrounding_rectangle_size/2) ) );

                    }
                  } else if (k == 1)
                  {
                    sum_current_rectangle += int(im_gray_.at<unsigned char>( (center_holes[i].y+surrounding_rectangle_size/2),
                    (center_holes[i].x+(surrounding_rectangle_size/2 - m)) ) );
                    if (m < (surrounding_rectangle_size-2))
                    {
                      sum_current_rectangle += int(im_gray_.at<unsigned char>( (center_holes[i].y+(surrounding_rectangle_size/2 - m - 1)),
                      (center_holes[i].x+surrounding_rectangle_size/2) ) );
                    }
                  }

                }
              }

              avg_current_rectangle = sum_current_rectangle / (surrounding_rectangle_size*4-4);
              surrounding_avg_brightness_level.push_back(avg_current_rectangle);
              cout << "Avg bound rect: " << avg_current_rectangle << "\n";

              //center +- radius -> max/avg value
              //current_object_array = //create image center_holes[i].x + radius_vect_holes[i]
              //sum(array___)
              //avg = ; 
              //im_objectmask = im_gray_(r);
              // end adaptive thr around centers

              /*for (int j = 0; j < center_objects.size(); j++)
              {
                if (center_holes[i].x < (center_objects[j].x + center_proximity) && 
                center_holes[i].x > (center_objects[j].x - center_proximity) && 
                center_holes[i].y < (center_objects[j].y + center_proximity) && 
                center_holes[i].y > (center_objects[j].y - center_proximity))
                {
                  true_object_center.push_back(center_objects[j]);
                  // flag i -> do not draw the circle
                  objects_flag[i] = 1;
                }
              }*/

            }
        }


        /*if (objects_flag[i] != 1) 
        {*/
            for (int j = 0; j < center_holes.size(); j++)
            {
              if (center_holes[i].x < (center_holes[j].x + center_proximity) &&
              center_holes[i].x > (center_holes[j].x - center_proximity) &&
              center_holes[i].y < (center_holes[j].y + center_proximity) && 
              center_holes[i].y > (center_holes[j].y - center_proximity))
              {
                center_holes.erase(center_holes.begin() + j);
              }
            }

        //}
        if (radius_vect_holes[i] > 0 && radius_vect_holes[i] < 30)
        {
          center_holes.push_back(center_holes[i]); // Fill the vector
          radius_vect_holes.push_back(radius_vect_holes[i]);
          // draw red circle if it is a hole
          // circle( image, center_holes[i], (int)radius_vect_holes[i], Scalar ( 0,0,255), 2, 8, 0 );
        }
      }

      //////// End of the holes centers

      Scalar mean;
      Scalar stdev;
      meanStdDev(surrounding_avg_brightness_level,mean,stdev);
      cout << "mean" << mean[0] << "STD" << stdev[0];


    vector <float> object_threshold_vec(im_objectholepatch.size());
    vector<Mat> im_objectholepatch_bin(im_objectholepatch.size());

    vector <vector <Point> > contours_objects;
    vector<Point> approx_objects;

    vector<Point2f>center_objects;
    vector<float>radius_vect_objects;
    center_objects.reserve(1);
    radius_vect_objects.reserve(1);

    for (int i = 0; i < im_objectholepatch.size(); i++)
    {
      //object_threshold_vec[i] = object_threshold;
      object_threshold_vec[i] = object_threshold - (mean[0] - surrounding_avg_brightness_level[i])/2;
      cout << "OBJ THR " << object_threshold_vec[i] << "\n";
      threshold(im_objectholepatch[i], im_objectholepatch_bin[i], object_threshold_vec[i], 255.0, THRESH_BINARY);
      findContours(im_objectholepatch_bin[i], contours_objects, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE); // Find the contours in the image based on threshold
      for (int j = 0; j < contours_objects.size(); j++)
      {
        approxPolyDP(Mat(contours_objects[j]), approx_objects, arcLength(Mat(contours_objects[j]), true)*0.02, true);
        if (fabs(contourArea(contours_objects[j])) < 100 || !isContourConvex(approx_objects))
                continue;
        if (approx_objects.size() > 6) // Detects just circles
        {
          double area_objects = contourArea(contours_objects[j]);
          Rect r_objects = boundingRect(contours_objects[j]);
          int radius_objects = r_objects.width / 2;
          
          if (abs(1 - ((double)r_objects.width / r_objects.height)) <= 0.3 &&
              abs(1 - (area_objects / (CV_PI * pow(radius_objects, 2)))) <= 0.3
              //&& area_objects > min_contour_area && area_objects < max_contour_area
              )
              {
                  circle( image, center_holes[i], (int)radius_vect_holes[i], Scalar ( 0,255,0), 2, 8, 0 ); // draw green circle around the contour
                  cout << "Center Object coordinates" << center_holes[i] << "\n";            // Show circle coordinates 

              }
        }
      }

    }


    //////// Find the object centers based on 1. thresholding of image (the "whitest" objects) and 2. Circle shape
    //threshold(im_gray_, im_bw_objects, object_threshold, 255.0, THRESH_BINARY);
    
    // Apply (uncomment) erosion if the objects are too close together
    /*Mat element = getStructuringElement( MORPH_ELLIPSE, 
                       Size( 2*erosion_size + 1, 2*erosion_size+1 ),
                       Point( erosion_size, erosion_size ));
    erode( im_bw_objects, im_bw_objects, element );*/
    
    /*vector< vector <Point> > contours_objects;
    vector<Point> approx_objects;
    findContours(im_bw_objects, contours_objects, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE); // Find the contours in the image based on threshold
    vector<Point2f>center_objects;
    vector<float>radius_vect_objects;
    center_objects.reserve(n_expected_objects);
    radius_vect_objects.reserve(n_expected_objects);

    int j = 0;
    for (int i = 0; i < contours_objects.size(); i ++)                         // iterate through each contour.
    {
      approxPolyDP(Mat(contours_objects[i]), approx_objects, arcLength(Mat(contours_objects[i]), true)*0.02, true);

      if (fabs(contourArea(contours_objects[i])) < 100 || !isContourConvex(approx_objects))
              continue;

        if (approx_objects.size() > 6) // Detects just circles
        {
          // Detect and label objects
          double area_objects = contourArea(contours_objects[i]);
          Rect r_objects = boundingRect(contours_objects[i]);
          int radius_objects = r_objects.width / 2;
          
          if (abs(1 - ((double)r_objects.width / r_objects.height)) <= 0.3 &&
              abs(1 - (area_objects / (CV_PI * pow(radius_objects, 2)))) <= 0.3 &&
              area_objects > min_contour_area && area_objects < max_contour_area)
              {
                //cout << "Object " << i << " area: " << area_objects << "\n";
                minEnclosingCircle( (Mat)contours_objects[i], center_objects[j], radius_vect_objects[j] );

                if (center_objects[j].x > 1.0 && center_objects[j].y > 1.0 
                && center_objects[j].x < 640.0 && center_objects[j].y < 480.0 )
                {
                  center_objects.push_back(center_objects[j]); // Fill the vector
                  radius_vect_objects.push_back(radius_vect_objects[j]);
                  circle( image, center_objects[j], (int)radius_vect_objects[j], Scalar ( 0,255,0), 2, 8, 0 ); // draw green circle around the contour
                  cout << "Center Object coordinates" << center_objects[j] << "\n";            // Show circle coordinates 
                  j++;
                }
              }
        }
    }*/


    //////// End of object centers
//
   


      /*for (int i = 0; i < true_object_center.size(); i++)
      {
        cout << "Object [" << i << "] coordinates -> x: " << true_object_center[i].x << ", y: " << true_object_center[i].y << "\n";
      }*/
      for (int i = 0; i < center_holes.size(); i++)
      {
        cout << "Hole [" << i << "] coordinates -> x: " << center_holes[i].x << ", y: " << center_holes[i].y << "\n";
      }
      
      /// Separate the image in 3 places ( B, G and R )
      vector<Mat> bgr_planes;

      /// Establish the number of bins
      int histSize = 256;

      /// Set the ranges ( for B,G,R) )
      float range[] = { 0, 256 } ;
      const float* histRange = { range };

      bool uniform = true; bool accumulate = false;

      Mat gr_hist;

      /// Compute the histograms:
      calcHist( &im_gray_, 1, 0, Mat(), gr_hist, 1, &histSize, &histRange, uniform, accumulate );

      // Draw the histograms for B, G and R
      int hist_w = 256; int hist_h = 400;
      int bin_w = cvRound( (double) hist_w/histSize );

      Mat histImage( hist_h, hist_w, CV_8UC3, Scalar( 0,0,0));

      /// Normalize the result to [ 0, histImage.rows ]
      normalize(gr_hist, gr_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat() );

      /// Draw for each channel
      for( int i = 1; i < histSize; i++ )
      {
          line( histImage, Point( bin_w*(i-1), hist_h - cvRound(gr_hist.at<float>(i-1)) ) ,
                          Point( bin_w*(i), hist_h - cvRound(gr_hist.at<float>(i)) ),
                          Scalar( 255, 0, 0), 2, 8, 0  );
      }

      /// Display
      namedWindow("calcHist Demo", CV_WINDOW_AUTOSIZE );
      imshow("calcHist Demo", histImage );



//

      //imshow("Color image", cv_ptr->image);
      imshow(OPENCV_WINDOW, image);

      imshow("Grayscale", im_gray_);
      imshow("Holes thresholding", im_bw_holes_);
      imshow("Canny holes", dst);
      ///////// End image processing /////////
    
      for (int i = 0; i < im_objectholepatch.size(); i++)
      {
        imshow("Patch", im_objectholepatch[i]);
        imshow("PatchBin",im_objectholepatch_bin[i]);
        waitKey(0);
      }
      //imshow("Color image", cv_ptr->image);
      imshow("Binary", im_bw_objects);
      ///////// End image processing /////////
      waitKey(0);
      return 0;
}


