#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect.hpp>
#include <iostream>

using namespace std;
using namespace cv;

void trackEyes(const Mat& frame, const vector<Rect>& eyes, const Rect& faceRect) {
    for (const auto& eye : eyes) {
        // Calculate the center of the eye relative to the entire frame
        Point center(eye.x + faceRect.x + eye.width / 2, eye.y + faceRect.y + eye.height / 2);
        // Draw a circle at the center of the eye
        circle(frame, center, 2, Scalar(0, 255, 0), 2);
    }
}

int main() {
    VideoCapture video(0);
    CascadeClassifier faceCascade, smileCascade, eyeCascade, profilefaceCascade;
    CascadeClassifier mouthCascade;
    Mat img;

    // 加载haar模型
    faceCascade.load("haarcascade_frontalface_default.xml");
    profilefaceCascade.load("haarcascade_profileface.xml");
    smileCascade.load("haarcascade_smile.xml");
    mouthCascade.load("haarcascade_mcs_mouth.xml");
    eyeCascade.load("haarcascade_eye.xml");
    
    if (faceCascade.empty() || smileCascade.empty() || eyeCascade.empty() || mouthCascade.empty()) {
        cerr << "Error loading Haar cascade files" << endl;
        return -1;
    }

    //嘴巴大小阈值
    double mouthHeightSurprise = 50;
    double mouthWidthSurprise  = 80;
    double mouthHeightSadness  = 40;
    double mouthwidthSadness   = 50;
    while (true) {
        video.read(img);
        //检测人脸，将正脸和侧脸混合
        vector<Rect> faces;
        faceCascade.detectMultiScale(img, faces, 1.3, 5);
        vector<Rect> profilefaces;
        profilefaceCascade.detectMultiScale(img, profilefaces, 1.3, 5);
        faces.insert(faces.end(), profilefaces.begin(), profilefaces.end());
        //判断函数
        bool smileDetected = false;
        bool surpriseDetected = false;
        bool sadnessDetected = false;
        for (int i = 0; i < faces.size(); i++) {
            Mat faceROI = img(faces[i]);
            vector<Rect> eyes;
            eyeCascade.detectMultiScale(faceROI, eyes, 1.1, 3);
            for (int j = 0; j < eyes.size(); j++) {
                Rect eyeRect(eyes[j].x + faces[i].x, eyes[j].y + faces[i].y, eyes[j].width, eyes[j].height);
                rectangle(img, eyeRect.tl(), eyeRect.br(), Scalar(255, 0, 0), 2);
            }

            trackEyes(img, eyes, faces[i]);
            //检测笑容
            vector<Rect> smiles;
            smileCascade.detectMultiScale(faceROI, smiles, 1.8, 20);

            for (int j = 0; j < smiles.size(); j++) {
                Rect smileRect(smiles[j].x + faces[i].x, smiles[j].y + faces[i].y, smiles[j].width, smiles[j].height);
                rectangle(img, smileRect.tl(), smileRect.br(), Scalar(0, 255, 0), 2);
                smileDetected = true;
            }

            rectangle(img, faces[i].tl(), faces[i].br(), Scalar(144, 238, 144), 2);
            //检测嘴巴
            vector<Rect> mouths;
            mouthCascade.detectMultiScale(faceROI, mouths, 1.1, 5, 0, Size(30, 30));

            for (int j = 0; j < mouths.size(); j++) {
                Rect mouthRect(mouths[j].x + faces[i].x, mouths[j].y + faces[i].y, mouths[j].width, mouths[j].height);
                rectangle(img, mouthRect.tl(), mouthRect.br(), Scalar(0, 0, 255), 2);
                if (mouths[j].height > mouthHeightSurprise && mouths[j].width > mouthWidthSurprise) {
                    surpriseDetected = true;
                }
                if (mouths[j].height < mouthHeightSadness && mouths[j].width < mouthWidthSurprise) {
                    sadnessDetected = true;
                }
            }
        }
        int box_width = 200;
        int box_height = 70;
        int box_x = (img.cols - box_width) / 2;
        rectangle(img, Point(box_x, 0), Point(box_x + box_width, box_height), Scalar(0, 0, 0), FILLED);

        //文本准备
        string text = to_string(faces.size()) + " Face Found";
        int font_face = FONT_HERSHEY_PLAIN ;
        double font_scale = 0.8;
        int thickness = 1;
        int baseline = 0;
        Size text_size = getTextSize(text, font_face, font_scale, thickness, &baseline);
        baseline += thickness;
        int text_x = box_x + (box_width - text_size.width) / 2;
        int text_y = (box_height + text_size.height) / 2;

       
        cv::putText(img, text, Point(text_x, text_y), font_face, font_scale, Scalar(255, 255, 255), thickness);

        
        if (smileDetected) {
            string smileText = "Smile Detected!";
            double font_scale_new = 0.6;
            Size smile_text_size = getTextSize(smileText, font_face, font_scale_new, thickness, &baseline);
            int smile_text_x = 10;
            int smile_text_y = 2 * (text_size.height + 4);
            cv::putText(img, smileText, Point(smile_text_x, smile_text_y), font_face, font_scale_new, Scalar(0, 255, 0), thickness);
        }

        // Prepare the text for surprise detected
        if (surpriseDetected) {
            string surpriseText = "Surprise Detected!";
            double font_scale_new = 0.6;
            Size surprise_text_size = getTextSize(surpriseText, font_face, font_scale_new, thickness, &baseline);
            int surprise_text_x = 10;
            int surprise_text_y = 3 * (text_size.height + 4);
            putText(img, surpriseText, Point(surprise_text_x, surprise_text_y), font_face, font_scale_new, Scalar(255, 0, 0), thickness);
        }
        if (sadnessDetected) {
            string sadnessText = "Sadness Detected!";
            double font_scale_new = 0.6;
            Size sadness_text_size = getTextSize(sadnessText, font_face, font_scale_new, thickness, &baseline);
            int sadness_text_x = 10;
            int sadness_text_y = 4 * (text_size.height + 4);
            putText(img, sadnessText, Point(sadness_text_x, sadness_text_y), font_face, font_scale_new, Scalar(255, 165, 0), thickness);
        }
        
        imshow("人脸识别", img);
        if (waitKey(1) == 27) { // Press 'Esc' to exit
            break;
        }
    }
    return 0;
}
