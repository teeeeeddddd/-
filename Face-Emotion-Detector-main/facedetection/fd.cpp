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
    CascadeClassifier faceCascade, smileCascade, eyeCascade,profilefaceCascade;
    Mat img;

    // Load the Haar cascade files for face and smile detection
    faceCascade.load("haarcascade_frontalface_default.xml");
    profilefaceCascade.load("haarcascade_profileface.xml");
    smileCascade.load("haarcascade_smile.xml");
    eyeCascade.load("haarcascade_eye.xml");


    if (faceCascade.empty() || smileCascade.empty() || eyeCascade.empty()) {
        cerr << "Error loading Haar cascade files" << endl;
        return -1;
    }

    while (true) {
        video.read(img);

        vector<Rect> faces;
        faceCascade.detectMultiScale(img, faces, 1.3, 5);
        vector<Rect> profilefaces;
        profilefaceCascade.detectMultiScale(img, profilefaces, 1.3, 5);
        faces.insert(faces.end(), profilefaces.begin(),profilefaces.end());
        bool smileDetected = false;

        for (int i = 0; i < faces.size(); i++) {
            Mat faceROI = img(faces[i]);

            vector<Rect> eyes;
            eyeCascade.detectMultiScale(faceROI, eyes, 1.1, 3);
            for (int j = 0; j < eyes.size(); j++) {
                Rect eyeRect(eyes[j].x + faces[i].x, eyes[j].y + faces[i].y, eyes[j].width, eyes[j].height);
                rectangle(img, eyeRect.tl(), eyeRect.br(), Scalar(255, 0, 0), 2);
            }

            trackEyes(img, eyes, faces[i]);

            vector<Rect> smiles;
            smileCascade.detectMultiScale(faceROI, smiles, 1.8, 20);

            for (int j = 0; j < smiles.size(); j++) {
                Rect smileRect(smiles[j].x + faces[i].x, smiles[j].y + faces[i].y, smiles[j].width, smiles[j].height);
                rectangle(img, smileRect.tl(), smileRect.br(), Scalar(0, 255, 0), 2);
                smileDetected = true;
            }

            rectangle(img, faces[i].tl(), faces[i].br(), Scalar(144, 238, 144), 2);
        }

        // Draw the black box at the top center
        int box_width = 200;
        int box_height = 70;
        int box_x = (img.cols - box_width) / 2;
        rectangle(img, Point(box_x, 0), Point(box_x + box_width, box_height), Scalar(0, 0, 0), FILLED);

        // Prepare the text for faces detected
        string text = to_string(faces.size()) + " Face Found";
        int font_face = FONT_HERSHEY_SCRIPT_COMPLEX;
        double font_scale = 0.8;
        int thickness = 1;
        int baseline = 0;
        Size text_size = getTextSize(text, font_face, font_scale, thickness, &baseline);
        baseline += thickness;
        int text_x = box_x + (box_width - text_size.width) / 2;
        int text_y = (box_height + text_size.height) / 2;

        // Put the text on the image
        putText(img, text, Point(text_x, text_y), font_face, font_scale, Scalar(255, 255, 255), thickness);

        // Prepare the text for smile detected
        if (smileDetected) {
            string smileText = "Smile Detected!";
            double font_scale_new = 0.6;
            Size smile_text_size = getTextSize(smileText, font_face, font_scale_new, thickness, &baseline);
            int smile_text_x = 10; // Adjusted x position to move text to the top left
            int smile_text_y = 2 * (text_size.height + 4); // Adjusted y position to move text to the top left
            putText(img, smileText, Point(smile_text_x, smile_text_y), font_face, font_scale_new, Scalar(0, 0, 0), thickness); // Changed font color to green
        }
        imshow("Frame", img);
        if (waitKey(1) == 27) { // Press 'Esc' to exit
            break;
        }
    }
    return 0;
}
