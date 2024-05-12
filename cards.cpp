#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>

using namespace cv;
using namespace std;

int main() {

    setlocale(LC_ALL, "Russian");
    Mat img_scene = imread("C:/Users/User/Downloads/cards.jpg");
    if (img_scene.empty()) {
        cout << "Невозможно загрузить изображение" << endl;
        return -1;
    }

    Ptr<SIFT> sift = SIFT::create();

    Mat img_cards[4];
    string card_paths[4] = { "C:/Users/User/Downloads/9_bub.png","C:/Users/User/Downloads/9_love.png", "C:/Users/User/Downloads/10_piki.png", "C:/Users/User/Downloads/10_kresti.png" };
    vector<vector<KeyPoint>> keypoints_cards;
    vector<Mat> descriptors_cards;

    for (int i = 0; i < 4; ++i) {
        img_cards[i] = imread(card_paths[i], IMREAD_GRAYSCALE);
        if (img_cards[i].empty()) {
            cout << "Невозможно загрузить изображение " << card_paths[i] << endl;
            return -1;
        }

        vector<KeyPoint> keypoints_card;
        Mat descriptors_card;
        sift->detectAndCompute(img_cards[i], noArray(), keypoints_card, descriptors_card);
        keypoints_cards.push_back(keypoints_card);
        descriptors_cards.push_back(descriptors_card);
    }

    vector<KeyPoint> keypoints_scene;
    Mat descriptors_scene;
    sift->detectAndCompute(img_scene, noArray(), keypoints_scene, descriptors_scene);

    FlannBasedMatcher matcher;
    vector<vector<DMatch>> knn_matches;
    vector<vector<Point2f>> scene_points_list;
    vector<vector<Point2f>> card_points_list;

    for (size_t i = 0; i < 4; ++i) {
        matcher.knnMatch(descriptors_cards[i], descriptors_scene, knn_matches, 2);

        vector<DMatch> good_matches;
        vector<Point2f> scene_points, card_points;
        for (size_t j = 0; j < knn_matches.size(); ++j) {
            if (knn_matches[j][0].distance < 0.7 * knn_matches[j][1].distance) {
                good_matches.push_back(knn_matches[j][0]);
                scene_points.push_back(keypoints_scene[knn_matches[j][0].trainIdx].pt);
                card_points.push_back(keypoints_cards[i][knn_matches[j][0].queryIdx].pt);
            }
        }

        scene_points_list.push_back(scene_points);
        card_points_list.push_back(card_points);
    }

    for (size_t i = 0; i < 4; ++i) {
        Mat H = findHomography(card_points_list[i], scene_points_list[i], RANSAC);

        vector<Point2f> card_corners(4);
        card_corners[0] = Point2f(0, 0);
        card_corners[1] = Point2f(img_cards[i].cols, 0);
        card_corners[2] = Point2f(img_cards[i].cols, img_cards[i].rows);
        card_corners[3] = Point2f(0, img_cards[i].rows);

        vector<Point2f> scene_corners(4);
        perspectiveTransform(card_corners, scene_corners, H);

        line(img_scene, scene_corners[0], scene_corners[1], Scalar(0, 255, 0), 2);
        line(img_scene, scene_corners[1], scene_corners[2], Scalar(0, 255, 0), 2);
        line(img_scene, scene_corners[2], scene_corners[3], Scalar(0, 255, 0), 2);
        line(img_scene, scene_corners[3], scene_corners[0], Scalar(0, 255, 0), 2);

        putText(img_scene, card_paths[i], scene_corners[0], FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 255, 0), 2);
    }

    imshow("Результат", img_scene);
    waitKey(0);

    return 0;
}