#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <map>
#include <limits>
#include <unordered_map>
#include <algorithm>
#include <deque>
#include <cmath>
#include <numeric>
#include <unordered_set>
#include <filesystem>
#include <opencv2/opencv.hpp>
#include <Eigen/Dense>
#include <Eigen/Core>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include "BSplineCurve_modified.h"
#include <tuple>
#include <nlohmann/json.hpp>
#include <stdlib.h>
#include <math.h>

namespace fs = std::filesystem;
using json = nlohmann::json;


Eigen::MatrixXf dropRowsByIndex(const Eigen::MatrixXf &matrix, const std::vector<int> &indices_to_remove)
{
    std::vector<int> indices_to_keep;
    for (int i = 0; i < matrix.rows(); ++i)
    {
        if (std::find(indices_to_remove.begin(), indices_to_remove.end(), i) == indices_to_remove.end())
        {
            indices_to_keep.push_back(i);
        }
    }

    Eigen::MatrixXf result(indices_to_keep.size(), matrix.cols());

    for (size_t i = 0; i < indices_to_keep.size(); ++i)
    {
        result.row(i) = matrix.row(indices_to_keep[i]);
    }

    return result;
}

std::vector<int> findOutliers(const Eigen::VectorXf &data)
{
    std::vector<int> outlierIndices;

    Eigen::VectorXf sortedData = data;
    std::sort(sortedData.data(), sortedData.data() + sortedData.size());
    float q1 = sortedData[(int)(sortedData.size() / 4)];
    float q3 = sortedData[(int)(3 * sortedData.size() / 4)];
    float iqr = q3 - q1;

    float lowerBound = q1 - 2 * iqr;
    float upperBound = q3 + 2 * iqr;

    for (int i = 0; i < data.size(); ++i)
    {
        if (data[i] < lowerBound || data[i] > upperBound)
        {
            outlierIndices.push_back(i);
        }
    }
    // for (int index : outlierIndices)
    // {
    //     std::cout << "Outlier Index: " << index << " Value: " << data(index) << std::endl;
    // }
    return outlierIndices;
}

Eigen::VectorXf polynomial_Fit_2nd_degree(int size, Eigen::VectorXf &x, Eigen::VectorXf &solve_var)
{
    Eigen::MatrixXf A(size, 3);
    A.col(0) = x.array().pow(2);
    A.col(1) = x;
    A.col(2) = Eigen::VectorXf::Ones(size);

    Eigen::VectorXf coeffs = A.colPivHouseholderQr().solve(solve_var);
    return coeffs;
}

float euclideanDistance(const Eigen::Vector3f &p1, const Eigen::Vector3f &p2)
{
    return (p1 - p2).norm();
}

struct NumericStringCompare
{
    bool operator()(const std::string &a, const std::string &b) const
    {
        return std::stoi(a) < std::stoi(b); // Compare as integers
    }
};

struct PointXYZIR
{
    float x, y, z, intensity;
    int ring;

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
} EIGEN_ALIGN16;

POINT_CLOUD_REGISTER_POINT_STRUCT(PointXYZIR,
                                  (float, x, x)(float, y, y)(float, z, z)(float, intensity, intensity)(int, ring, ring))

namespace cv
{
    template <>
    class DataType<PointXYZIR>
    {
    public:
        typedef PointXYZIR value_type;
        typedef int work_type;
        enum
        {
            generic_type = 0,
            depth = DataType<work_type>::depth,
            channels = 5,
            fmt = DataType<work_type>::fmt,
            type = CV_MAKETYPE(depth, channels)
        };
    };
}

std::vector<PointXYZIR> convertMatrixToPoints(const Eigen::MatrixXf &matrix)
{
    std::vector<PointXYZIR> points;
    points.reserve(matrix.rows()); 
    
    for (int i = 0; i < matrix.rows(); ++i)
    {
        PointXYZIR point;
        point.x = matrix(i, 0);
        point.y = matrix(i, 1);
        point.z = matrix(i, 2);
        point.intensity = 0;
        point.ring = static_cast<int>(matrix(i, 0));

        points.push_back(point);
    }

    return points;
}


float calculateSSE2(Eigen::VectorXf &y, Eigen::VectorXf &predicted_y)
{
    Eigen::VectorXf residuals = y - predicted_y;
    return residuals.squaredNorm();
}

float calculateSSE(Eigen::VectorXf &x1, Eigen::VectorXf &x2)
{

    std::vector<int> valid_indices_set1;
    std::vector<int> valid_indices_set2;

    float threshold1 = std::min(x1.minCoeff(), x2.minCoeff());
    float threshold2 = std::max(x1.maxCoeff(), x2.maxCoeff());

    for (int i = 0; i < x1.size() / 3; ++i)
    {
        if (x1(i) > threshold1 && x1(i) < threshold2)
        {
            valid_indices_set1.push_back(i); 
        }
    }

    for (int i = 0; i < x2.size() / 3; ++i)
    {
        if (x2(i) > threshold1 && x2(i) < threshold2)
        {
            valid_indices_set2.push_back(i); 
        }
    }

    if (valid_indices_set1.empty() || valid_indices_set2.empty())
    {
        return std::numeric_limits<float>::infinity();
    }

    int valid_size = std::min(valid_indices_set1.size(), valid_indices_set2.size());

    Eigen::VectorXf filtered_set1(valid_size * 3); 
    Eigen::VectorXf filtered_set2(valid_size * 3);

    for (int i = 0; i < valid_size; ++i)
    {
        filtered_set1.segment(i * 3, 3) = x1.segment(valid_indices_set1[i] * 3, 3);
        filtered_set2.segment(i * 3, 3) = x2.segment(valid_indices_set2[i] * 3, 3);
    }

    Eigen::VectorXf residuals = filtered_set1 - filtered_set2;
    residuals = residuals / residuals.rows();
    return residuals.squaredNorm();
}

class WaymoDatasetHandler
{
private:
    std::string tfrecord_file_path;

    std::string openlane_labels_dir;
    std::vector<std::string> openlane_labels_files_list;

    std::string front_camera_segmentation_masks_dir;
    std::vector<std::string> front_camera_segmentation_masks_files_list;

    std::string output_dir;
    std::string pointcloud_file;
    std::string front_image_file;
    std::string mask_file;
    std::string labels_file;
    std::string depthmap_file;

    int image_height;
    int image_width;
    std::string frame_index;
    std::map<std::string, std::vector<Eigen::VectorXf>> tcks;
    std::map<std::string, std::vector<std::vector<float>>> tcks2;
    std::unordered_map<int, int> track_no_objects;

public:
    cv::Mat front_camera_image;
    cv::Mat front_camera_mask;

    Eigen::Matrix4f lidar_to_ego_trans;

    Eigen::Matrix4f front_camera_to_ego_trans;

    Eigen::Matrix3f front_camera_intrinsic;

    Eigen::Matrix4f ego_to_front_camera_trans;

    cv::Mat front_camera_depthmap;
    std::vector<PointXYZIR> lidars_pointcloud;
    Eigen::MatrixXf pointcloud_in_camera;

    std::map<std::string, std::vector<std::pair<int, int>>> connected_lanes;
    std::map<std::string, std::vector<Eigen::MatrixXf>, NumericStringCompare> lanes_labels_per_frame;

    WaymoDatasetHandler()
    {
        image_height = 0;
        image_width = 0;
        frame_index = "";
        tfrecord_file_path = "waymo/waymo_dataset/segment-15832924468527961_1564_160_1584_160_with_camera_labels.tfrecord";
        openlane_labels_dir = "waymo/waymo_dataset/segment-15832924468527961_1564_160_1584_160_with_camera_labels";
        front_camera_segmentation_masks_dir = "waymo/waymo_dataset/segment-15832924468527961_1564_160_1584_160_with_camera_labels_front_camera_masks";

        output_dir = "saved_results";
        lidar_to_ego_trans << -0.85282908, -0.52218559, -0.00218336, 1.43,
            0.52218959, -0.85281457, -0.00503366, 0.0,
            0.00076651, -0.00543298, 0.99998495, 2.184,
            0.0, 0.0, 0.0, 1.0;

        front_camera_to_ego_trans << 0.99968703, -0.00304848, -0.02483053, 1.54436336,
            0.00353814, 0.99979967, 0.01970015, -0.02267064,
            0.0247655, -0.01978184, 0.99949755, 2.11579855,
            0.0, 0.0, 0.0, 1.0;

        front_camera_intrinsic << 2084.21565711, 0.0, 982.69940656,
            0.0, 2084.21565711, 647.34228763,
            0.0, 0.0, 1.0;

        ego_to_front_camera_trans = front_camera_to_ego_trans.inverse();
        for (const auto &entry : fs::directory_iterator(openlane_labels_dir))
        {
            openlane_labels_files_list.push_back(entry.path().filename().string());
        }
        std::sort(openlane_labels_files_list.begin(), openlane_labels_files_list.end());

        for (const auto &entry : fs::directory_iterator(front_camera_segmentation_masks_dir))
        {
            front_camera_segmentation_masks_files_list.push_back(entry.path().filename().string());
        }
        std::sort(front_camera_segmentation_masks_files_list.begin(), front_camera_segmentation_masks_files_list.end());
    }

    Eigen::MatrixXf cart2hom(const std::vector<PointXYZIR> &pts_3d)
    {
        int n = pts_3d.size();
        Eigen::MatrixXf pts_3d_hom(n, 4);

        for (int i = 0; i < n; ++i)
        {
            const PointXYZIR &point = pts_3d[i];
            pts_3d_hom(i, 0) = point.x;
            pts_3d_hom(i, 1) = point.y;
            pts_3d_hom(i, 2) = point.z;
            pts_3d_hom(i, 3) = 1.0;
        }

        return pts_3d_hom;
    }

    Eigen::MatrixXf transformation(const std::vector<PointXYZIR> &pts_3d, const Eigen::Matrix4f &trans)
    {

        Eigen::MatrixXf pts_3d_hom = cart2hom(pts_3d);

        return pts_3d_hom * trans.transpose();
    }

    Eigen::MatrixXf project_cam_to_image(const Eigen::MatrixXf &pts_3d_rect, const Eigen::MatrixXf &cam_intrinsic)
    {

        Eigen::MatrixXf pts_2d = pts_3d_rect * cam_intrinsic.transpose();

        for (int i = 0; i < pts_2d.rows(); ++i)
        {
            pts_2d(i, 0) /= pts_2d(i, 2); // x / z
            pts_2d(i, 1) /= pts_2d(i, 2); // y / z
        }

        return pts_2d.leftCols(2);
    }

    Eigen::MatrixXf convert_waymo_camera_points_to_standard_camera_coordinates(Eigen::MatrixXf &pts_3d_rect)
    {

        Eigen::MatrixXf reordered_pts_3d_rect(pts_3d_rect.rows(), 3);
        reordered_pts_3d_rect.col(0) = pts_3d_rect.col(1);
        reordered_pts_3d_rect.col(1) = pts_3d_rect.col(2);
        reordered_pts_3d_rect.col(2) = pts_3d_rect.col(0);

        reordered_pts_3d_rect.col(0) = -reordered_pts_3d_rect.col(0);
        reordered_pts_3d_rect.col(1) = -reordered_pts_3d_rect.col(1);
        return reordered_pts_3d_rect;
    }

    cv::Mat get_depth_map(const std::vector<PointXYZIR> &points_in_ego)
    {

        pointcloud_in_camera = transformation(points_in_ego, ego_to_front_camera_trans).leftCols(3);
        Eigen::MatrixXf points_in_cam = convert_waymo_camera_points_to_standard_camera_coordinates(pointcloud_in_camera);
        Eigen::MatrixXf points_in_image;
        Eigen::MatrixXf points_in_img = project_cam_to_image(points_in_cam, front_camera_intrinsic);
        points_in_image = points_in_img.array().round().cast<float>();
        cv::Mat depth_map(image_height, image_width, CV_32F, cv::Scalar(0));
        for (int i = 0; i < points_in_image.rows(); ++i)
        {
            int u = points_in_image(i, 0);
            int v = points_in_image(i, 1);
            if (u >= 0 && u < image_width && v >= 0 && v < image_height && points_in_cam(i, 2) > 0)
            {
                depth_map.at<float>(v, u) = points_in_cam(i, 2);
            }
        }
        return depth_map;
    }

    void get_frame(std::string frame_index)
    {

        this->frame_index = frame_index;
        std::string str = front_camera_segmentation_masks_files_list[std::atoi(frame_index.c_str())];
        std::string firstElement = (str.find('.') != std::string::npos) ? str.substr(0, str.find('.')) : str;
        pointcloud_file = output_dir + "/pointclouds/frame_" + frame_index + "_lidar.npy";
        front_image_file = "waymo/waymo_dataset/segment-15832924468527961_1564_160_1584_160_with_camera_labels_front_camera_images/"+ frame_index + "_front_image.jpg";
        mask_file = "waymo/waymo_dataset/segment-15832924468527961_1564_160_1584_160_with_camera_labels_front_camera_masks/" + frame_index + ".jpg";
        labels_file = output_dir + "/labels/frame_" + frame_index + "_lane_labels.json";
        depthmap_file = output_dir + "/depthmaps/frame_" + frame_index + "_depthmap.npy";
        
        lidars_pointcloud = readNpyFile(pointcloud_file);

        front_camera_image = cv::imread(front_image_file, cv::IMREAD_COLOR);
        image_height = front_camera_image.rows;
        image_width = front_camera_image.cols;

        front_camera_mask = cv::imread(mask_file, cv::IMREAD_GRAYSCALE);
        front_camera_depthmap = get_depth_map(lidars_pointcloud);
    }

    void process_lanes_from_mask()
    {

        cv::Mat lanes_binary_mask;
        cv::inRange(front_camera_mask, cv::Scalar(200, 200, 200), cv::Scalar(255, 255, 255), lanes_binary_mask);
        std::vector<std::vector<cv::Point>> lanes_contours;
        cv::findContours(lanes_binary_mask, lanes_contours, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);

        std::vector<Eigen::MatrixXf> lanes_dict;
        int lane_contour_color_idx = 0;

        for (const auto &lane_contour : lanes_contours)
        {
            Eigen::MatrixXf lane_xyz_points_in_ego;

            cv::Mat lane_contour_mask = cv::Mat::zeros(lanes_binary_mask.size(), CV_8UC1);
            cv::fillPoly(lane_contour_mask, std::vector<std::vector<cv::Point>>{lane_contour}, cv::Scalar(255));
            lane_contour_mask.convertTo(lane_contour_mask, CV_32F, 1.0 / 255);
            cv::Mat lane_contour_depthmap = lane_contour_mask.mul(front_camera_depthmap);
            std::vector<cv::Point> lane_contour_depthmap_valid_indices;
            for (int y = 0; y < lane_contour_depthmap.rows; ++y)
            {
                for (int x = 0; x < lane_contour_depthmap.cols; ++x)
                {
                    if (lane_contour_depthmap.at<float>(y, x) != 0)
                    {
                        lane_contour_depthmap_valid_indices.push_back(cv::Point(x, y));
                    }
                }
            }

            std::vector<float> lane_contour_depths;
            for (const auto &pt : lane_contour_depthmap_valid_indices)
            {
                lane_contour_depths.push_back(front_camera_depthmap.at<float>(pt.y, pt.x));
            }

            std::vector<PointXYZIR> lane_contour_xyz_points;
            for (int i = 0; i < pointcloud_in_camera.rows(); ++i)
            {
                float x_value = pointcloud_in_camera(i, 0);
                if (std::find(lane_contour_depths.begin(), lane_contour_depths.end(), x_value) != lane_contour_depths.end())
                {
                    PointXYZIR point;
                    point.x = pointcloud_in_camera(i, 0);
                    point.y = pointcloud_in_camera(i, 1);
                    point.z = pointcloud_in_camera(i, 2);
                    point.intensity = 100.0f;
                    point.ring = 1;

                    lane_contour_xyz_points.push_back(point);
                }
            }

            Eigen::MatrixXf lane_contour_xyz_points_in_ego = transformation(lane_contour_xyz_points, front_camera_to_ego_trans);
            if (lane_contour_xyz_points_in_ego.rows() > 1)
            {
                lanes_dict.push_back(lane_contour_xyz_points_in_ego.leftCols(3));
            }
            lane_contour_color_idx += 1;
        }

        std::vector<Eigen::MatrixXf> lanes_splines_list;
        for (auto &lane_dict : lanes_dict)
        {
            if (lane_dict.rows() > 3)
            {
                Eigen::VectorXf x = lane_dict.col(0);
                Eigen::VectorXf y = lane_dict.col(1);
                Eigen::VectorXf z = lane_dict.col(2);
                std::vector<int> indices_to_remove = findOutliers(y);
                std::set<int> unique_elements(indices_to_remove.begin(), indices_to_remove.end());
                std::vector<int> unique_vec(unique_elements.begin(), unique_elements.end());
                Eigen::MatrixXf matrixI(x.size(), 3);
                matrixI << x, y, z;
                Eigen::MatrixXf new_matrix = dropRowsByIndex(matrixI, unique_vec);
                if (new_matrix.rows() < 4)
                {
                    continue;
                }
                Eigen::MatrixXf lane_dict_eval = generate_spline(new_matrix);
                if (lane_dict_eval.rows() < 4)
                {
                    continue;
                }
                generate_coeffs(lane_dict_eval);
                lanes_splines_list.push_back(lane_dict_eval);
            }
        }
        connected_lanes[frame_index] = connect_lanes(lanes_splines_list);
        lanes_labels_per_frame[frame_index] = lanes_splines_list;
    }

    Eigen::MatrixXf generate_spline(const Eigen::MatrixXf &points)
    {

        std::vector<float> pairwise_distances;
        int num_points = points.rows();
        for (int i = 0; i < num_points; ++i)
        {
            for (int j = i + 1; j < num_points; ++j)
            {
                Eigen::Vector3f p1 = points.row(i).head<3>();
                Eigen::Vector3f p2 = points.row(j).head<3>();
                pairwise_distances.push_back(euclideanDistance(p1, p2));
            }
        }

        Eigen::VectorXf x = points.col(0);
        Eigen::VectorXf y = points.col(1);
        Eigen::VectorXf z = points.col(2);

        float x_min = x.minCoeff();
        float x_max = x.maxCoeff();

        float max_distance = *max_element(pairwise_distances.begin(), pairwise_distances.end());
        int num_points_in_linspace = static_cast<int>(max_distance * 2);

        std::vector<int> sorted_indices(points.rows());
        std::iota(sorted_indices.begin(), sorted_indices.end(), 0);
        std::sort(sorted_indices.begin(), sorted_indices.end(),
                  [&points](int i, int j)
                  { return points(i, 0) < points(j, 0); });

        std::vector<int> unique_indices;
        unique_indices.push_back(sorted_indices[0]);

        for (int i = 1; i < sorted_indices.size(); ++i)
        {
            if ((points(sorted_indices[i], 0) - points(sorted_indices[i - 1], 0)) > 0.00009)
            {
                unique_indices.push_back(sorted_indices[i]);
            }
        }

        Eigen::MatrixXf points_3d(unique_indices.size(), points.cols());
        for (int i = 0; i < unique_indices.size(); ++i)
        {
            points_3d.row(i) = points.row(unique_indices[i]);
        }

        Eigen::VectorXf t(num_points_in_linspace);
        for (int i = 0; i < num_points_in_linspace; ++i)
        {
            t(i) = x_min + (x_max - x_min) * static_cast<float>(i) / (num_points_in_linspace - 1);
        }

        std::vector<double> x_v(points_3d.rows());
        std::vector<double> y_v(points_3d.rows());
        std::vector<double> z_v(points_3d.rows());

        for (int i = 0; i < x_v.size(); ++i)
        {
            x_v[i] = points_3d(i, 0);
            y_v[i] = points_3d(i, 1);
            z_v[i] = points_3d(i, 2);
        }
        std::vector<float> x_v2(num_points_in_linspace);
        std::vector<float> y_v2(num_points_in_linspace);
        std::vector<float> z_v2(num_points_in_linspace);

        int degree = 2;
        fitpackpp::BSplineCurve_modified Bcurve = fitpackpp::BSplineCurve_modified();
        Bcurve.createSplineFromPoints(x_v, y_v, degree, 30);
        std::vector<double> coefs_ = Bcurve.coefs();
        fitpackpp::BSplineCurve_modified Bcurve2 = fitpackpp::BSplineCurve_modified();
        Bcurve2.createSplineFromPoints(x_v, z_v, degree, 30);

        std::vector<float> y_eval(y_v2.size());
        std::vector<float> z_eval(z_v2.size());

        for (int i = 0; i < t.size(); ++i)
        {
            y_eval[i] = Bcurve.eval(t[i]);
            z_eval[i] = Bcurve2.eval(t[i]);
        }
        Eigen::MatrixXf lane_dict_eval(t.size(), 3);
        for (size_t i = 0; i < t.size(); ++i)
        {
            lane_dict_eval(i, 0) = static_cast<float>(t[i]);
            lane_dict_eval(i, 1) = static_cast<float>(y_eval[i]);
            lane_dict_eval(i, 2) = static_cast<float>(z_eval[i]);
        }
        return lane_dict_eval;
    }

    void generate_coeffs(const Eigen::MatrixXf &points)
    {

        Eigen::VectorXf x = points.col(0);
        Eigen::VectorXf y = points.col(1);
        Eigen::VectorXf z = points.col(2);
        Eigen::VectorXf coeffs = polynomial_Fit_2nd_degree(points.rows(), x, y);
        tcks[frame_index].push_back(coeffs);
    }

    std::vector<std::pair<int, int>> connect_lanes(const std::vector<Eigen::MatrixXf> &lanes_splines_list, float tolerance = 12)
    {
        std::vector<std::pair<int, int>> connected_indices;
        Eigen::VectorXf coeffs;
        std::vector<float> coeffsJ;
        std::map<float, std::pair<int, int>> myMap;
        for (int i = 0; i < lanes_splines_list.size(); ++i)
        {
            for (int j = i + 1; j < lanes_splines_list.size(); ++j)
            {
                const Eigen::MatrixXf &lane1 = lanes_splines_list[i];
                const Eigen::MatrixXf &lane2 = lanes_splines_list[j];
                coeffs = tcks[frame_index][i];
                Eigen::VectorXf x = lane2.col(0);
                Eigen::VectorXf y = lane2.col(1);
                Eigen::VectorXf y1 = lane1.col(1);

                float a = coeffs(0);
                float b = coeffs(1);
                float c = coeffs(2);
                Eigen::VectorXf predicted_y = a * x.array().square() + b * x.array() + c;

                float sse = calculateSSE2(y, predicted_y);
                myMap[sse] = std::make_pair(i, j);
            }
        }
        auto it = myMap.begin();
        float minKey = myMap.begin()->first;
        while (it != myMap.end() && minKey < tolerance)
        {
            connected_indices.push_back(it->second);
            ++it;
            if (it != myMap.end())
            {
                minKey = it->first;
            }
        }

        return connected_indices;
    }

    void run()
    {
        for (int frame_idx = 0; frame_idx < 5; ++frame_idx)
        {
            std::cout << "Processing frame: " << frame_idx << std::endl;
            get_frame(std::to_string(frame_idx));
            process_lanes_from_mask();
        }
    }
};

class TrackManager
{
private:
    std::map<int, std::deque<int>> track_recent_visits;
    int inactive_threshold;
    std::vector<int> tracks_to_remove;

public:
    std::map<int, int> track_visits;
    TrackManager(int frame_rate = 7)
        : track_visits(), track_recent_visits(),
          inactive_threshold(frame_rate), tracks_to_remove() {}

    void create_or_update_track(int track_id)
    {
        if (track_visits.find(track_id) == track_visits.end())
        {
            track_visits[track_id] = 0;
            track_recent_visits[track_id] = std::deque<int>();
        }
        track_visits[track_id] += 1;
        track_recent_visits[track_id].push_back(1);

        if (track_recent_visits[track_id].size() > inactive_threshold)
        {
            track_recent_visits[track_id].pop_front();
        }
    }

    void remove_inactive_tracks()
    {
        for (auto &visit : track_visits)
        {
            int track_id = visit.first;
            if (track_recent_visits[track_id].size() < inactive_threshold)
            {
                tracks_to_remove.push_back(track_id);
            }
        }
        std::unordered_set<int> a(tracks_to_remove.begin(), tracks_to_remove.end());
        for (auto track_id : a)
        {
            if (track_visits.find(track_id) != track_visits.end())
            {
                track_visits.erase(track_id);
            }
            if (track_recent_visits.find(track_id) != track_recent_visits.end())
            {
                track_recent_visits.erase(track_id);
            }
        }
    }
    std::vector<int> get_active_tracks()
    {
        std::vector<int> vec;
        for (const auto &elem : track_visits)
        {
            vec.push_back(elem.first);
        }
        return vec;
    }

    std::vector<int> get_dropped_tracks()
    {
        return tracks_to_remove;
    }

    std::vector<Eigen::MatrixXf> concatenate_lanes(
        const std::map<std::string, std::vector<std::pair<int, int>>> &connected_lanes,
        std::string frame_idx,
        const std::vector<Eigen::MatrixXf> &road_points)
    {
        std::vector<Eigen::MatrixXf> new_road_points;
        std::unordered_set<int> processed_indices;
        auto &indices_ = connected_lanes.at(frame_idx);
        std::vector<Eigen::MatrixXf> concatenated_array;
        for (auto indices : indices_)
        {
            int firstValue = indices.first;
            int secondValue = indices.second;

            Eigen::MatrixXf concatenated = Eigen::MatrixXf(road_points[firstValue].rows() + road_points[secondValue].rows(), road_points[firstValue].cols());
            concatenated << road_points[firstValue], road_points[secondValue];
            new_road_points.push_back(concatenated);
            processed_indices.insert(firstValue);
            processed_indices.insert(secondValue);
        }

        for (int idx = 0; idx < road_points.size(); ++idx)
        {
            if (processed_indices.find(idx) == processed_indices.end())
            {
                new_road_points.push_back(road_points[idx]);
            }
        }
        return new_road_points;
    }

    float curves_intersect(Eigen::MatrixXf &curve1_points, Eigen::MatrixXf &curve2_points)
    {

        auto sortByY = [&](Eigen::MatrixXf &points)
        {
            std::vector<int> indices(points.rows());
            std::iota(indices.begin(), indices.end(), 0);
            std::sort(indices.begin(), indices.end(), [&](int i, int j)
                      { return points(i, 0) < points(j, 0); });
            Eigen::MatrixXf sorted_points(points.rows(), points.cols());
            for (size_t i = 0; i < indices.size(); ++i)
            {
                sorted_points.row(i) = points.row(indices[i]);
            }
            return sorted_points;
        };

        curve1_points = sortByY(curve1_points);
        curve2_points = sortByY(curve2_points);

        size_t length = std::min(curve1_points.rows(), curve2_points.rows());
        Eigen::VectorXf x1 = Eigen::VectorXf::Zero(length);
        Eigen::VectorXf y1 = Eigen::VectorXf::Zero(length);

        for (int i = 0; i < length; ++i)
        {
            x1[i] = curve1_points(i, 0);
            y1[i] = curve1_points(i, 1);
        }

        Eigen::VectorXf x2 = Eigen::VectorXf::Zero(length);
        Eigen::VectorXf y2 = Eigen::VectorXf::Zero(length);

        for (int i = 0; i < length; ++i)
        {
            x2[i] = curve2_points(i, 0);
            y2[i] = curve2_points(i, 1);
        }

        Eigen::VectorXf coeffs1 = polynomial_Fit_2nd_degree(x1.size(), x1, y1);
        Eigen::VectorXf coeffs2 = polynomial_Fit_2nd_degree(x2.size(), x2, y2);

        Eigen::VectorXf predicted_y1 = coeffs2(0) * x1.array().square() + coeffs2(1) * x1.array() + coeffs2(2);
        Eigen::VectorXf predicted_y2 = coeffs1(0) * x2.array().square() + coeffs1(1) * x2.array() + coeffs1(2);

        float sse1 = calculateSSE(y1, predicted_y1);
        float sse2 = calculateSSE(y2, predicted_y2);
        return std::min(sse1, sse2);
    }
};

class LaneTracker
{
private:
    WaymoDatasetHandler lane_processor;
    TrackManager track_manager;
    std::map<int, Eigen::MatrixXf> moving_clusters;
    int frame_rate;

public:
    std::map<std::string, std::map<int, Eigen::MatrixXf>, NumericStringCompare> frames_object_ids{};

    std::map<int, Eigen::MatrixXf> prev_moving_clusters;
    std::vector<int> tracking_ids = {};
    std::vector<int> tracking_ids_copy = {};

    LaneTracker(TrackManager track_manager, WaymoDatasetHandler lane_processor, int frame_rate = 7) : track_manager(track_manager), lane_processor(lane_processor), frame_rate(frame_rate) {}

    void trackingStates(bool visualize = false)
    {

        float threshold = .2;
        int j = 0;
        for (auto it = lane_processor.lanes_labels_per_frame.begin(); it != lane_processor.lanes_labels_per_frame.end(); ++it, ++j)
        {
            std::string frame_idx = it->first;
            std::vector<Eigen::MatrixXf> road_points = it->second;
            if (road_points.empty())
            {
                continue;
            }

            std::vector<Eigen::MatrixXf> clusters = track_manager.concatenate_lanes(lane_processor.connected_lanes, frame_idx, road_points);
            std::vector<int> labels;
            for (int i = 0; i < clusters.size(); ++i)
            {
                labels.push_back(i);
            }

            if (j == 0)
            {
                for (int label : labels)
                {
                    Eigen::MatrixXf cluster = clusters[label];
                    frames_object_ids[frame_idx][label] = cluster;
                    auto tracking_id = label;
                    track_manager.create_or_update_track(tracking_id);
                    tracking_ids.push_back(tracking_id);
                    moving_clusters[tracking_id] = cluster;
                    prev_moving_clusters[tracking_id] = moving_clusters[tracking_id];
                    track_manager.track_visits[tracking_id] = 1;
                }
            }
            else
            {
                tracking_ids_copy.clear();
                tracking_ids_copy = tracking_ids;

                for (auto &cluster : clusters)
                {
                    std::vector<float> distances = {};
                    for (auto k : tracking_ids_copy)
                    {
                        distances.push_back(track_manager.curves_intersect(prev_moving_clusters[k], cluster));
                    }
                    auto min_distance_it = std::min_element(distances.begin(), distances.end());
                    float distance = *min_distance_it;
                    int id_min_dis = std::distance(distances.begin(), min_distance_it);
                    int tracking_id;
                    if (distance < threshold)
                    {
                        tracking_id = tracking_ids_copy[id_min_dis];
                    }
                    else
                    {
                        auto max = std::max_element(tracking_ids_copy.begin(), tracking_ids_copy.end());
                        tracking_id = *max + 1;
                        std::vector<int> dropped_tracks = track_manager.get_dropped_tracks();
                        if (!dropped_tracks.empty())
                        {
                            if (std::find(dropped_tracks.begin(), dropped_tracks.end(), tracking_id) != dropped_tracks.end())
                            {
                                auto max = std::max_element(dropped_tracks.begin(), dropped_tracks.end());
                                tracking_id = *max + 1;
                            }
                        }
                    }
                    track_manager.create_or_update_track(tracking_id);
                    moving_clusters[tracking_id] = cluster;
                    frames_object_ids[frame_idx][tracking_id] = cluster;

                    if (track_manager.track_visits.find(tracking_id) == track_manager.track_visits.end())
                    {
                        track_manager.track_visits[tracking_id] = 0;
                    }
                    track_manager.track_visits[tracking_id] += 1;
                }
                for (const auto &cluster : moving_clusters)
                {
                    int track_id = cluster.first;
                    prev_moving_clusters[track_id] = moving_clusters[track_id];
                }

                tracking_ids.clear();
                tracking_ids = track_manager.get_active_tracks();

                if (j % frame_rate == 0)
                {
                    track_manager.remove_inactive_tracks();
                }
            }
        }

        if (visualize)
        {
            std::vector<cv::Scalar> colors = {
                cv::Scalar(255, 0, 0), cv::Scalar(0, 128, 0), cv::Scalar(0, 0, 255), cv::Scalar(0, 255, 255), cv::Scalar(255, 0, 255), cv::Scalar(255, 255, 0),
                cv::Scalar(255, 165, 0), cv::Scalar(128, 0, 128), cv::Scalar(255, 192, 203), cv::Scalar(165, 42, 42), cv::Scalar(128, 128, 128), cv::Scalar(128, 128, 0),
                cv::Scalar(128, 0, 0), cv::Scalar(0, 128, 128), cv::Scalar(0, 255, 0), cv::Scalar(75, 0, 130), cv::Scalar(255, 215, 0), cv::Scalar(255, 127, 80),
                cv::Scalar(64, 224, 208), cv::Scalar(238, 130, 238), cv::Scalar(0, 0, 139), cv::Scalar(0, 100, 0), cv::Scalar(139, 0, 0), cv::Scalar(255, 140, 0),
                cv::Scalar(148, 0, 211), cv::Scalar(240, 230, 140), cv::Scalar(173, 216, 230), cv::Scalar(144, 238, 144), cv::Scalar(250, 128, 114), cv::Scalar(205, 133, 63),
                cv::Scalar(46, 139, 87), cv::Scalar(221, 160, 221), cv::Scalar(218, 112, 214), cv::Scalar(160, 82, 45), cv::Scalar(210, 105, 30), cv::Scalar(220, 20, 60),
                cv::Scalar(189, 183, 107), cv::Scalar(240, 128, 128), cv::Scalar(255, 218, 185)};

            for (const auto &frame : frames_object_ids)
            {
                std::string frame_idx = frame.first;
                std::cout << "frame_idx: " << frame_idx << std::endl;
                lane_processor.get_frame(frame_idx);
                cv::Mat _image = lane_processor.front_camera_image.clone();
                for (const auto &track : frame.second)
                {
                    int tracking_id = track.first;
                    const Eigen::MatrixXf &lane_spline = track.second;

                    std::vector<PointXYZIR> lane_spline_vector = convertMatrixToPoints(lane_spline);
                    Eigen::MatrixXf points_in_cam = lane_processor.transformation(lane_spline_vector, lane_processor.ego_to_front_camera_trans);
                    points_in_cam = lane_processor.convert_waymo_camera_points_to_standard_camera_coordinates(points_in_cam);
                    Eigen::MatrixXf points_in_image = lane_processor.project_cam_to_image(points_in_cam, lane_processor.front_camera_intrinsic);

                    cv::Scalar color = colors[tracking_id % colors.size()];
                    for (int i = 0; i < points_in_image.rows(); ++i)
                    {
                        int x = static_cast<int>(points_in_image(i, 0));
                        int y = static_cast<int>(points_in_image(i, 1));

                        cv::circle(_image, cv::Point(x, y), 10, color, -1);
                    }

                    if (points_in_image.rows() > 0)
                    {
                        float sum_x = 0.0f, sum_y = 0.0f;
                        for (int i = 0; i < points_in_image.rows(); ++i)
                        {
                            sum_x += points_in_image(i, 0);
                            sum_y += points_in_image(i, 1);
                        }

                        int center_x = static_cast<int>(sum_x / points_in_image.rows());
                        int center_y = static_cast<int>(sum_y / points_in_image.rows());

                        int text_x = center_x + 10;
                        int text_y = center_y - 10;

                        cv::putText(_image, std::to_string(tracking_id), cv::Point(text_x, text_y), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 0, 0), 3);
                    }
                }

                cv::imwrite("lanes_after_tracking/lanes_" + frame_idx + ".jpg", _image);
            }
        }
    }
};

int main()
{

    int frame_rate = 7;
    WaymoDatasetHandler lane_processor = WaymoDatasetHandler();
    TrackManager track_manager = TrackManager(frame_rate);
    lane_processor.run();
    LaneTracker lane_tracker = LaneTracker(track_manager, lane_processor, frame_rate);

    std::cout << "Starting lane tracking..." << std::endl;
    lane_tracker.trackingStates();

    return 0;
}
