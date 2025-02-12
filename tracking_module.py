import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import json
from scipy.spatial.distance import pdist
from scipy.interpolate import splprep, splev
from waymo_open_dataset.utils import frame_utils
from waymo_open_dataset import dataset_pb2 as open_dataset
import tensorflow 
from collections import deque
import copy

class WaymoDatasetHandler:
    def __init__(self):
        
        self.tfrecord_file_path  = "waymo/waymo_dataset/segment-15832924468527961_1564_160_1584_160_with_camera_labels.tfrecord"
        self.waymo_dataset = tensorflow.data.TFRecordDataset(self.tfrecord_file_path, compression_type='')
        self.current_frame = open_dataset.Frame()

        self.openlane_labels_dir = "waymo/waymo_dataset/segment-15832924468527961_1564_160_1584_160_with_camera_labels"
        self.openlane_labels_files_list = os.listdir(self.openlane_labels_dir)
        self.openlane_labels_files_array = np.array(self.openlane_labels_files_list)
        self.openlane_labels_files_array_sorted = np.sort(self.openlane_labels_files_array)

        self.front_camera_segmentation_masks_dir = "waymo/waymo_dataset/segment-15832924468527961_1564_160_1584_160_with_camera_labels_front_camera_masks"
        front_camera_segmentation_masks_files_list = os.listdir(self.front_camera_segmentation_masks_dir)
        front_camera_segmentation_masks_files_array = np.array(front_camera_segmentation_masks_files_list)
        self.front_camera_segmentation_masks_files_sorted_array = np.sort(front_camera_segmentation_masks_files_array)


        self.lidar_to_ego_trans = np.array([[-0.85282908, -0.52218559, -0.00218336 , 1.43      ],
                                            [ 0.52218959, -0.85281457, -0.00503366 , 0.        ],
                                            [ 0.00076651, -0.00543298,  0.99998495 , 2.184     ],
                                            [ 0.        ,  0.        ,  0.         , 1.        ]])


        self.front_camera_to_ego_trans = np.array([[ 0.99968703, -0.00304848 ,-0.02483053 , 1.54436336],
                                                   [ 0.00353814,  0.99979967 , 0.01970015 ,-0.02267064],
                                                   [ 0.0247655 , -0.01978184 , 0.99949755 , 2.11579855],
                                                   [ 0.        ,  0.         , 0.         , 1.        ]])


        self.front_camera_intrinsic = np.array([[2084.21565711,    0.        ,  982.69940656],
                                                [   0.        , 2084.21565711,  647.34228763],
                                                [   0.        ,    0.        ,    1.        ]])


        self.ego_to_front_camera_trans = np.linalg.inv(self.front_camera_to_ego_trans)

        self.image_height = 0 
        self.image_width = 0 
        self.lanes_labels = None 
        self.lanes_labels_per_frame = {} 
        self.front_camera_image = None 
        self.front_camera_mask = None 
        self.front_camera_depthmap = None 
        self.lidars_pointcloud = None 
        self.pointcloud_in_camera = None 
        self.debug_image = None
        self.frame_index = 0
        self.rs = None 
        self.min_distance = 10
        self.min_angle = 5
        self.connected_lanes = {}
        self.lanes_labels_per_frame = {}

    def get_frame(self, frame_index):

        current_dataset = self.waymo_dataset.skip(frame_index).take(1)  
        for frame_data in current_dataset:
            self.current_frame.ParseFromString(bytearray(frame_data.numpy()))

        range_images, camera_projections, _, range_image_top_pose = frame_utils.parse_range_image_and_camera_projection(self.current_frame)
        points, _ = frame_utils.convert_range_image_to_point_cloud(self.current_frame,range_images,camera_projections,range_image_top_pose)
        self.lidars_pointcloud  = np.concatenate(points, axis=0)
        front_image_waymo = self.current_frame.images[0] #0 -> front camera
        front_image_np = np.frombuffer(front_image_waymo.image, dtype=np.uint8)
        self.front_camera_image = cv2.imdecode(front_image_np, cv2.IMREAD_COLOR)

        self.image_height = self.front_camera_image.shape[0]
        self.image_width  = self.front_camera_image.shape[1] 

        openlane_label_file_path = self.openlane_labels_files_array_sorted[frame_index]
        openlane_label_file = open(self.openlane_labels_dir + "/" + openlane_label_file_path)
        self.lanes_labels = json.load(openlane_label_file)

        self.front_camera_mask = cv2.imread(self.front_camera_segmentation_masks_dir  + "/" + self.front_camera_segmentation_masks_files_sorted_array[frame_index])
        self.front_camera_depthmap = self.get_depth_map(self.lidars_pointcloud)

    def get_depth_map(self, points_in_ego):
                
        ############################# Transform Points  ########################
        # print(points_in_ego.shape)
        # for y in range(100):
        #     print(points_in_ego[y])
        self.pointcloud_in_camera = self.transformation(points_in_ego,self.ego_to_front_camera_trans)[:,:3]
        # print(self.pointcloud_in_camera.shape)
        # for y in range(100):
        #     print(self.pointcloud_in_camera[y])
        points_in_cam = self.convert_waymo_camera_ponts_to_standard_camera_coordinates(self.pointcloud_in_camera)
        points_in_image = self.project_cam_to_image(points_in_cam,self.front_camera_intrinsic).astype(int)


        ############################# Filter Points  ########################
        pixels_positions = np.around(points_in_image)
        valid_positions_mask = np.logical_and( points_in_cam[:, 2] > 0,np.logical_and(
        np.logical_and(pixels_positions[:, 0] > 0, pixels_positions[:, 0] <  self.image_width),
        np.logical_and(pixels_positions[:, 1] > 0, pixels_positions[:, 1] <  self.image_height)))

        ############################# Create Depth Map Image  ########################
        valid_pixel_depths = points_in_cam[valid_positions_mask, 2]  
        valid_pixel_positions = pixels_positions[valid_positions_mask].astype(int)
        depthmap_image = np.zeros(( self.image_height,  self.image_width))
        depthmap_image[valid_pixel_positions[:, 1], valid_pixel_positions[:, 0]] = valid_pixel_depths 

        return depthmap_image
    
    def cart2hom(self,pts_3d):
        n = pts_3d.shape[0]
        pts_3d_hom = np.hstack((pts_3d, np.ones((n,1))))
        return pts_3d_hom
    
    def transformation(self,pts_3d,trans):
        pts_3d = self.cart2hom(pts_3d) 
        return np.dot(pts_3d, np.transpose(trans))
    
    def project_cam_to_image(self,pts_3d_rect,cam_intrinsic):
    
        pts_2d = np.dot(pts_3d_rect, np.transpose(cam_intrinsic)) 
        pts_2d[:,0] /= pts_2d[:,2]
        pts_2d[:,1] /= pts_2d[:,2]
        return pts_2d[:,0:2]
    
    def convert_waymo_camera_ponts_to_standard_camera_coordinates(self,pts_3d_rect):
        # Waymo (OpenLane) camera coord sys. x-front, y-left, z-up
        # normal (aka. standard) camera coord sys widely used. x-right, y-down, z-front
        pts_3d_rect = pts_3d_rect[:,[1,2,0]]
        pts_3d_rect[:,0] = - pts_3d_rect[:,0]
        pts_3d_rect[:,1] = - pts_3d_rect[:,1]
        return pts_3d_rect

    def run(self):
        for frame_idx in range(197):
            print("frame_idx: ",frame_idx)
            self.frame_index = frame_idx
            self.get_frame(frame_idx)
            self.process_lanes_from_mask()

    def process_lanes_from_mask(self):

        ############################## Processing ###########################################
        lanes_binary_mask = cv2.inRange(self.front_camera_mask, (200, 200, 200), (255, 255, 255))
        lanes_contours, _ = cv2.findContours(lanes_binary_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        lanes_xyzc_points_in_ego = np.empty((0, 4))
        lane_contour_color_idx = 0
        lanes_dict = []
        
        # Create a random colormap to assign unique colors to each lane using 'tab20' colormap
        color_map = plt.cm.get_cmap("tab20", len(lanes_contours) + 10)
    
        for lane_contour_idx in range(len(lanes_contours)):
            lane_dict = {}
            black_image = np.zeros_like(lanes_binary_mask)
            lane_contour_mask = cv2.fillPoly(black_image , pts=np.int32([lanes_contours[lane_contour_idx]]) , color=(255,255,255))
            lane_contour_depthmap = (lane_contour_mask * self.front_camera_depthmap)/255
            lane_contour_depthmap_valid_indices =  np.nonzero(lane_contour_depthmap)
            # print("lane_contour_depthmap_valid_indices: ",lane_contour_depthmap_valid_indices)
            # print(len(lane_contour_depthmap_valid_indices[0]))
            lane_contour_depths = self.front_camera_depthmap[lane_contour_depthmap_valid_indices]
            lane_contour_xyz_points = self.pointcloud_in_camera[np.isin(self.pointcloud_in_camera[:, 0], lane_contour_depths)]
            # print("lane_contour_xyz_points: ",lane_contour_xyz_points)
            lane_contour_xyz_points_in_ego = self.transformation(lane_contour_xyz_points,self.front_camera_to_ego_trans)[:,:3]
            if(lane_contour_xyz_points_in_ego.shape[0] != 0 ):
                lane_dict["xyz"] = lane_contour_xyz_points_in_ego
                lane_dict["contour_idx"] = lane_contour_color_idx
                lanes_dict.append(lane_dict)
                lane_contour_color_colum = np.zeros((lane_contour_xyz_points_in_ego.shape[0],1)) + (lane_contour_color_idx * 10 ) 
                lane_contour_color_idx = lane_contour_color_idx + 1
                lane_contour_xyzc_points_in_ego = np.concatenate((lane_contour_xyz_points_in_ego,lane_contour_color_colum), axis=1)
                lanes_xyzc_points_in_ego = np.concatenate((lanes_xyzc_points_in_ego,lane_contour_xyzc_points_in_ego),axis=0)


        lanes_splines_dicts = []
        lanes_splines = np.empty((0,4))
        lane_dict_idx = 0
        lanes_splines_list = []
        tcks =[]

        for lane_dict in lanes_dict:
            if(lane_dict["xyz"].shape[0] > 1):
                
                if(lane_dict["xyz"].shape[0] > 11):
                    # print("****************** lane line {}**********************".format(lane_dict_idx))

                    # Use the tab20 color map, adjusting the index to get a unique color
                    color = color_map(lane_dict_idx + 10)[:3]  # Extract RGB values from colormap (remove alpha)
                    color = tuple([int(c * 255) for c in color])  # Convert to 0-255 range for OpenCV

                    lane_spline_dict = lane_dict

                    pairwise_distances = pdist(lane_dict["xyz"], metric='euclidean')
                    max_distance = np.max(pairwise_distances)
                    t = np.linspace(0, 1, int(max_distance * 2))
                    tck, u = splprep([lane_dict["xyz"][:, 0], lane_dict["xyz"][:, 1], lane_dict["xyz"][:, 2]], k=2, s=10)
                    
                    lane_line_spline_knots = tck[0]
                    lane_line_spline_coefficients = np.array(tck[1]).transpose()

                    lane_spline_dict["spline_start_polynomial"] = lane_line_spline_coefficients[0:3,:]
                    lane_spline_dict["spline_end_polynomial"] = lane_line_spline_coefficients[lane_line_spline_coefficients.shape[0]-3:lane_line_spline_coefficients.shape[0],:]
                    x_spline, y_spline, z_spline = splev(t, tck)
                    coeffs = np.polyfit(x_spline, y_spline, 2) 
                    tcks.append(coeffs)
                    lane_spline_color = np.zeros((x_spline.shape[0])) + (lane_dict_idx * 10 )
                    lane_spline = np.vstack((x_spline,y_spline,z_spline,lane_spline_color)).transpose()
                    lane_spline_dict["lane_spline_xyz"] = lane_spline
                    lanes_splines = np.concatenate((lanes_splines,lane_spline),axis=0)
                    lane_dict_idx = lane_dict_idx + 1 
                    lanes_splines_dicts.append(lane_spline_dict)
                    if lane_spline.shape[0]>1:
                        # lanes_splines_list.append(lane_dict["xyz"][:,:3])
                        lanes_splines_list.append(lane_spline[:,:3])
                    
        print(self.frame_index)
        self.connected_lanes[self.frame_index] = self.connect_lanes(lanes_splines_list, tcks)
        self.lanes_labels_per_frame[self.frame_index] = lanes_splines_list


    def connect_lanes(self, lanes_splines_list, tcks, tolerance=6):
        
        connected_indices = []
        for i in range(len(lanes_splines_list)):
            for j in range(i + 1, len(lanes_splines_list)):
                x = lanes_splines_list[j][:, 0]
                y = lanes_splines_list[j][:, 1]
                a, b, c = tcks[i]
                predicted_y = a * x ** 2 + b * x + c
                residuals = y - predicted_y
                sse = np.sum(residuals ** 2)
                
                points1 = np.vstack((x, predicted_y))
                points2 = np.vstack((x, y))
                alignment = self.check_alignment_and_collinearity(points1.T, points2.T, 0.15)

                if alignment and sse < tolerance:
                    connected_indices.append((i, j))

        return connected_indices

    def check_alignment_and_collinearity(self, points1, points2, tolerance_dot):
        
        # Normalize vectors
        vector1 = points1[-1] - points1[-2] 
        vector2 = points2[1] - points2[0] 
        
        vector1 = vector1 / np.linalg.norm(vector1) if np.linalg.norm(vector1) != 0 else np.array([0, 0, 0])
        vector2 = vector2 / np.linalg.norm(vector2) if np.linalg.norm(vector2) != 0 else np.array([0, 0, 0])

        dot_product = np.dot(vector1, vector2)

        aligned = abs(dot_product) > 1 - tolerance_dot

        return aligned

class TrackManager:

    def __init__(self, frame_rate):
        self.track_visits = {}
        self.track_recent_visits = {}
        self.INACTIVE_THRESHOLD = frame_rate  # Stop appearing for more than 10 consecutive updates
        self.tracks_to_remove = []

    def create_or_update_track(self, track_id):
        if track_id not in self.track_visits:
            self.track_visits[track_id] = 0
            self.track_recent_visits[track_id] = deque(maxlen=self.INACTIVE_THRESHOLD)

        self.track_visits[track_id] += 1
        self.track_recent_visits[track_id].append(1)

    def remove_inactive_tracks(self):
        # tracks_to_remove = []
        for track_id in self.track_visits:
            if len(self.track_recent_visits[track_id]) < self.INACTIVE_THRESHOLD:
                self.tracks_to_remove.append(track_id)

        for track_id in set(self.tracks_to_remove):
            if track_id in self.track_visits:
                del self.track_visits[track_id]
            if track_id in self.track_recent_visits:
                del self.track_recent_visits[track_id]

    def get_active_tracks(self):
        return self.track_visits

    def get_dropped_tracks(self):
        return self.tracks_to_remove


    @staticmethod
    def curves_intersect(curve1_points, curve2_points,visualize=False):

        y_values = curve1_points[:, 0] 
        sorted_indices = np.argsort(y_values)
        curve1_points = curve1_points[sorted_indices]

        y_values = curve2_points[:, 0]
        sorted_indices = np.argsort(y_values)
        curve2_points = curve2_points[sorted_indices]

        length = curve1_points.shape[0] if curve1_points.shape[0]<curve2_points.shape[0] else curve2_points.shape[0]
        coeffs1 = np.polyfit(curve1_points[:length,0], curve1_points[:length,1], 2)
        coeffs2 = np.polyfit(curve2_points[:length,0], curve2_points[:length,1], 2)
        
        x = curve1_points[:length,0]
        y = curve1_points[:length,1]
        a, b, c = coeffs2
        predicted_y = a * x**2 + b * x + c
        residuals = y - predicted_y
        residuals = residuals/len(residuals)
        sse = np.sum(residuals**2)  # Sum of squared errors

        x2 = curve2_points[:length,0]
        y2 = curve2_points[:length,1]
        a, b, c = coeffs1
        predicted_y2 = a * x2**2 + b * x2 + c
        residuals2 = y2 - predicted_y2
        residuals2 = residuals2/len(residuals)
        sse2 = np.sum(residuals2**2)  # Sum of squared errors

        sse_r = sse if sse<sse2 else sse2
        if visualize:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            # print("-------------------------------------------")
            # print("sse: ",sse)

            ax.scatter(curve1_points[:,0], curve1_points[:,1], (curve1_points[:,2]*0)+1, c='blue', marker='^', label='Group 0 Min')

            ax.scatter(curve2_points[:,0], curve2_points[:,1], (curve2_points[:,2]*0)+1, c='red', marker='o', label='Group 1 Min')

            # Adding labels, title, and legend
            ax.set_xlabel('X Coordinate')
            ax.set_ylabel('Y Coordinate')
            ax.set_zlabel('Z Coordinate')
            ax.set_title('3D Scatter Plot of Points')
            ax.legend(loc='upper left')  # Show legend to help identify the groups

            # Show plot
            plt.show()
            print("-------------------------------------------")
        
        return sse_r

    @staticmethod
    def concatenate_lanes(connected_lanes, frame_id, road_points):
        new_road_points = []
        processed_indices = set() 

        for indices in connected_lanes[frame_id]:
            arrays_to_concat = [road_points[idx] for idx in indices]
            concatenated_array = np.concatenate(arrays_to_concat, axis=0)
            new_road_points.append(concatenated_array)
            processed_indices.update(indices)  

        for idx in range(len(road_points)):
            if idx not in processed_indices:
                new_road_points.append(road_points[idx])

        return new_road_points

class LaneTracker:
    def __init__(self, lane_processor, track_manager):
        self.lane_processor = lane_processor
        self.track_manager = track_manager

    def tracking(self):

        tracking_ids = []
        frames_object_ids = {}
        threshold = 0.06  # Threshold for distance to match lanes
        moving_clusters = {}
        prev_moving_clusters = {}
        self.frame_rate = 7  # This controls when to remove inactive tracks

        # for frame_id, lane_splines in self.lane_processor.lanes_labels_per_frame.items():
        #     lane_splines_serializable = [arr.tolist() for arr in lane_splines]
        #     filename = f"labels_json_folder/lanes_labels_frame_{frame_id}.json"
        #     with open(filename, 'w') as f:
        #         json.dump(lane_splines_serializable, f)

        #     print(f"Frame {frame_id} data saved to {filename}")
                        
        for j, frame_idx in enumerate(self.lane_processor.lanes_labels_per_frame):
            print("frame_idx tracking: ",frame_idx)
            road_points = self.lane_processor.lanes_labels_per_frame[frame_idx]

            if not road_points:
                continue

            clusters = TrackManager.concatenate_lanes(self.lane_processor.connected_lanes, frame_idx, road_points)
            labels = np.arange(len(clusters))  # Assign labels to clusters
            frames_object_ids[frame_idx] = {}
            
            if j == 0:
                # For the first frame, create new tracks
                for label, cluster in zip(labels, clusters):
                    frames_object_ids[frame_idx][label] = cluster
                    tracking_id = label
                    self.track_manager.create_or_update_track(tracking_id)  # Create or update track
                    tracking_ids.append(tracking_id)
                    moving_clusters[tracking_id] = cluster
                    prev_moving_clusters[tracking_id] = moving_clusters[tracking_id]
                    self.track_manager.track_visits[tracking_id] = 1  # Track has been visited once
            else:
                # For subsequent frames, update tracks based on proximity
                tracking_ids_copy = copy.deepcopy(tracking_ids)
                for c in range(len(clusters)):
                    # Calculate distances between current cluster and previous ones
                    distances = np.array([
                        TrackManager.curves_intersect(np.array(prev_moving_clusters[k]).reshape(-1, 3), clusters[c]) 
                        for k in tracking_ids_copy
                    ])
                    id_min_dis = np.argmin(distances)
                    distance = distances[id_min_dis]
                    print("frame_idx: ",frame_idx)
                    print("distances: ")
                    print(distances)
                    if distance < threshold:
                        tracking_id = tracking_ids_copy[id_min_dis]  # Assign the same tracking ID
                    else:
                        tracking_id = max(tracking_ids_copy) + 1  # Create a new tracking ID
                        if tracking_id in self.track_manager.get_dropped_tracks():
                            tracking_id = max(self.track_manager.get_dropped_tracks()) + 1
                    
                    self.track_manager.create_or_update_track(tracking_id)  # Create or update the track
                    moving_clusters[tracking_id] = clusters[c]
                    frames_object_ids[frame_idx][tracking_id] = clusters[c]
                    
                    # Update track visits
                    if tracking_id not in self.track_manager.track_visits:
                        self.track_manager.track_visits[tracking_id] = 0
                    self.track_manager.track_visits[tracking_id] += 1

                # Update tracking ids
                tracking_ids = list(self.track_manager.track_visits.keys())
                # Update previous moving clusters
                for track_id in moving_clusters:
                    prev_moving_clusters[track_id] = moving_clusters[track_id]
            
            # Remove inactive tracks after a certain number of frames
            if j % self.frame_rate == 0:
                self.track_manager.remove_inactive_tracks()
            
        # Track objects count based on frame visits
        track_no_objects = {}
        colors = [
            (255, 0, 0), (0, 128, 0), (0, 0, 255), (0, 255, 255), (255, 0, 255), (255, 255, 0), 
            (255, 165, 0), (128, 0, 128), (255, 192, 203), (165, 42, 42), (128, 128, 128), (128, 128, 0), 
            (128, 0, 0), (0, 128, 128), (0, 255, 0), (75, 0, 130), (255, 215, 0), (255, 127, 80), 
            (64, 224, 208), (238, 130, 238), (0, 0, 139), (0, 100, 0), (139, 0, 0), (255, 140, 0), 
            (148, 0, 211), (240, 230, 140), (173, 216, 230), (144, 238, 144), (250, 128, 114), (205, 133, 63),
            (46, 139, 87), (221, 160, 221), (218, 112, 214), (160, 82, 45), (210, 105, 30), (220, 20, 60),
            (189, 183, 107), (240, 128, 128), (255, 218, 185)
        ]
        
        for track_id in range(len(prev_moving_clusters)):
            track_no_objects[track_id] = 0
            for frame_idx in frames_object_ids:
                if track_id in frames_object_ids[frame_idx]:
                    track_no_objects[track_id] += 1
            
        print("Track object counts:", track_no_objects)

        for frame_idx in frames_object_ids:
            self.lane_processor.get_frame(frame_idx)
            _image = np.copy(self.lane_processor.front_camera_image)
            for tracking_id in frames_object_ids[frame_idx]:
                lane_spline = frames_object_ids[frame_idx][tracking_id]
                points_in_cam = self.lane_processor.transformation(lane_spline, self.lane_processor.ego_to_front_camera_trans)[:, :3]
                points_in_cam = self.lane_processor.convert_waymo_camera_ponts_to_standard_camera_coordinates(points_in_cam)
                points_in_image = self.lane_processor.project_cam_to_image(points_in_cam, self.lane_processor.front_camera_intrinsic).astype(int)
                color = colors[tracking_id]
                radius = 10

                for point in points_in_image:
                    cv2.circle(_image, tuple(point), radius, color, -1)  

                if len(points_in_image) > 0:
                    center_point = (int(np.mean([p[0] for p in points_in_image])), 
                                    int(np.mean([p[1] for p in points_in_image])))
                    
                    if len(points_in_image) == 1:
                        cv2.putText(_image, "this lane is one point", center_point, cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 4, cv2.LINE_AA)
                            
                    cv2.putText(_image, str(tracking_id), center_point, cv2.FONT_HERSHEY_SIMPLEX, 1.2, (238, 130, 238), 4, cv2.LINE_AA)

            cv2.imwrite(f"lanes_after_tracking/lanes_{frame_idx}.jpeg", _image)
            
        return frames_object_ids, prev_moving_clusters

def main():
    
    lane_processor = WaymoDatasetHandler()
    track_manager = TrackManager(7)

    lane_processor.run()
    lane_tracker = LaneTracker(lane_processor, track_manager)

    print("Starting lane tracking...")
    lane_tracker.tracking()

if __name__ == "__main__":
    main()