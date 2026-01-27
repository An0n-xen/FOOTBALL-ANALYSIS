from ultralytics import YOLO
import supervision as sv
import cv2
import pickle
import os
import sys
sys.path.append('../')
from utils import get_center_of_bbox, get_bbox_width

class Tracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()
    
    def detect_frames(self, frames):
        batch_size=20
        detections = []

        for i in range(0, len(frames), batch_size):
            detections_batch = self.model.predict(frames[i:i+batch_size], conf=0.1)
            detections.extend(detections_batch)

        return detections

    def get_object_tracks(self, frames, read_from_stub=False, stub_path=None):
        if read_from_stub and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                return pickle.load(f)


        detections = self.detect_frames(frames)

        tracks = {
            'players': [],
            'ball': [],
            'referees': []
        }

        for frame_num, detections in enumerate(detections):
            cls_names =  detections.names
            cls_names_inv = {v:k for k,v in cls_names.items()}
            
            # convert to supervision format
            detections_supervision = sv.Detections.from_ultralytics(detections)

            # conver Goalkeeper to player
            for object_ind, class_id in enumerate(detections_supervision.class_id):
                if cls_names[class_id] == "goalkeeper":
                    try:
                        detections_supervision.class_id[object_ind] = cls_names_inv['player']
                    except:
                        print("KeyError: 'player'")
                        print(cls_names_inv)

            detections_with_tracks = self.tracker.update_with_detections(detections_supervision)

            tracks['players'].append({})
            tracks['ball'].append({})
            tracks['referees'].append({})

            for frame_detection in detections_with_tracks:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                track_id = frame_detection[4]

                if cls_id == cls_names_inv['player']:
                    tracks['players'][frame_num][track_id] = {"bbox":bbox}
                
                if cls_id == cls_names_inv['referee']:
                    tracks['referees'][frame_num][track_id] = {"bbox":bbox}

            for frame_detection in detections_supervision:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                
                if cls_id == cls_names_inv['ball']:
                    tracks['ball'][frame_num][track_id] = {"bbox":bbox}

        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(tracks, f)

        return tracks

    def draw_ellipse(self, frame, bbox, color, track_id=None):
        y2 = int(bbox[3])
        
        x_center, _ = get_center_of_bbox(bbox)
        width = get_bbox_width(bbox)

        cv2.ellipse(
            frame, 
            center=(x_center, y2), 
            axes=(int(width), int(0.35*width)), 
            angle=0, 
            startAngle=-45, 
            endAngle=235, 
            color=color, 
            thickness=2,
            lineType=cv2.LINE_4
        )

        rectangle_width = 40
        rectangle_height = 20
        x1_rect = x_center - rectangle_width//2
        x2_rect = x_center + rectangle_width//2
        y1_rect = (y2 - rectangle_height // 2) + 15
        y2_rect = (y2 + rectangle_height // 2) + 15

        if track_id is not None:
            cv2.rectangle(
                frame,
                (x1_rect, y1_rect),
                (x2_rect, y2_rect),
                color,
                cv2.FILLED,
            )

            x1_text = x1_rect + 12
            if track_id > 99:
                x1_text -= 10

            cv2.putText(
                frame,
                str(track_id),
                (x1_text, y1_rect + 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                2,
                cv2.LINE_4
            )
            
        return frame
        
    def draw_annotations(self, frames, tracks):
        output_video_frames = []
        for frame_num, frame in enumerate(frames):
            frame = frame.copy()
            
            player_dict = tracks['players'][frame_num]
            ball_dict = tracks['ball'][frame_num]
            referee_dict = tracks['referees'][frame_num]

            # Draw Players
            for track_id, player in player_dict.items():
                frame = self.draw_ellipse(frame, player['bbox'], (0, 0, 255), track_id)
            
            # Draw referee
            for track_id, referee in referee_dict.items():
                frame = self.draw_ellipse(frame, referee['bbox'], (0, 255, 0), track_id)

            output_video_frames.append(frame)

        return output_video_frames
            
