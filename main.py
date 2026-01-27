from utils import read_video, save_video
from trackers import Tracker
from team_assigner import TeamAssigner

def main():
    # read video
    video_frames = read_video("./input_file/08fd33_4.mp4")

    # Initialize tracker
    tracker = Tracker("./models/best.pt")

    # Get object tracks
    object_tracks = tracker.get_object_tracks(video_frames,
                                            read_from_stub=True,
                                            stub_path="./stubs/track_stubs.pkl")

    # Assign Player Teams
    team_assigner = TeamAssigner()
    team_assigner.assign_team_color(video_frames[0], object_tracks['players'][0])
    
    for frame_num, player_track in enumerate(object_tracks['players']):
        for player_id, track in player_track.items():
            team = team_assigner.get_player_team(video_frames[frame_num],   
                                                 track['bbox'],
                                                 player_id)
            object_tracks['players'][frame_num][player_id]['team'] = team 
            object_tracks['players'][frame_num][player_id]['team_color'] = team_assigner.team_colors[team]

    # Draw output
    output_video_frames = tracker.draw_annotations(video_frames, object_tracks)

    # save video
    save_video(output_video_frames, "./output_videos/output_video.avi")
    

if __name__ == "__main__":
    main()