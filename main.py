from utils import read_video, save_video
from trackers import Tracker
from team_assigner import TeamAssigner
from player_ball_assigner import PlayerBallAssigner

def main():
    # read video
    video_frames = read_video("./input_file/08fd33_4.mp4")

    # Initialize tracker
    tracker = Tracker("./models/best.pt")

    # Get object tracks
    object_tracks = tracker.get_object_tracks(video_frames,
                                            read_from_stub=True,
                                            stub_path="./stubs/track_stubs.pkl")

    # interpolate ball positions
    object_tracks['ball'] = tracker.interpolate_ball_positions(object_tracks['ball'])

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

    # Assign ball aqusistion 
    player_assigner = PlayerBallAssigner()
    for frame_num, player_track in enumerate(object_tracks['players']):
        ball_dict = object_tracks['ball'][frame_num].get(1, {})
        ball_bbox = ball_dict.get('bbox', None)
        if ball_bbox is None:
            continue  # Skip frames where ball isn't detected
        assigned_player = player_assigner.assign_ball_to_player(player_track,ball_bbox)

        if assigned_player != -1:
            object_tracks['players'][frame_num][assigned_player]['has_ball'] = True
            


    # Draw output
    output_video_frames = tracker.draw_annotations(video_frames, object_tracks)

    # save video
    save_video(output_video_frames, "./output_videos/output_video.avi")
    

if __name__ == "__main__":
    main()