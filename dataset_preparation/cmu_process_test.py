from cmu_process import analyze_video
import cv2
import logging


logging.getLogger().setLevel(logging.DEBUG)

# Dictionary mapping video paths to their test ranges
TEST_RANGES = {
    "/data/cmu/pose_dataset/171026_pose1/hdVideos/hd_00_10.mp4": [
        ("1:02", "1:22", "green_screen"),
        ("3:40", "3:50", "stationary"), # person is present, but not moving
        ("3:50", "4:00", None),
        ("4:09", "4:13", "stationary"),
        ("4:26", "4:30", None),
        ("1:03", "1:07", "green_screen"),
        ("12:00", "12:04", "stationary"),
        ("13:00", "13:10", "stationary"),
    ],
    "/data/cmu/multi_human_dataset/160224_mafia1/hdVideos/hd_00_10.mp4": [
        ("0:00", "0:20", "stationary"),
        ("0:30", "0:40", None),
        ("7:40", "8:16", "stationary"),
    ]
}

def timecode_to_seconds(timecode):
    """Convert mm:ss format to seconds."""
    minutes, seconds = map(int, timecode.split(":"))
    return minutes * 60 + seconds

def seconds_to_timecode(seconds):
    """Convert seconds to mm:ss format."""
    minutes = int(seconds // 60)
    seconds = int(seconds % 60)
    return f"{minutes}:{seconds:02d}"

def test_frame_range(video_path, start_timecode, end_timecode, expected_type):
    """Test a specific frame range for frame sequence detection.

    Args:
        video_path (str): Path to the video file
        start_timecode (str): Start time in mm:ss format
        end_timecode (str): End time in mm:ss format
        expected_type (str): Expected sequence type ('stationary', 'green_screen', 'black_screen', or None)
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    # Convert timecode to seconds and then to frames
    start_time = timecode_to_seconds(start_timecode)
    end_time = timecode_to_seconds(end_timecode)
    start_frame = int(start_time * fps)
    end_frame = int(end_time * fps)

    print(f"\nAnalyzing frames {start_frame} to {end_frame}")
    print(f"Time: {start_timecode} to {end_timecode}")
    print(f"Expected: {expected_type if expected_type else 'Normal sequence'}")

    # Analyze video for the specific frame range
    analysis_results = analyze_video(
        video_path,
        event_name="test",
        cache_dir=None,
        start_frame=start_frame,
        end_frame=end_frame
    )

    total_frames = end_frame - start_frame

    # Calculate ratios for each type of sequence
    sequence_ratios = {}
    for seq_type in ['stationary', 'green_screen', 'black_screen']:
        sequences = analysis_results[seq_type]
        total_type_frames = 0

        for start, end in sequences:
            # Clip the sequence to our range of interest
            sequence_start = max(start, start_frame)
            sequence_end = min(end, end_frame)
            if sequence_end > sequence_start:  # Only count if there's overlap
                total_type_frames += sequence_end - sequence_start

        ratio = total_type_frames / total_frames if total_frames > 0 else 0
        sequence_ratios[seq_type] = ratio

    print("\nResults:")
    print(f"Total frames analyzed: {total_frames}")
    for seq_type, ratio in sequence_ratios.items():
        print(f"{seq_type.title()} ratio: {ratio*100:.1f}%")

    # Print detected sequences
    for seq_type in ['stationary', 'green_screen', 'black_screen']:
        sequences = analysis_results[seq_type]
        if sequences:
            print(f"\nDetected {seq_type} sequences:")
            for start, end in sequences:
                if end > start_frame and start < end_frame:  # Only show relevant sequences
                    start_tc = seconds_to_timecode(start/fps)
                    end_tc = seconds_to_timecode(end/fps)
                    print(f"Frames {start}-{end} ({start_tc}-{end_tc})")
                    print(f"Length: {end-start} frames ({(end-start)/fps:.2f}s)")

    # Evaluate accuracy of detection
    # Different thresholds for different sequence types
    is_correct = False
    detected_type = None

    stationary_ratio = 0.7
    
    if expected_type == 'stationary':
        # For stationary, >50% frames is considered correct
        if sequence_ratios['stationary'] > stationary_ratio:
            detected_type = 'stationary'
            is_correct = True
    elif expected_type in ['green_screen', 'black_screen']:
        # For green/black screen, any detection (>0%) is considered correct
        if sequence_ratios[expected_type] > 0:
            detected_type = expected_type
            is_correct = True
    else:
        # For normal sequences (expected_type is None)
        # Consider it correct only if no green/black screen is detected at all
        # and stationary is not dominant
        has_special_sequence = (sequence_ratios['green_screen'] > 0 or 
                              sequence_ratios['black_screen'] > 0)
        has_dominant_stationary = sequence_ratios['stationary'] > stationary_ratio
        
        is_correct = not (has_special_sequence or has_dominant_stationary)
        detected_type = None

        if has_special_sequence:
            # Determine which special sequence was detected
            if sequence_ratios['green_screen'] > 0:
                detected_type = 'green_screen'
            elif sequence_ratios['black_screen'] > 0:
                detected_type = 'black_screen'
        elif has_dominant_stationary:
            detected_type = 'stationary'

    print(
        f"\nDetection result: {detected_type.upper() if detected_type else 'NORMAL SEQUENCE'}")
    print(f"Accuracy: {'✓ CORRECT' if is_correct else '✗ INCORRECT'}")

    return analysis_results


if __name__ == "__main__":
    for video_path, ranges in TEST_RANGES.items():
        print(f"\nTesting video: {video_path}")
        print("="*50)
        
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()

        for start_tc, end_tc, expected_type in ranges:
            results = test_frame_range(video_path, start_tc, end_tc, expected_type)
            print("\n" + "="*50 + "\n")
