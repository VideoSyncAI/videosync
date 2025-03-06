import os
import cv2
import pickle
import itertools
import numpy as np
from tqdm import tqdm
import multiprocessing
import signal
import sys
import random
import logging
import json
import time
from datetime import timedelta
import av

# Set random seed for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# Global pool reference to be used in signal handler
pool = None

# Set up logging for retries and errors
logging.basicConfig(filename="video_processing.log", level=logging.INFO)


def log_failed_task(video1_output_path, video2_output_path, reason):
    """Log a failed task with a reason for failure."""
    with open(FAILURE_LOG_FILE, 'a') as f:
        f.write(f"{video1_output_path},{video2_output_path},{reason}\n")
    logging.warning(
        f"Logged failed task: {video1_output_path}, {video2_output_path}, Reason: {reason}")


FAILURE_LOG_FILE = "failed_tasks.log"


def load_failed_tasks():
    """Load failed tasks and their reasons from the log file."""
    failed_tasks = {}
    if os.path.exists(FAILURE_LOG_FILE):
        with open(FAILURE_LOG_FILE, "r") as f:
            for line in f.readlines():
                video1_output_path, video2_output_path, reason = line.strip().split(",")
                failed_tasks[(video1_output_path, video2_output_path)] = reason
    return failed_tasks


def cleanup_videos(video1_output_path, video2_output_path):
    """Remove corrupted video files."""
    if os.path.exists(video1_output_path):
        os.remove(video1_output_path)
        logging.info(f"Deleted {video1_output_path}")
    if os.path.exists(video2_output_path):
        os.remove(video2_output_path)
        logging.info(f"Deleted {video2_output_path}")


def is_green_screen_frame(frame, green_threshold=0.8):
    """Detect if a frame is mostly green, indicating a green screen."""
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_green = np.array([35, 40, 40])
    upper_green = np.array([85, 255, 255])
    green_mask = cv2.inRange(hsv_frame, lower_green, upper_green)
    green_ratio = np.sum(green_mask > 0) / (frame.shape[0] * frame.shape[1])
    return green_ratio > green_threshold


def is_black_screen_frame(frame, black_threshold=0.8):
    """Detect if a frame is mostly black, indicating a black screen."""
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    black_mask = gray_frame < 50  # Pixel intensity threshold for black
    black_ratio = np.sum(black_mask) / (frame.shape[0] * frame.shape[1])
    return black_ratio > black_threshold


def are_frames_similar_motion(frame1, frame2, motion_threshold=0.15):
    """Compare two frames using optical flow to detect small motions.

    Args:
        frame1, frame2: Input frames to compare
        motion_threshold: Maximum average motion magnitude to consider frames similar
    Returns:
        bool: True if frames are similar (minimal motion), False otherwise
    """
    # Convert frames to grayscale
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    # Calculate optical flow using Farneback method
    flow = cv2.calcOpticalFlowFarneback(
        gray1, gray2,
        None,  # No initial flow
        pyr_scale=0.5,  # Pyramid scale between layers
        levels=3,  # Number of pyramid layers
        winsize=15,  # Averaging window size
        iterations=3,  # Number of iterations at each pyramid level
        poly_n=5,  # Size of pixel neighborhood
        poly_sigma=1.2,  # Standard deviation of Gaussian for polynomial expansion
        flags=0
    )

    # Calculate magnitude of motion vectors
    magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
    avg_magnitude = np.mean(magnitude)

    return avg_magnitude < motion_threshold


def process_video_pair(args, failed_tasks):
    video1_path, video2_path, start_frame, end_frame, offset, video1_output_path, video2_output_path = args

    # Check if this pair has already failed with a specific reason
    if (video1_output_path, video2_output_path) in failed_tasks:
        reason = failed_tasks[(video1_output_path, video2_output_path)]
        logging.info(
            f"Skipping previously failed task: {video1_output_path}, {video2_output_path}, Reason: {reason}")
        return False

    # Skip processing if output files already exist
    if os.path.exists(video1_output_path) and os.path.exists(video2_output_path):
        logging.info(
            f"Skipping already existing files: {video1_output_path} and {video2_output_path}")
        return True

    # Process the video pair without retries
    success, reason = extract_and_save_video_pair(
        video1_path, video2_path, start_frame, end_frame, offset,
        video1_output_path, video2_output_path
    )

    if success:
        return True  # Processing succeeded
    else:
        # Clean up and log the failure reason
        cleanup_videos(video1_output_path, video2_output_path)
        log_failed_task(video1_output_path, video2_output_path, reason)
        logging.error(
            f"Processing failed for {video1_output_path} and {video2_output_path}. Reason: {reason}")
        return False


def seconds_to_timecode(seconds):
    """Convert seconds to mm:ss format."""
    minutes = int(seconds // 60)
    seconds = int(seconds % 60)
    return f"{minutes}:{seconds:02d}"


def analyze_video(video_path, event_name, cache_dir=None, start_frame=None, end_frame=None):
    """Analyze a video and cache its problematic frame ranges using PyAV."""
    # Define frame skip values
    FRAME_SKIP = 2  # For green/black screen detection

    base_name = f"{event_name}_{os.path.basename(video_path)}" if event_name else os.path.basename(
        video_path)

    # Cache handling
    if cache_dir:
        frame_range_str = f"_{start_frame}_{end_frame}" if start_frame is not None and end_frame is not None else ""
        cache_file = os.path.join(
            cache_dir, f"{base_name}{frame_range_str}.json")
        if os.path.exists(cache_file):
            with open(cache_file, 'r') as f:
                return json.load(f)

    # Open the video using PyAV
    container = av.open(video_path)
    stream = container.streams.video[0]

    # Get video info
    video_total_frames = stream.frames
    start_frame = start_frame if start_frame is not None else 0
    end_frame = end_frame if end_frame is not None else video_total_frames
    total_frames = end_frame - start_frame

    problematic_ranges = {
        'stationary': [],
        'green_screen': [],
        'black_screen': []
    }

    # Define maximum expected gap between keyframes (e.g., 90 frames = 3 seconds at 30fps)
    MAX_KEYFRAME_GAP = 90

    try:
        previous_frame = None
        current_range = None
        start_time = time.time()
        last_pts = None
        last_frame_idx = None

        # Set stream to only decode keyframes
        stream.codec_context.skip_frame = 'NONKEY'

        # Convert frame number to time base units for seeking
        time_base = stream.time_base
        average_rate = stream.average_rate

        pts = int(start_frame / average_rate * (1 / time_base))

        try:
            container.seek(pts, stream=stream)
        except Exception as e:
            logging.error(
                f"Seek failed for {base_name} at frame {start_frame}: {e}")
            raise

        while True:
            try:
                frame = next(container.decode(video=0))
                frame_idx = int(frame.pts * time_base * average_rate)

                # Skip if frame is beyond our end point
                if frame_idx >= end_frame:
                    break

                # Skip if this is the same frame we processed before
                if last_pts is not None and frame.pts == last_pts:
                    continue

                # Skip if this frame is too close to the last processed frame
                if last_frame_idx is not None and frame_idx < last_frame_idx + FRAME_SKIP:
                    continue

                # Log warning if keyframe gap is too large
                if last_frame_idx is not None:
                    gap = frame_idx - last_frame_idx
                    if gap > MAX_KEYFRAME_GAP:
                        logging.warning(
                            f"Large keyframe gap detected in {base_name}: "
                            f"{gap} frames between {last_frame_idx} and {frame_idx}"
                        )

                last_pts = frame.pts
                last_frame_idx = frame_idx

                # Convert PyAV frame to numpy array and resize
                numpy_frame = frame.to_ndarray(format='bgr24')
                numpy_frame = cv2.resize(numpy_frame, (224, 224))

                logging.debug(
                    f"Processing frame {frame_idx} ({seconds_to_timecode(frame_idx / average_rate)})")

                # Check for green/black screens
                if is_green_screen_frame(numpy_frame):
                    logging.debug(
                        f"Green screen detected at frame {frame_idx} ({seconds_to_timecode(frame_idx / average_rate)})")
                    if current_range is None or current_range[0] != 'green_screen':
                        if current_range:
                            problematic_ranges[current_range[0]].append(
                                (current_range[1], frame_idx))
                        current_range = ('green_screen', frame_idx)

                elif is_black_screen_frame(numpy_frame):
                    logging.debug(
                        f"Black screen detected at frame {frame_idx} ({seconds_to_timecode(frame_idx / average_rate)})")
                    if current_range is None or current_range[0] != 'black_screen':
                        if current_range:
                            problematic_ranges[current_range[0]].append(
                                (current_range[1], frame_idx))
                        current_range = ('black_screen', frame_idx)

                # If frame is not green screen or black screen, close those ranges if they were open
                elif current_range and current_range[0] in ['green_screen', 'black_screen']:
                    problematic_ranges[current_range[0]].append(
                        (current_range[1], frame_idx))
                    current_range = None

                # Check for stationary frames
                elif previous_frame is not None:
                    previous_frame_temp = previous_frame  # Store the previous frame temporarily
                    previous_frame = numpy_frame  # Update for next iteration

                    if are_frames_similar_motion(previous_frame_temp, numpy_frame):
                        logging.debug(
                            f"Stationary frame detected at frame {frame_idx} ({seconds_to_timecode(frame_idx / average_rate)})")
                        if current_range is None or current_range[0] != 'stationary':
                            if current_range:
                                problematic_ranges[current_range[0]].append(
                                    (current_range[1], frame_idx))
                            current_range = ('stationary', frame_idx)
                    else:
                        if current_range and current_range[0] == 'stationary':
                            problematic_ranges[current_range[0]].append(
                                (current_range[1], frame_idx))
                            current_range = None
                else:
                    previous_frame = numpy_frame  # Set initial previous_frame

            except (StopIteration, av.error.EOFError):
                logging.info(
                    f"Reached end of file for {base_name} at frame {frame_idx}")
                break
            except Exception as e:
                logging.error(f"Error processing frame in {base_name}: {e}")
                raise

        # Handle final range
        if current_range:
            problematic_ranges[current_range[0]].append(
                (current_range[1], start_frame + total_frames))

        # Final progress update
        elapsed_time = time.time() - start_time
        print(
            f"Completed {base_name} "
            f"[{start_frame}-{end_frame}] "
            f"[{str(timedelta(seconds=int(elapsed_time)))}]"
        )
        sys.stdout.flush()

        # Convert frame numbers to timestamps and log results
        for problem_type, ranges in problematic_ranges.items():
            if ranges:
                logging.info(
                    f"Found {len(ranges)} {problem_type} sequences in {base_name}")

                # Convert each range to timestamps
                for start_frame, end_frame in ranges:
                    # Ensure we're working with numeric values
                    start_frame = float(start_frame)
                    end_frame = float(end_frame)
                    average_rate = float(average_rate)

                    start_seconds = int(start_frame / average_rate)
                    end_seconds = int(end_frame / average_rate)

                    logging.info(
                        f"  {problem_type}: [{start_seconds//60:02d}:{start_seconds%60:02d} - {end_seconds//60:02d}:{end_seconds%60:02d}]")

        total_time = time.time() - start_time
        print(
            f"Completed {base_name} in {str(timedelta(seconds=int(total_time)))}")
        sys.stdout.flush()

    except Exception as e:
        logging.error(f"Error processing {base_name}: {str(e)}")
        print(f"Error processing {base_name}: {str(e)}")
        sys.stdout.flush()
        raise  # Re-raise the exception to handle it in the calling function

    finally:
        try:
            container.close()
        except Exception as e:
            logging.error(f"Error closing container for {base_name}: {str(e)}")

    if cache_dir:
        try:
            os.makedirs(cache_dir, exist_ok=True)
            with open(cache_file, 'w') as f:
                json.dump(problematic_ranges, f, indent=2)
            print(f"Saved analysis cache for {base_name} to {cache_file}")
            sys.stdout.flush()
        except Exception as e:
            logging.error(f"Error saving cache for {base_name}: {str(e)}")
            print(f"Error saving cache for {base_name}: {str(e)}")
            sys.stdout.flush()

    return problematic_ranges


def find_valid_frame_ranges(video_analysis, min_length):
    """Find continuous ranges of frames that are not problematic."""
    all_problematic_ranges = []
    for ranges in video_analysis.values():
        all_problematic_ranges.extend(ranges)

    # Sort ranges by start frame
    all_problematic_ranges.sort(key=lambda x: x[0])

    valid_ranges = []
    last_end = 0

    for start, end in all_problematic_ranges:
        if start - last_end >= min_length:
            valid_ranges.append((last_end, start))
        last_end = end

    return valid_ranges


def analyze_video_with_event(args):
    """Wrapper function to analyze video with event name."""
    video_path, event_name = args
    return analyze_video(video_path, cache_dir="video_analysis_cache", event_name=event_name)


def process_event_with_analysis(event_dir, output_dir, event_id, k, max_offset, num_frames, num_workers):
    """Process event with pre-analyzed videos."""
    event_name = os.path.basename(event_dir)
    video_dir = os.path.join(event_dir, "hdVideos" if os.path.exists(
        os.path.join(event_dir, "hdVideos")) else "kinectVideos")

    if not os.path.exists(video_dir):
        return [], []

    video_files = [f for f in os.listdir(video_dir) if f.endswith(".mp4")]

    # Analyze all videos in parallel using the pool with specified number of workers
    video_paths = [os.path.join(video_dir, video_file)
                   for video_file in video_files]
    print(
        f"\nAnalyzing {len(video_paths)} videos from {event_name} using {num_workers} workers...")

    # Create list of tuples containing (video_path, event_name)
    analysis_args = [(path, event_name) for path in video_paths]

    with multiprocessing.Pool(processes=num_workers) as analysis_pool:
        video_analyses = dict(zip(
            video_files,
            analysis_pool.map(analyze_video_with_event, analysis_args)
        ))

    # Find valid frame ranges for each video
    valid_ranges = {
        video_file: find_valid_frame_ranges(analysis, num_frames)
        for video_file, analysis in video_analyses.items()
    }

    # Generate video pairs and tasks
    video_pairs = list(itertools.combinations(video_files, 2))
    tasks = []
    dataset = []

    for pair in video_pairs:
        video1_path = os.path.join(video_dir, pair[0])
        video2_path = os.path.join(video_dir, pair[1])

        # Get valid ranges for both videos
        ranges1 = valid_ranges[pair[0]]
        ranges2 = valid_ranges[pair[1]]

        # Find compatible range pairs (within max_offset frames)
        compatible_ranges = []
        for r1 in ranges1:
            for r2 in ranges2:
                # Calculate the overlap between ranges
                overlap_start = max(r1[0], r2[0])
                overlap_end = min(r1[1], r2[1])
                overlap_length = overlap_end - overlap_start

                # Check if there's enough overlap to extract num_frames
                if overlap_length >= num_frames:
                    compatible_ranges.append((r1, r2, overlap_length))

        if not compatible_ranges:
            continue

        # Sort ranges by length to prioritize longer sequences
        compatible_ranges.sort(key=lambda x: x[2], reverse=True)

        # Calculate k allocation for each range based on its length
        total_length = sum(r[2] for r in compatible_ranges)
        allocated_k = []
        remaining_k = k

        for _, _, length in compatible_ranges:
            # Allocate k proportionally to range length, minimum 1 if any remaining
            range_k = max(1, int((length / total_length) * k))
            range_k = min(range_k, remaining_k)  # Don't exceed remaining k
            allocated_k.append(range_k)
            remaining_k -= range_k

            if remaining_k <= 0:
                break

        # Process each compatible range pair with its allocated k
        for (range1, range2, _), num_samples in zip(compatible_ranges, allocated_k):
            for _ in range(num_samples):
                # Calculate valid start frame range
                # At least max_offset to allow for negative offset
                start_min = max(range1[0], range2[0], max_offset)
                # Leave room for positive offset
                start_max = max(
                    min(range1[1], range2[1]) - num_frames - max_offset, start_min)

                if start_max <= start_min:
                    continue

                # Randomly select start frame
                start_frame = random.randint(start_min, start_max)
                end_frame = start_frame + num_frames

                # Random offset within max_offset
                offset = random.randint(-max_offset, max_offset)

                video1_output_file = f"{event_name}_{pair[0].split('.')[0]}_{start_frame+offset:06}_{end_frame+offset:06}.mp4"
                video2_output_file = f"{event_name}_{pair[1].split('.')[0]}_{start_frame:06}_{end_frame:06}.mp4"

                video1_output_path = os.path.join(
                    output_dir, video1_output_file)
                video2_output_path = os.path.join(
                    output_dir, video2_output_file)

                task = (
                    video1_path,
                    video2_path,
                    start_frame,
                    end_frame,
                    offset,
                    video1_output_path,
                    video2_output_path,
                )

                data_entry = {
                    "id": event_id,
                    "name": event_name,
                    "video_file_0": video1_output_path,
                    "video_file_1": video2_output_path,
                    "seq_len_0": num_frames,
                    "seq_len_1": num_frames,
                    "label_0": offset,
                    "label_1": 0,
                }

                tasks.append(task)
                dataset.append(data_entry)

    return tasks, dataset


def extract_and_save_video_pair(video1_path, video2_path, start_frame, end_frame, offset, video1_output_path, video2_output_path):
    """Extract and save synchronized video segments using PyAV."""
    timings = {
        'decode': {'video1': 0.0, 'video2': 0.0},
        'reformat': {'video1': 0.0, 'video2': 0.0},
        'encode': {'video1': 0.0, 'video2': 0.0},
        'mux': {'video1': 0.0, 'video2': 0.0}
    }
    frame_counts = {'video1': 0, 'video2': 0}
    start_total = time.time()

    try:
        # Open input containers
        start_time = time.time()
        container1 = av.open(video1_path)
        container2 = av.open(video2_path)
        stream1 = container1.streams.video[0]
        stream2 = container2.streams.video[0]
        stream1.thread_type = "AUTO"
        stream2.thread_type = "AUTO"
        timings['open_inputs'] = time.time() - start_time

        # Create output containers
        start_time = time.time()
        output1 = av.open(video1_output_path, mode='w')
        output2 = av.open(video2_output_path, mode='w')

        # Add streams with same timebase as input
        output_stream1 = output1.add_stream('h264', rate=stream1.average_rate)
        output_stream1.time_base = stream1.time_base
        output_stream2 = output2.add_stream('h264', rate=stream2.average_rate)
        output_stream2.time_base = stream2.time_base

        # Set stream parameters
        output_stream1.width = 224
        output_stream1.height = 224
        output_stream1.pix_fmt = 'yuv420p'
        output_stream1.options = {'crf': '23'}

        output_stream2.width = 224
        output_stream2.height = 224
        output_stream2.pix_fmt = 'yuv420p'
        output_stream2.options = {'crf': '23'}
        timings['setup_outputs'] = time.time() - start_time

        # Calculate frame positions and seek
        start_time = time.time()
        start_frame1 = start_frame + offset
        start_frame2 = start_frame

        seek_pts1 = int(
            (start_frame1 / stream1.average_rate) / stream1.time_base)
        seek_pts2 = int(
            (start_frame2 / stream2.average_rate) / stream2.time_base)

        container1.seek(seek_pts1, stream=stream1)
        container2.seek(seek_pts2, stream=stream2)
        timings['seek'] = time.time() - start_time

        frames_needed = end_frame - start_frame

        # Process first video
        for container, output, output_stream, video_key in [
            (container1, output1, output_stream1, 'video1'),
            (container2, output2, output_stream2, 'video2')
        ]:
            while frame_counts[video_key] < frames_needed:
                # Decode
                decode_start = time.time()
                try:
                    frame = next(container.decode(video=0))
                except StopIteration:
                    break
                timings['decode'][video_key] += time.time() - decode_start

                # Reformat
                reformat_start = time.time()
                frame = frame.reformat(width=224, height=224, format='yuv420p')
                timings['reformat'][video_key] += time.time() - reformat_start

                # Encode
                encode_start = time.time()
                packets = list(output_stream.encode(frame))
                timings['encode'][video_key] += time.time() - encode_start

                # Mux
                mux_start = time.time()
                for packet in packets:
                    output.mux(packet)
                timings['mux'][video_key] += time.time() - mux_start

                frame_counts[video_key] += 1

        # Flush encoders
        start_time = time.time()
        for output, output_stream, video_key in [
            (output1, output_stream1, 'video1'),
            (output2, output_stream2, 'video2')
        ]:
            # Encode
            encode_start = time.time()
            packets = list(output_stream.encode(None))
            timings['encode'][video_key] += time.time() - encode_start

            # Mux
            mux_start = time.time()
            for packet in packets:
                output.mux(packet)
            timings['mux'][video_key] += time.time() - mux_start
        timings['flush_encoders'] = time.time() - start_time

        if frame_counts['video1'] < frames_needed or frame_counts['video2'] < frames_needed:
            return False, f"insufficient_frames: got {frame_counts['video1']}/{frame_counts['video2']}, needed {frames_needed}"

        timings['total'] = time.time() - start_total

        # Log detailed timing information
        logging.debug(
            f"Video processing timings for {os.path.basename(video1_output_path)}:")
        logging.debug(
            f"  Total frames processed: video1={frame_counts['video1']}, video2={frame_counts['video2']}")

        # Log setup times
        for op in ['open_inputs', 'setup_outputs', 'seek', 'flush_encoders', 'total']:
            if op in timings:
                logging.debug(f"  {op}: {timings[op]:.3f}s")

        # Log per-frame operation times
        for video_key in ['video1', 'video2']:
            frames = frame_counts[video_key]
            if frames > 0:
                logging.debug(f"  {video_key} per-frame averages:")
                for op in ['decode', 'reformat', 'encode', 'mux']:
                    total_time = timings[op][video_key]
                    avg_time = total_time / frames
                    logging.debug(
                        f"    {op}: {total_time:.3f}s total, {avg_time*1000:.2f}ms/frame")

        return True, None

    except Exception as e:
        logging.error(f"Error in sync_and_save_video_pair: {str(e)}")
        return False, f"processing_error: {str(e)}"

    finally:
        # Clean up resources
        try:
            container1.close()
            container2.close()
            output1.close()
            output2.close()
        except Exception as e:
            logging.error(f"Error closing containers: {str(e)}")


def signal_handler(sig, frame):
    global pool
    print("\nInterrupt received, terminating processes...")
    if pool:
        pool.terminate()  # Forcefully stop all worker processes
        pool.join()  # Wait for termination
    sys.exit(0)  # Exit the main process


def main(data_root, output_dir, k, max_offset, num_frames, num_workers):
    global pool
    os.makedirs(output_dir, exist_ok=True)
    processed_videos_dir = os.path.join(output_dir, "processed_videos")
    os.makedirs(processed_videos_dir, exist_ok=True)
    processed_data = []

    event_id = 0
    task_list = []

    # Set up a signal handler for interrupt (Ctrl+C)
    signal.signal(signal.SIGINT, signal_handler)

    # Iterate over events and create tasks
    for event_dir in os.listdir(data_root):
        full_event_dir = os.path.join(data_root, event_dir)
        if not os.path.isdir(full_event_dir):
            continue

        tasks, event_dataset = process_event_with_analysis(
            full_event_dir,
            processed_videos_dir,
            event_id,
            k,
            max_offset,
            num_frames,
            num_workers=num_workers
        )

        logging.info(f"Event {event_id}: {len(tasks)} tasks created.")
        event_id += 1
        processed_data.extend(event_dataset)

        # Add each task with its corresponding dataset entry
        for task, dataset_entry in zip(tasks, event_dataset):
            # Link each task to its corresponding entry
            task_list.append((task, dataset_entry))

    # Use a multiprocessing pool with a fixed number of CPUs
    pool = multiprocessing.Pool(processes=num_workers)

    pbar = tqdm(total=0, desc="Processing all events", dynamic_ncols=True)

    def update_progress(_):
        pbar.update(1)

    pbar.total = len(task_list)

    logging.info(f"Total tasks: {len(task_list)}")
    logging.info(
        f"Total processed data before processing: {len(processed_data)} entries"
    )

    failed_tasks = load_failed_tasks()
    # Process tasks
    for task, dataset_entry in task_list:
        result = pool.apply_async(
            process_video_pair, args=(
                task, failed_tasks), callback=update_progress
        )

        if not result.get():  # If the task fails
            logging.warning(f"Processing failed for entry: {dataset_entry}")
            if dataset_entry in processed_data:
                # Remove only the failed task's entry
                processed_data.remove(dataset_entry)

    pool.close()
    pool.join()

    pbar.close()

    logging.info(f"Final processed data length: {len(processed_data)} entries")

    # Shuffle the processed_data before splitting
    random.shuffle(processed_data)

    # Split the data into 70% train and 30% val
    split_index = len(processed_data) // 10 * 7
    train_data = processed_data[:split_index]
    val_data = processed_data[split_index:]

    # Save train and val data
    with open(os.path.join(output_dir, "train.pkl"), "wb") as f:
        pickle.dump(train_data, f)
    logging.info(
        f"Training data saved to train.pkl with {len(train_data)} entries")

    with open(os.path.join(output_dir, "val.pkl"), "wb") as f:
        pickle.dump(val_data, f)
    logging.info(
        f"Validation data saved to val.pkl with {len(val_data)} entries")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python cmu_process.py <data_root> <output_dir>")
        sys.exit(1)
    data_root = sys.argv[1]
    output_dir = sys.argv[2]
    main(data_root, output_dir, k=5, max_offset=30, num_frames=240, num_workers=2)
