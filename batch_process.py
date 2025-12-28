"""
Batch process all videos in input folder with Volleyball Hawk-Eye
"""

import os
from pathlib import Path
import logging
from hawkeye_complete import volleyball_hawkeye_pipeline
import torch

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

def process_all_videos(input_folder, weights_path, output_folder=None):
    """
    Process all videos in input folder
    
    Args:
        input_folder: Folder containing input videos
        weights_path: RT-DETR weights path
        output_folder: Output folder (defaults to input_folder)
    """
    input_path = Path(input_folder)
    
    if output_folder is None:
        output_folder = input_path.parent / "output_videos"
    
    output_path = Path(output_folder)
    output_path.mkdir(exist_ok=True)
    
    # Find all video files
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.MP4', '.AVI', '.MOV']
    video_files = []
    
    for ext in video_extensions:
        video_files.extend(list(input_path.glob(f'*{ext}')))
    
    if not video_files:
        logger.error(f"No videos found in {input_folder}")
        return
    
    logger.info("="*70)
    logger.info(f"BATCH PROCESSING: {len(video_files)} videos found")
    logger.info("="*70)
    
    for i, video_file in enumerate(video_files, 1):
        logger.info(f"\n{'-'*70}")
        logger.info(f"Processing video {i}/{len(video_files)}: {video_file.name}")
        logger.info(f"{'-'*70}")
        
        output_name = f"hawkeye_{video_file.stem}.mp4"
        output_file = output_path / output_name
        
        try:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            
            volleyball_hawkeye_pipeline(
                video_path=str(video_file),
                weights_path=weights_path,
                output_path=str(output_file),
                device=device,
                use_siglip=False
            )
            
            logger.info(f"✓ Success: {output_name}")
            
        except Exception as e:
            logger.error(f"✗ Failed: {video_file.name} - {e}")
            continue
    
    logger.info("\n" + "="*70)
    logger.info("BATCH PROCESSING COMPLETE")
    logger.info(f"Output folder: {output_path}")
    logger.info("="*70)

if __name__ == "__main__":
    INPUT_FOLDER = r"C:\Users\xghostrider\Downloads\NEw_ProJect\Volleyball\input_videos"
    WEIGHTS_PATH = r"C:\Users\xghostrider\Downloads\best(2).pt"
    OUTPUT_FOLDER = r"C:\Users\xghostrider\Downloads\NEw_ProJect\Volleyball\output_videos"
    
    # Verify inputs
    if not Path(INPUT_FOLDER).exists():
        logger.error(f"Input folder not found: {INPUT_FOLDER}")
        exit(1)
    
    if not Path(WEIGHTS_PATH).exists():
        logger.error(f"Weights not found: {WEIGHTS_PATH}")
        exit(1)
    
    # Process all videos
    process_all_videos(
        input_folder=INPUT_FOLDER,
        weights_path=WEIGHTS_PATH,
        output_folder=OUTPUT_FOLDER
    )
