"""
Download video from Google Drive folder for inference testing
"""
import os
import gdown

# Google Drive folder URL
folder_url = "https://drive.google.com/drive/folders/1nTK95Lpn_KR5gthvCEie2nQbrFwzalNm"

print("=" * 60)
print("Downloading Volleyball Video from Google Drive")
print("=" * 60)

# Create input directory
os.makedirs("input_videos", exist_ok=True)

# Download the folder (will get all videos)
print("\nDownloading videos from Google Drive folder...")
print("This may take a few minutes depending on file size...")

try:
    # Download folder contents
    gdown.download_folder(folder_url, output="input_videos", quiet=False, use_cookies=False)
    print("\n[OK] Download complete!")
    print(f"Videos saved to: input_videos/")
    
    # List downloaded videos
    import glob
    videos = glob.glob("input_videos/**/*.mp4", recursive=True) + \
             glob.glob("input_videos/**/*.avi", recursive=True) + \
             glob.glob("input_videos/**/*.mov", recursive=True)
    
    if videos:
        print(f"\nFound {len(videos)} video(s):")
        for v in videos:
            size_mb = os.path.getsize(v) / (1024 * 1024)
            print(f"  - {v} ({size_mb:.1f} MB)")
        print("\n[OK] Ready for inference!")
    else:
        print("\n[WARNING] No video files found. Trying direct file download...")
        
except Exception as e:
    print(f"\n[ERROR] Download failed: {e}")
    print("\nTrying alternative method...")
    
    # Try downloading first file directly
    # You may need to make the folder publicly accessible
    print("Please ensure the Google Drive folder is publicly accessible.")
    print("Share settings should be: 'Anyone with the link can view'")
