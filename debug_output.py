"""
Debug script to check if output video is being created properly
"""
import os
import cv2

def check_directories():
    """Check if required directories exist"""
    print("=" * 60)
    print("Checking Directories...")
    print("=" * 60)
    
    dirs = ['uploads', 'static', 'static/results', 'templates']
    for d in dirs:
        exists = os.path.exists(d)
        print(f"{'✓' if exists else '✗'} {d}: {'EXISTS' if exists else 'MISSING'}")
        if not exists:
            os.makedirs(d, exist_ok=True)
            print(f"  → Created {d}")
    print()

def check_output_videos():
    """List all output videos"""
    print("=" * 60)
    print("Output Videos in static/results/")
    print("=" * 60)
    
    results_dir = 'static/results'
    if not os.path.exists(results_dir):
        print("❌ Directory doesn't exist!")
        return
    
    files = os.listdir(results_dir)
    if not files:
        print("❌ No files found!")
        return
    
    for file in files:
        filepath = os.path.join(results_dir, file)
        size = os.path.getsize(filepath)
        print(f"📹 {file}")
        print(f"   Size: {size:,} bytes ({size/1024/1024:.2f} MB)")
        
        # Check if video can be opened
        cap = cv2.VideoCapture(filepath)
        if cap.isOpened():
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            print(f"   Resolution: {width}x{height}")
            print(f"   FPS: {fps}")
            print(f"   Frames: {frame_count}")
            print(f"   ✓ Video can be opened by OpenCV")
            cap.release()
        else:
            print(f"   ✗ Cannot open video with OpenCV")
        print()

def test_video_codec():
    """Test which video codecs are available"""
    print("=" * 60)
    print("Testing Video Codecs...")
    print("=" * 60)
    
    test_file = 'test_codec.mp4'
    codecs = ['mp4v', 'avc1', 'H264', 'X264', 'XVID', 'MJPG']
    
    for codec in codecs:
        try:
            fourcc = cv2.VideoWriter_fourcc(*codec)
            out = cv2.VideoWriter(test_file, fourcc, 20.0, (640, 480))
            if out.isOpened():
                # Write a test frame
                frame = cv2.imread('static/results/frame.jpg') if os.path.exists('static/results/frame.jpg') else None
                if frame is None:
                    import numpy as np
                    frame = np.zeros((480, 640, 3), dtype=np.uint8)
                out.write(frame)
                out.release()
                
                # Check if file was created and has size
                if os.path.exists(test_file) and os.path.getsize(test_file) > 0:
                    print(f"✓ {codec}: WORKS (file size: {os.path.getsize(test_file)} bytes)")
                    os.remove(test_file)
                else:
                    print(f"✗ {codec}: Created but empty")
            else:
                print(f"✗ {codec}: Cannot open VideoWriter")
        except Exception as e:
            print(f"✗ {codec}: Error - {e}")
    print()

def check_flask_static():
    """Check Flask static file serving"""
    print("=" * 60)
    print("Flask Static File Configuration")
    print("=" * 60)
    print("Make sure your Flask app has:")
    print("  app = Flask(__name__)")
    print("  # Flask automatically serves files from 'static' folder")
    print()
    print("URLs should be:")
    print("  http://localhost:5000/static/results/output_video.mp4")
    print()

def main():
    print("\n🔍 Car Counter Debug Tool\n")
    
    check_directories()
    check_output_videos()
    test_video_codec()
    check_flask_static()
    
    print("=" * 60)
    print("Recommendations:")
    print("=" * 60)
    print("1. Check browser console (F12) for errors")
    print("2. Try accessing the video URL directly in browser:")
    print("   http://localhost:5000/static/results/output_video.mp4")
    print("3. If video plays directly but not in page, check browser video codec support")
    print("4. Try different browsers (Chrome, Firefox, Edge)")
    print("5. Check file permissions on static/results folder")
    print()

if __name__ == '__main__':
    main()