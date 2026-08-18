"""
DOWNLOAD & ORGANIZE GAIT DATA
==============================
Extracts gait sensor data from the existing archive and organizes by category.
Also attempts to download additional PhysioNet gait data.

Usage:
  python download_gait_data.py
"""

import sys
import os
import zipfile
import shutil
from pathlib import Path
import urllib.request

if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

# Paths
BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BACKEND_DIR.parent
DATA_DIR = PROJECT_DIR / "data"
GAIT_DIR = DATA_DIR / "gait"
HEALTHY_DIR = GAIT_DIR / "healthy"
DIAGNOSED_DIR = GAIT_DIR / "diagnosed"
ARCHIVE_PATH = DATA_DIR / "archive (2).zip"
TEMP_EXTRACT_DIR = DATA_DIR / "_temp_extract"


def extract_archive():
    """Extract the existing archive and organize gait files."""
    if not ARCHIVE_PATH.exists():
        print(f"Archive not found at: {ARCHIVE_PATH}")
        return 0, 0

    print(f"Extracting archive ({ARCHIVE_PATH.stat().st_size / (1024*1024):.0f} MB)...")
    print("This may take a few minutes...")

    TEMP_EXTRACT_DIR.mkdir(parents=True, exist_ok=True)
    healthy_count = 0
    diagnosed_count = 0

    try:
        with zipfile.ZipFile(str(ARCHIVE_PATH), 'r') as zf:
            # List all files in the archive
            all_names = zf.namelist()
            txt_files = [n for n in all_names if n.endswith('.txt') and not n.endswith('/')]
            print(f"  Found {len(txt_files)} .txt files in archive")

            for fname in txt_files:
                basename = Path(fname).name

                # Skip non-gait files (README, etc.)
                if not any(prefix in basename for prefix in ['Ga', 'Ju', 'Si']):
                    continue

                # Classify by filename
                is_healthy = any(ctrl in basename for ctrl in ['Co'])
                is_diagnosed = any(pt in basename for pt in ['Pt'])

                if is_healthy:
                    target_dir = HEALTHY_DIR
                elif is_diagnosed:
                    target_dir = DIAGNOSED_DIR
                else:
                    continue

                target_path = target_dir / basename

                # Skip if already exists
                if target_path.exists():
                    if is_healthy:
                        healthy_count += 1
                    else:
                        diagnosed_count += 1
                    continue

                # Extract to target
                try:
                    data = zf.read(fname)
                    if len(data) > 200:  # Skip tiny/empty files
                        with open(target_path, 'wb') as f:
                            f.write(data)
                        if is_healthy:
                            healthy_count += 1
                        else:
                            diagnosed_count += 1
                except Exception as e:
                    print(f"  Error extracting {basename}: {e}")

    except zipfile.BadZipFile:
        print("  ERROR: Archive is corrupted or not a valid zip file")
    except Exception as e:
        print(f"  ERROR: {e}")
    finally:
        # Clean up temp
        if TEMP_EXTRACT_DIR.exists():
            shutil.rmtree(str(TEMP_EXTRACT_DIR), ignore_errors=True)

    return healthy_count, diagnosed_count


def download_physionet_data():
    """Try to download additional gait data from PhysioNet."""
    BASE_URL = "https://physionet.org/files/gaitpdb/1.0.0/"

    # Known file patterns from PhysioNet Gait in PD database
    # Ga = Gait, Ju = Juggling (dual-task), Si = Sitting
    prefixes = {
        'healthy': ['GaCo', 'JuCo', 'SiCo'],
        'diagnosed': ['GaPt', 'JuPt', 'SiPt'],
    }

    # Try to get the file listing
    print("\nAttempting to download from PhysioNet...")
    downloaded = 0

    # Try common subject IDs (01-30) with trial numbers (01, 02, 10)
    for label, file_prefixes in prefixes.items():
        target_dir = HEALTHY_DIR if label == 'healthy' else DIAGNOSED_DIR
        for prefix in file_prefixes:
            for subj_id in range(1, 31):
                for trial in ['01', '02', '10']:
                    filename = f"{prefix}{subj_id:02d}_{trial}.txt"
                    target_path = target_dir / filename

                    if target_path.exists():
                        continue

                    url = f"{BASE_URL}{filename}"
                    try:
                        urllib.request.urlretrieve(url, str(target_path))
                        # Verify file is not an error page
                        if target_path.stat().st_size < 200:
                            target_path.unlink()
                            continue
                        downloaded += 1
                        print(f"  Downloaded: {filename}")
                    except Exception:
                        if target_path.exists():
                            target_path.unlink(missing_ok=True)
                        continue

    return downloaded


def count_files():
    """Count and report files in each directory."""
    healthy = list(HEALTHY_DIR.glob("*.txt"))
    diagnosed = list(DIAGNOSED_DIR.glob("*.txt"))

    print(f"\n{'='*50}")
    print(f"GAIT DATA SUMMARY")
    print(f"{'='*50}")
    print(f"Healthy controls: {len(healthy)} files")
    for f in sorted(healthy)[:5]:
        print(f"  - {f.name} ({f.stat().st_size / 1024:.0f} KB)")
    if len(healthy) > 5:
        print(f"  ... and {len(healthy) - 5} more")

    print(f"\nPD patients: {len(diagnosed)} files")
    for f in sorted(diagnosed)[:5]:
        print(f"  - {f.name} ({f.stat().st_size / 1024:.0f} KB)")
    if len(diagnosed) > 5:
        print(f"  ... and {len(diagnosed) - 5} more")

    print(f"\nTotal: {len(healthy) + len(diagnosed)} gait sensor files")
    return len(healthy), len(diagnosed)


if __name__ == "__main__":
    print("=" * 50)
    print("GAIT DATA DOWNLOAD & ORGANIZATION")
    print("=" * 50)

    # Ensure directories exist
    HEALTHY_DIR.mkdir(parents=True, exist_ok=True)
    DIAGNOSED_DIR.mkdir(parents=True, exist_ok=True)

    # Step 1: Extract from archive
    print("\nStep 1: Extracting from archive...")
    h_archive, d_archive = extract_archive()
    print(f"  From archive: {h_archive} healthy, {d_archive} diagnosed")

    # Step 2: Download from PhysioNet
    print("\nStep 2: Downloading from PhysioNet...")
    downloaded = download_physionet_data()
    print(f"  Downloaded: {downloaded} new files")

    # Step 3: Summary
    count_files()

    print("\nDone! Run retrain_gait_models.py to train models on expanded data.")
