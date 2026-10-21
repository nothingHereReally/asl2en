from pathlib import Path

import cv2
import numpy as np
from mediapipe.python.solutions.holistic import Holistic
from mediapipe.python.solutions import drawing_utils, holistic

PROJ_ROOT: Path= Path(__file__).parent.parent.parent
IMAGE_PATH: Path= PROJ_ROOT /"src_asl2gloss" /"pics_others" /"human_whole_body.png"
RESULT: Path= PROJ_ROOT /"src_asl2gloss" /"see_opencv2help_mediapipe" /"results"

def make_variants(image):
    lab= cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    light, a, b= cv2.split(lab)
    clahe= cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8)).apply(light)
    gentle_clahe= cv2.cvtColor(cv2.merge((clahe, a, b)), cv2.COLOR_LAB2BGR)
    denoised= cv2.bilateralFilter(image, d=5, sigmaColor=25, sigmaSpace=25)
    blurred= cv2.GaussianBlur(image, (0, 0), 2)
    sharpened= cv2.addWeighted(image, 1.8, blurred, -0.8, 0)
    return {
        "original": image,
        "clahe_mild": gentle_clahe,
        "bilateral_mild": denoised,
        "sharpen_strong": sharpened,
    }


def draw_results(image, result):
    rendered= image.copy()
    if result.pose_landmarks:
        drawing_utils.draw_landmarks(rendered, result.pose_landmarks, holistic.POSE_CONNECTIONS)
    if result.face_landmarks:
        drawing_utils.draw_landmarks(rendered, result.face_landmarks, holistic.FACEMESH_CONTOURS)
    if result.left_hand_landmarks:
        drawing_utils.draw_landmarks(rendered, result.left_hand_landmarks, holistic.HAND_CONNECTIONS)
    if result.right_hand_landmarks:
        drawing_utils.draw_landmarks(rendered, result.right_hand_landmarks, holistic.HAND_CONNECTIONS)
    found= [
        f"pose={bool(result.pose_landmarks)}",
        f"face={bool(result.face_landmarks)}",
        f"left_hand={bool(result.left_hand_landmarks)}",
        f"right_hand={bool(result.right_hand_landmarks)}",
    ]
    cv2.putText(rendered, "  ".join(found), (20, 42), cv2.FONT_HERSHEY_SIMPLEX,
                0.85, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(rendered, "  ".join(found), (20, 42), cv2.FONT_HERSHEY_SIMPLEX,
                0.85, (255, 255, 255), 2, cv2.LINE_AA)
    return rendered


def save_jpg(path, image):
    if not cv2.imwrite(str(path), image, [cv2.IMWRITE_JPEG_QUALITY, 92]):
        raise OSError(f"Could not write {path}")


def main():
    RESULT.mkdir(parents=True, exist_ok=True)
    source= cv2.imread(str(IMAGE_PATH), cv2.IMREAD_COLOR)
    if source is None:
        raise FileNotFoundError(f"Could not read sample image: {IMAGE_PATH}")

    reports, tiles= [], []
    for name, image in make_variants(source).items():
        with Holistic(static_image_mode=True, model_complexity=2,
                      min_detection_confidence=0.55,
                      min_tracking_confidence=0.55) as mph_fph:
            result= mph_fph.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        flags= {
            "pose": bool(result.pose_landmarks),
            "face": bool(result.face_landmarks),
            "left_hand": bool(result.left_hand_landmarks),
            "right_hand": bool(result.right_hand_landmarks),
        }
        save_jpg(RESULT / f"{name}_input.jpg", image)
        rendered= draw_results(image, result)
        save_jpg(RESULT / f"{name}_holistic.jpg", rendered)
        small= cv2.resize(rendered, (480, 600), interpolation=cv2.INTER_AREA)
        cv2.putText(small, name, (12, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (20, 20, 20), 3, cv2.LINE_AA)
        cv2.putText(small, name, (12, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(small)
        reports.append(f"{name}: " + ", ".join(f"{key}={value}" for key, value in flags.items()))

    sheet= np.vstack((np.hstack(tiles[:2]), np.hstack(tiles[2:])))
    save_jpg(RESULT / "comparison_contact_sheet.jpg", sheet)
    report= "\n".join(reports) + "\n"
    (RESULT / "detection_results.txt").write_text(report, encoding="utf-8")
    print(report, end="")
    print(f"Saved JPG inputs, Holistic overlays, and contact sheet to: {RESULT}")


if __name__ == "__main__":
    main()
