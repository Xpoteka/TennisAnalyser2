# Labeling the spike test set

These labels are the ground truth for the spike. Every accuracy number and every confidence
calibration is measured against them, so they must show only what you can actually see. Total
effort is about 2–3 hours, and it can be split over several sessions.

## What to label

| Set | Frames | Taken from |
| --- | --- | --- |
| Ball | about 1,500, as **3 runs of about 500 consecutive frames** (about 8 s each at 60 fps) | at least 2 camera heights, both cameras |
| Pose | about 100 frames, **spread out** (for example every 30th frame) | the same clips; both players in each frame |

Pick stretches with real rallies, including serves, net play and moments where the ball is hidden
behind a player.

## Tool: CVAT

Use [CVAT](https://www.cvat.ai). The free cloud version works; self-host it if you'd rather not
upload your footage.

1. Create one **task per clip** and upload the original 4K60 file. Leave **frame step at 1 and
   start frame at 0**, so CVAT's frame numbers match the scripts' frame numbers. If you cut the
   clip first, upload the cut file and use that same file in `samples/`.
2. Add two labels:
   - `ball`, type **Points**
   - `player`, type **Skeleton**, with the 17 COCO keypoints in this order: nose, left_eye,
     right_eye, left_ear, right_ear, left_shoulder, right_shoulder, left_elbow, right_elbow,
     left_wrist, right_wrist, left_hip, right_hip, left_knee, right_knee, left_ankle, right_ankle.
     "Left" means the player's own left, not the left of the image.

## Ball rules

- One point per frame at the **centre of the ball**. For a motion-blurred streak, the centre of the streak.
- Ball not visible (hidden, out of frame, can't tell it apart) → **no point on that frame**. Never
  place it where you think it should be.
- Use **shape** mode (a point on each frame), not track mode. Track mode fills the frames between
  your clicks with interpolated positions, which are guesses.
- More than one ball in view (a spare ball on the court) → label only the ball in play.

## Pose rules

- One skeleton per player on court. Ignore people off court, ball kids and spectators.
- Each keypoint gets one state:
  - **visible**: you can see it, so place it exactly.
  - **occluded**: hidden (behind the body or the racket), but its position is obvious from the
    body. Place it and set CVAT's *occluded* flag.
  - **outside**: you can't tell where it is. Mark it as *outside*/not labeled. Never guess.
- Wrists and elbows matter most for shot detection later, so be most careful with those.

## Export and hand-off

| Set | CVAT export format | Save as |
| --- | --- | --- |
| Ball | CVAT for video 1.1 | `/mnt/HDDs/tennis/labels/<clip>.ball.xml` |
| Pose | COCO Keypoints 1.0 | `/mnt/HDDs/tennis/labels/<clip>.pose.json` |

`<clip>` is the video's file name without its extension, exactly as it appears in `samples/`.

Don't pre-label with any of the models being tested (YOLO11-Pose, RTMPose, TrackNet, WASB).
Correcting a model's output biases the labels towards that model's mistakes.
