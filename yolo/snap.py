import cv2, pyrealsense2 as rs, numpy as np, itertools, os
os.makedirs("data", exist_ok=True)
pipe = rs.pipeline(); pipe.start()
for i in itertools.count():
    f = pipe.wait_for_frames().get_color_frame()
    img = np.asanyarray(f.get_data())
    cv2.imshow("snap (SPACE=save, q=quit)", img)
    k = cv2.waitKey(1) & 0xFF
    if k == ord(' '):
        cv2.imwrite(f"data/box_{i:04d}.jpg", img); print("saved", i)
    if k == ord('q'): break
