import cv2
import os

import torch
import ultralytics
# from process import Process
from process_830 import Process
import time
import numpy as np
# from only_draw_mask import Process as Process_only_draw
# from process_zhoushan import Process as Process_zhoushan

print(ultralytics.__version__)


# pts1 = np.float32([(450, 674), (750, 674), (490, 30), (710, 30)])
# pts2 = np.float32([(450, 674), (750, 674), (450, 0), (750, 0)])

pts1 = np.float32([(450, 654), (750, 654), (470, 20), (730, 20)])
pts2 = np.float32([(450, 674), (750, 674), (450, 0), (750, 0)])
matrix = cv2.getPerspectiveTransform(pts1, pts2)

device = torch.device('cuda:0')
# process = Process(device, matrix)
process = Process()
# process_only_draw = Process_only_draw(device, matrix)
# process_zhoushan = Process_zhoushan(device, matrix)


try:
    cap = cv2.VideoCapture('./test_videos/dahuavideo.mp4')
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output_dir = "dahua"
    os.makedirs(output_dir, exist_ok=True)

    frame_count = 0
    processed_frame_count = 0
    frame_interval = 3
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Failed to grab frame")
            break

        frame_count += 1
        if frame_count % frame_interval != 0:
            continue
        # YOLO推理
        processed_frame_count += 1
        strat_time = time.time()
        frame = process.process_frame(frame, width=width, height=height
                                                           )
        # frame, _, _, _, _, _, _, _ = process_only_draw.process_frame(frame, width=width, height=height
        #                                                    , accident_flag=False, stop_car_flag=False, slow_car_flag=False, crowd_flag=False
        #                                                    )
        # frame, _, _, _, _, _, _, _ = process_zhoushan.process_frame(frame, width=width, height=height
        #                                                    # , accident_flag=False, stop_car_flag=False, slow_car_flag=False, crowd_flag=False
        #                                                    )

        elapsed_time = time.time() - strat_time
        print(f"Time: {elapsed_time:02f}")

        filename = os.path.join(output_dir, f"{processed_frame_count:04d}.jpg")
        cv2.imwrite(filename, frame)
        # if processed_frame_count > 20:
        #     break

finally:
    # 清理资源
    cap.release()
    cv2.destroyAllWindows()
    process.shutdown()
    print("处理完成，资源已释放")
