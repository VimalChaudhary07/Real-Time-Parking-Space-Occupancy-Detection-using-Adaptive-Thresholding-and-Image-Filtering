import cv2

from parking import (
    GlobalShiftEstimator,
    ParkingOccupancyDetector,
    build_occupancy_mask,
    load_detector_config,
    load_slots,
    save_slots,
)


VIDEO_PATH = "carPark.mp4"
SLOTS_PATH = "CarParkPos"


def main():
    config = load_detector_config()
    slots = load_slots(SLOTS_PATH, config.default_slot_width, config.default_slot_height)
    if not slots:
        raise RuntimeError("No parking slots found. Run ParkingSpacePicker.py first.")

    # Persist migrated format if legacy data was loaded.
    save_slots(SLOTS_PATH, slots)

    cap = cv2.VideoCapture(VIDEO_PATH)
    ref_success, reference_frame = cap.read()
    if not ref_success or reference_frame is None:
        raise RuntimeError("Could not read reference frame from carPark.mp4")

    reference_gray = cv2.cvtColor(reference_frame, cv2.COLOR_BGR2GRAY)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    shift_estimator = GlobalShiftEstimator(
        reference_gray=reference_gray,
        scale=config.shift_scale,
        response_threshold=config.shift_response_threshold,
        smooth_alpha=config.shift_smooth_alpha,
        max_shift_pixels=config.max_shift_pixels,
    )
    detector = ParkingOccupancyDetector(slots=slots, config=config)

    cv2.namedWindow("Image", cv2.WINDOW_NORMAL)

    while True:
        if cap.get(cv2.CAP_PROP_POS_FRAMES) >= cap.get(cv2.CAP_PROP_FRAME_COUNT):
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        success, frame = cap.read()
        if not success:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        gray, occupancy_mask = build_occupancy_mask(frame)
        shift_x, shift_y = shift_estimator.estimate(gray)

        detector.process_frame(
            occupancy_mask=occupancy_mask,
            draw_frame=frame,
            shift_x=shift_x,
            shift_y=shift_y,
            frame_bgr=frame,
        )

        cv2.imshow("Image", frame)
        if cv2.waitKey(10) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
