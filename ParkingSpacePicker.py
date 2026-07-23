import cv2
import numpy as np

from parking import (
    clone_slot,
    load_detector_config,
    load_slots,
    move_slot,
    point_in_slot,
    polygon_from_points,
    rect_from_points,
    save_slots,
    slot_points,
)


SLOTS_PATH = "CarParkPos"
VIDEO_PATH = "carPark.mp4"
FALLBACK_IMAGE_PATH = "carParkImg.png"
MIN_SLOT_SIZE = 10
HANDLE_RADIUS = 7


class SlotEditor:
    def __init__(self):
        self.config = load_detector_config()
        self.slots = load_slots(SLOTS_PATH, self.config.default_slot_width, self.config.default_slot_height)
        self.base_img = self.load_reference_image()
        self.mode = "rect"
        self.selected_index = None
        self.is_drawing_rect = False
        self.is_moving = False
        self.is_panning = False
        self.drag_handle = None
        self.start_point = (0, 0)
        self.current_point = (0, 0)
        self.last_world = (0, 0)
        self.last_screen = (0, 0)
        self.zoom = 1.0
        self.pan_x = 0
        self.pan_y = 0
        self.polygon_points = []

    def load_reference_image(self):
        cap = cv2.VideoCapture(VIDEO_PATH)
        success, frame = cap.read()
        cap.release()

        if success and frame is not None:
            return frame

        img = cv2.imread(FALLBACK_IMAGE_PATH)
        if img is None:
            raise FileNotFoundError("Could not load carPark.mp4 first frame or carParkImg.png")
        return img

    def save(self):
        save_slots(SLOTS_PATH, self.slots)

    def screen_to_world(self, x, y):
        return int((x - self.pan_x) / self.zoom), int((y - self.pan_y) / self.zoom)

    def world_to_screen(self, point):
        x, y = point
        return int(x * self.zoom + self.pan_x), int(y * self.zoom + self.pan_y)

    def slot_at(self, point):
        x, y = point
        for index in range(len(self.slots) - 1, -1, -1):
            if point_in_slot(x, y, self.slots[index]):
                return index
        return None

    def handle_at(self, point):
        if self.selected_index is None:
            return None

        slot = self.slots[self.selected_index]
        points = slot_points(slot)
        for index, handle in enumerate(points):
            if abs(point[0] - handle[0]) <= HANDLE_RADIUS and abs(point[1] - handle[1]) <= HANDLE_RADIUS:
                return index
        return None

    def resize_slot(self, slot, handle_index, point):
        if slot.get("type") == "polygon":
            points = list(slot["points"])
            points[handle_index] = point
            if len(points) >= 3:
                slot.update(polygon_from_points(points))
            return

        x, y, w, h = slot["x"], slot["y"], slot["w"], slot["h"]
        x2, y2 = x + w, y + h
        px, py = point
        if handle_index == 0:
            nx, ny, nw, nh = min(px, x2), min(py, y2), abs(x2 - px), abs(y2 - py)
        elif handle_index == 1:
            nx, ny, nw, nh = min(x, px), min(py, y2), abs(px - x), abs(y2 - py)
        elif handle_index == 2:
            nx, ny, nw, nh = min(x, px), min(y, py), abs(px - x), abs(py - y)
        else:
            nx, ny, nw, nh = min(px, x2), min(y, py), abs(x2 - px), abs(py - y)
        if nw >= MIN_SLOT_SIZE and nh >= MIN_SLOT_SIZE:
            slot.update({"x": nx, "y": ny, "w": nw, "h": nh})

    def add_rect(self, p1, p2):
        rect = rect_from_points(p1, p2)
        if rect["w"] >= MIN_SLOT_SIZE and rect["h"] >= MIN_SLOT_SIZE:
            rect["id"] = len(self.slots) + 1
            self.slots.append(rect)
            self.selected_index = len(self.slots) - 1
            self.save()

    def finish_polygon(self):
        if len(self.polygon_points) < 3:
            return
        slot = polygon_from_points(self.polygon_points)
        slot["id"] = len(self.slots) + 1
        self.slots.append(slot)
        self.selected_index = len(self.slots) - 1
        self.polygon_points = []
        self.save()

    def duplicate_selected(self):
        if self.selected_index is None:
            return
        cloned = clone_slot(self.slots[self.selected_index])
        cloned["id"] = len(self.slots) + 1
        self.slots.append(cloned)
        self.selected_index = len(self.slots) - 1
        self.save()

    def mouse(self, event, x, y, flags, params):
        world = self.screen_to_world(x, y)

        if event == cv2.EVENT_MOUSEWHEEL:
            old_zoom = self.zoom
            self.zoom = max(0.25, min(4.0, self.zoom * (1.15 if flags > 0 else 0.87)))
            self.pan_x = int(x - (x - self.pan_x) * (self.zoom / old_zoom))
            self.pan_y = int(y - (y - self.pan_y) * (self.zoom / old_zoom))
            return

        if event == cv2.EVENT_MBUTTONDOWN:
            self.is_panning = True
            self.last_screen = (x, y)
            return

        if event == cv2.EVENT_MBUTTONUP:
            self.is_panning = False
            return

        if event == cv2.EVENT_RBUTTONDOWN:
            index = self.slot_at(world)
            if index is not None:
                self.slots.pop(index)
                self.selected_index = None
                self.save()
            elif self.polygon_points:
                self.polygon_points.pop()
            return

        if event == cv2.EVENT_LBUTTONDOWN:
            if self.mode == "polygon" and self.selected_index is None:
                self.polygon_points.append(world)
                return

            self.drag_handle = self.handle_at(world)
            if self.drag_handle is not None:
                self.last_world = world
                return

            index = self.slot_at(world)
            if index is not None:
                self.selected_index = index
                self.is_moving = True
                self.last_world = world
                return

            if self.mode == "polygon":
                self.selected_index = None
                self.polygon_points.append(world)
                return

            self.selected_index = None
            self.is_drawing_rect = True
            self.start_point = world
            self.current_point = world
            return

        if event == cv2.EVENT_MOUSEMOVE:
            if self.is_panning:
                self.pan_x += x - self.last_screen[0]
                self.pan_y += y - self.last_screen[1]
                self.last_screen = (x, y)
            elif self.drag_handle is not None and self.selected_index is not None:
                self.resize_slot(self.slots[self.selected_index], self.drag_handle, world)
            elif self.is_moving and self.selected_index is not None:
                dx = world[0] - self.last_world[0]
                dy = world[1] - self.last_world[1]
                move_slot(self.slots[self.selected_index], dx, dy)
                self.last_world = world
            elif self.is_drawing_rect:
                self.current_point = world

        if event == cv2.EVENT_LBUTTONUP:
            if self.drag_handle is not None or self.is_moving:
                self.save()
            elif self.is_drawing_rect:
                self.add_rect(self.start_point, world)
            self.drag_handle = None
            self.is_moving = False
            self.is_drawing_rect = False

    def draw_slot(self, img, slot, selected=False):
        color = (0, 255, 255) if selected else (255, 0, 255)
        points = np.array([self.world_to_screen(point) for point in slot_points(slot)], dtype=np.int32)
        cv2.polylines(img, [points], True, color, 2)

        label_pos = tuple(points[0])
        cv2.putText(img, str(slot["id"]), (label_pos[0] + 3, label_pos[1] + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        if selected:
            for point in points:
                cv2.circle(img, tuple(point), HANDLE_RADIUS, (0, 255, 255), -1)

    def draw_details(self, img):
        mode_text = f"Mode: {self.mode.upper()}  Zoom: {self.zoom:.2f}x"
        cv2.putText(img, mode_text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        cv2.putText(
            img,
            "R rect | P polygon | Enter finish polygon | D duplicate | Del remove | Wheel zoom | MMB pan | Q quit",
            (20, 58),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
        )

        if self.selected_index is None:
            details = f"Slots: {len(self.slots)}"
        else:
            slot = self.slots[self.selected_index]
            details = f"Selected #{slot['id']}  type={slot.get('type', 'rect')}  x={slot['x']} y={slot['y']} w={slot['w']} h={slot['h']}"

        cv2.putText(img, details, (20, img.shape[0] - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    def frame(self):
        h, w = self.base_img.shape[:2]
        img = cv2.resize(self.base_img, (int(w * self.zoom), int(h * self.zoom)))
        canvas = np.zeros_like(self.base_img)
        view_h, view_w = canvas.shape[:2]

        x1 = max(0, self.pan_x)
        y1 = max(0, self.pan_y)
        x2 = min(view_w, self.pan_x + img.shape[1])
        y2 = min(view_h, self.pan_y + img.shape[0])
        sx1 = max(0, -self.pan_x)
        sy1 = max(0, -self.pan_y)
        sx2 = sx1 + max(0, x2 - x1)
        sy2 = sy1 + max(0, y2 - y1)
        if x2 > x1 and y2 > y1:
            canvas[y1:y2, x1:x2] = img[sy1:sy2, sx1:sx2]

        for index, slot in enumerate(self.slots):
            self.draw_slot(canvas, slot, index == self.selected_index)

        if self.is_drawing_rect:
            preview = rect_from_points(self.start_point, self.current_point)
            self.draw_slot(canvas, preview)

        if self.polygon_points:
            points = np.array([self.world_to_screen(point) for point in self.polygon_points], dtype=np.int32)
            cv2.polylines(canvas, [points], False, (0, 200, 255), 2)
            for point in points:
                cv2.circle(canvas, tuple(point), 4, (0, 200, 255), -1)

        self.draw_details(canvas)
        return canvas

    def run(self):
        cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Image", self.base_img.shape[1], self.base_img.shape[0])
        cv2.setMouseCallback("Image", self.mouse)

        while True:
            cv2.imshow("Image", self.frame())
            key = cv2.waitKey(1) & 0xFF

            if key == ord("r"):
                self.mode = "rect"
                self.polygon_points = []
            elif key == ord("p"):
                self.mode = "polygon"
                self.selected_index = None
            elif key in (13, 10):
                self.finish_polygon()
            elif key == ord("d"):
                self.duplicate_selected()
            elif key in (8, 127) and self.selected_index is not None:
                self.slots.pop(self.selected_index)
                self.selected_index = None
                self.save()
            elif key == ord("u") and self.slots:
                self.slots.pop()
                self.selected_index = None
                self.save()
            elif key == ord("c"):
                self.slots.clear()
                self.selected_index = None
                self.polygon_points = []
                self.save()
            elif key == ord("q"):
                break

        cv2.destroyAllWindows()


def main():
    SlotEditor().run()


if __name__ == "__main__":
    main()
