from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import cv2
import numpy as np
import onnxruntime as ort
import yaml


@dataclass
class Detection:
    cls_id: int
    cls_name: str
    conf: float
    xyxy: Tuple[int, int, int, int]


class YOLOOnnxDetector:
    def __init__(self, onnx_path: Path, data_yaml: Path, conf_thresh: float = 0.4, iou_thresh: float = 0.45):
        self.session = ort.InferenceSession(str(onnx_path), providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
        with open(data_yaml, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        self.names: List[str] = data["names"]
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        inputs = self.session.get_inputs()
        self.input_name = inputs[0].name
        _, self.input_c, self.input_h, self.input_w = inputs[0].shape

    def _preprocess(self, img: np.ndarray) -> Tuple[np.ndarray, float, Tuple[int, int]]:
        h0, w0 = img.shape[:2]
        r = min(self.input_h / h0, self.input_w / w0)
        new_shape = (int(w0 * r), int(h0 * r))
        resized = cv2.resize(img, new_shape, interpolation=cv2.INTER_LINEAR)
        canvas = np.full((self.input_h, self.input_w, 3), 114, dtype=np.uint8)
        pad_w = (self.input_w - new_shape[0]) // 2
        pad_h = (self.input_h - new_shape[1]) // 2
        canvas[pad_h:pad_h + new_shape[1], pad_w:pad_w + new_shape[0]] = resized
        img_rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        img_trans = img_rgb.transpose(2, 0, 1).astype(np.float32) / 255.0
        img_input = np.expand_dims(img_trans, 0)
        return img_input, r, (pad_w, pad_h)

    def _nms(self, boxes: np.ndarray, scores: np.ndarray, iou_thres: float) -> List[int]:
        idxs = scores.argsort()[::-1]
        keep = []
        while idxs.size > 0:
            i = idxs[0]
            keep.append(i)
            if idxs.size == 1:
                break
            xx1 = np.maximum(boxes[i, 0], boxes[idxs[1:], 0])
            yy1 = np.maximum(boxes[i, 1], boxes[idxs[1:], 1])
            xx2 = np.minimum(boxes[i, 2], boxes[idxs[1:], 2])
            yy2 = np.minimum(boxes[i, 3], boxes[idxs[1:], 3])
            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h
            ovr = inter / ( (boxes[i, 2]-boxes[i,0])*(boxes[i,3]-boxes[i,1]) + (boxes[idxs[1:], 2]-boxes[idxs[1:],0])*(boxes[idxs[1:],3]-boxes[idxs[1:],1]) - inter + 1e-6)
            idxs = idxs[1:][ovr <= iou_thres]
        return keep

    def infer(self, img_bgr: np.ndarray) -> List[Detection]:
        inp, r, (pad_w, pad_h) = self._preprocess(img_bgr)
        outputs = self.session.run(None, {self.input_name: inp})
        pred = outputs[0][0]  # [num, 85] for yolov5 format: x,y,w,h,conf,classes...
        boxes = []
        scores = []
        classes = []
        for row in pred:
            obj_conf = row[4]
            cls_scores = row[5:]
            cls_id = int(np.argmax(cls_scores))
            cls_conf = cls_scores[cls_id] * obj_conf
            if cls_conf < self.conf_thresh:
                continue
            cx, cy, w, h = row[:4]
            x1 = (cx - w / 2 - pad_w) / r
            y1 = (cy - h / 2 - pad_h) / r
            x2 = (cx + w / 2 - pad_w) / r
            y2 = (cy + h / 2 - pad_h) / r
            boxes.append([x1, y1, x2, y2])
            scores.append(float(cls_conf))
            classes.append(cls_id)
        if not boxes:
            return []
        boxes_np = np.array(boxes)
        scores_np = np.array(scores)
        keep = self._nms(boxes_np, scores_np, self.iou_thresh)
        detections: List[Detection] = []
        for i in keep:
            x1, y1, x2, y2 = boxes_np[i]
            cls_id = classes[i]
            detections.append(
                Detection(
                    cls_id=cls_id,
                    cls_name=self.names[cls_id] if 0 <= cls_id < len(self.names) else str(cls_id),
                    conf=scores_np[i],
                    xyxy=(int(x1), int(y1), int(x2), int(y2)),
                )
            )
        return detections


class GestureDetector:
    """Gesture detection using MediaPipe Hands when available, with heuristic fallback.

    Jetson-friendly path: attempts to import `mediapipe` (if wheel or build exists).
    If unavailable, falls back to the fast OpenCV heuristic.
    """

    def __init__(self):
        self._mp_ok = False
        self._hands = None
        # Hysteresis thresholds on normalized, scale-invariant distances
        self._open_thr = 0.40
        self._closed_thr = 0.25
        from collections import deque
        self._history = deque(maxlen=5)
        self._last_stable = "none"
        self.last_metric = 0.0
        try:
            import mediapipe as mp
            self._mp = mp
            self._hands = mp.solutions.hands.Hands(
                static_image_mode=False,
                max_num_hands=1,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
                model_complexity=0,
            )
            self._mp_ok = True
        except Exception:
            self._mp_ok = False

    def _heuristic(self, img_bgr: np.ndarray) -> Optional[str]:
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        lower = np.array([0, 30, 60])
        upper = np.array([20, 150, 255])
        mask1 = cv2.inRange(img, lower, upper)
        lower2 = np.array([160, 30, 60])
        upper2 = np.array([180, 150, 255])
        mask = cv2.bitwise_or(mask1, cv2.inRange(img, lower2, upper2))
        mask = cv2.medianBlur(mask, 5)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        c = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(c)
        if area < 1000:
            return None
        hull = cv2.convexHull(c)
        hull_area = cv2.contourArea(hull) + 1e-6
        solidity = area / hull_area
        x, y, w, h = cv2.boundingRect(c)
        extent = area / (w * h + 1e-6)
        if solidity < 0.85 and extent > 0.35:
            return "open_hand"
        if solidity >= 0.9 and extent < 0.3:
            return "closed_hand"
        return None

    def detect(self, img_bgr: np.ndarray) -> Optional[str]:
        if not self._mp_ok:
            # Fallback returns open_hand/closed_hand/None
            raw = self._heuristic(img_bgr)
            return raw
        # MediaPipe expects RGB
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        res = self._hands.process(img_rgb)
        if not res.multi_hand_landmarks:
            self._history.append("none")
            return None
        lm = res.multi_hand_landmarks[0]
        pts = np.array([[p.x, p.y] for p in lm.landmark], dtype=np.float32)
        palm_idx = [0, 1, 5, 9, 13, 17]
        palm = pts[palm_idx].mean(axis=0)
        tips_idx = [4, 8, 12, 16, 20]
        tips = pts[tips_idx]
        min_xy = pts.min(axis=0)
        max_xy = pts.max(axis=0)
        scale = float(max(max_xy - min_xy)) + 1e-6
        d = float(np.linalg.norm(tips - palm, axis=1).mean() / scale)
        self.last_metric = d
        if d > self._open_thr:
            self._history.append("open_hand")
        elif d < self._closed_thr:
            self._history.append("closed_hand")
        else:
            self._history.append("neutral")
        if len(self._history) < self._history.maxlen:
            return self._history[-1] if self._history[-1] != "neutral" else None
        vals, counts = np.unique(list(self._history), return_counts=True)
        majority = vals[np.argmax(counts)]
        if majority not in ("neutral", "none"):
            self._last_stable = majority
            return majority
        return self._last_stable if self._last_stable not in ("neutral", "none") else None


class POSession:
    def __init__(self, prices: Dict[str, float]):
        self.prices = prices
        self.active = False
        self.items: List[Tuple[str, float]] = []
        self._last_item: Optional[str] = None
        self._last_time: float = 0.0

    def start(self):
        self.active = True
        self.items = []
        self._last_item = None
        self._last_time = 0.0

    def end(self) -> float:
        self.active = False
        total = sum(price for _, price in self.items)
        return total

    def maybe_add_item(self, name: str, conf: float, min_conf: float = 0.5, debounce_ms: int = 400) -> Optional[Tuple[str, float]]:
        if not self.active:
            return None
        if conf < min_conf:
            return None
        now = time.time() * 1000
        if self._last_item == name and (now - self._last_time) < debounce_ms:
            return None
        self._last_item = name
        self._last_time = now
        price = self.prices.get(name)
        if price is None:
            return None
        self.items.append((name, price))
        return name, price
