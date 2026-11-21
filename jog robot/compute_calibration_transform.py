#!/usr/bin/env python3
"""
compute_calibration_transform.py (รองรับทั้ง dobot1 และ dobot2)

วิธีรัน:
  - สำหรับ Dobot 1: python3 compute_calibration_transform.py --robot dobot1
  - สำหรับ Dobot 2: python3 compute_calibration_transform.py --robot dobot2
"""

import argparse
import csv
import numpy as np


def load_points(filename):
    calib_cam, calib_robot = [], []
    valid_cam, valid_robot = [], []

    with open(filename, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            cam = np.array([
                float(row['cam_x']) * 1000.0,
                float(row['cam_y']) * 1000.0,
                float(row['cam_z']) * 1000.0,
            ])
            robot = np.array([
                float(row['robot_x']),
                float(row['robot_y']),
                float(row['robot_z']),
            ])
            if row['label'] == 'calib':
                calib_cam.append(cam)
                calib_robot.append(robot)
            elif row['label'] == 'validation':
                valid_cam.append(cam)
                valid_robot.append(robot)

    return (np.array(calib_cam), np.array(calib_robot),
            np.array(valid_cam), np.array(valid_robot))


def kabsch_rigid_transform(source_pts, target_pts):
    assert source_pts.shape == target_pts.shape
    n = source_pts.shape[0]
    if n < 3:
        raise ValueError(f"ต้องมีอย่างน้อย 3 จุดเพื่อคำนวณ rigid transform (มีแค่ {n} จุด)")

    src_centroid = source_pts.mean(axis=0)
    tgt_centroid = target_pts.mean(axis=0)

    src_centered = source_pts - src_centroid
    tgt_centered = target_pts - tgt_centroid

    H = src_centered.T @ tgt_centered
    U, S, Vt = np.linalg.svd(H)

    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1, 1, d])

    R = Vt.T @ D @ U.T
    t = tgt_centroid - R @ src_centroid

    return R, t


def apply_transform(R, t, points):
    return (R @ points.T).T + t


def report_errors(predicted, actual, label):
    diffs = predicted - actual
    errors_mm = np.linalg.norm(diffs, axis=1)

    print(f"\n--- {label} (n={len(errors_mm)} จุด) ---")
    for i, e in enumerate(errors_mm):
        print(f"  จุดที่ {i+1}: error = {e:.2f} mm  "
              f"(predicted={np.round(predicted[i], 2)}, actual={np.round(actual[i], 2)})")

    print(f"\n  Mean error : {errors_mm.mean():.2f} mm")
    print(f"  Max error  : {errors_mm.max():.2f} mm")
    print(f"  Std dev    : {errors_mm.std():.2f} mm")

    return errors_mm


def main():
    parser = argparse.ArgumentParser(description='Compute Kabsch Rigid Transform for Dobot')
    parser.add_argument('--robot', default='dobot1', choices=['dobot1', 'dobot2'])
    args = parser.parse_args()

    input_file = 'calibration_points.csv' if args.robot == 'dobot1' else f'calibration_points_{args.robot}.csv'
    transform_output = 'camera_to_robot_transform.npz' if args.robot == 'dobot1' else f'camera_to_{args.robot}_transform.npz'

    calib_cam, calib_robot, valid_cam, valid_robot = load_points(input_file)
    print(f"[{args.robot.upper()}] โหลดข้อมูลจาก {input_file}: calib={len(calib_cam)} จุด, validation={len(valid_cam)} จุด")

    if len(calib_cam) < 3:
        print("!! จุด calib ไม่พอสำหรับคำนวณ (ต้องอย่างน้อย 3 จุด ไม่ collinear)")
        return

    R, t = kabsch_rigid_transform(calib_cam, calib_robot)

    print(f"\n=== Rigid Transform ของ {args.robot.upper()} (camera_mm -> {args.robot}_mm) ===")
    print("Rotation matrix R:")
    print(R)
    print("Translation t (mm):")
    print(t)

    calib_pred = apply_transform(R, t, calib_cam)
    report_errors(calib_pred, calib_robot, "Error บนจุด CALIB (fit เอง)")

    if len(valid_cam) >= 1:
        valid_pred = apply_transform(R, t, valid_cam)
        report_errors(valid_pred, valid_robot, "Error บนจุด VALIDATION (ใช้รายงานผล)")
    else:
        print("\n!! ไม่มีจุด validation")

    np.savez(transform_output, R=R, t=t)
    print(f"\nบันทึก transform ลง {transform_output} เรียบร้อยแล้ว!")


if __name__ == '__main__':
    main()
