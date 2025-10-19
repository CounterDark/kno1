import argparse
import math
import sys

import numpy as np
import tensorflow as tf


def getArgs():
    parser = argparse.ArgumentParser(
        description="Liczenie kąta punktu na płaszczyźnie."
    )
    parser.add_argument("-x", required=True, type=float, help="Koordynat x punktu")
    parser.add_argument("-y", required=True, type=float, help="Koordynat y punktu")
    parser.add_argument(
        "-a", required=True, type=float, help="Kąt punktu do obliczenia"
    )
    return parser.parse_args()


@tf.function
def rotate_point(x, y, angle_deg):
    """
    Obrót punktu (x, y) wokół (0,0) o kąt angle_deg (w stopniach) przy użyciu NumPy.
    """
    # Zamiana stopni na radiany
    angle_rad = np.deg2rad(angle_deg)

    # Macierz obrotu
    rotation_matrix = tf.constant(
        [
            [math.cos(angle_rad), -math.sin(angle_rad)],
            [math.sin(angle_rad), math.cos(angle_rad)],
        ],
        dtype=tf.float32,
    )

    # Punkt jako tensor kolumnowy
    point = tf.constant([[x], [y]], dtype=tf.float32)

    # Mnożenie macierzy
    rotated_point = tf.matmul(rotation_matrix, point)

    print(rotated_point)

    return rotated_point[0, 0], rotated_point[1, 0]


def main():
    args = getArgs()

    x_rot, y_rot = rotate_point(args.x, args.y, args.a)
    print(f"Nowy punkt (NumPy): ({x_rot:.2f}, {y_rot:.2f})")


if __name__ == "__main__":
    sys.exit(main())
