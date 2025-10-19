import argparse
import sys

import tensorflow as tf


def getArgs():
    parser = argparse.ArgumentParser(
        description="Liczenie kąta punktu na płaszczyźnie."
    )
    parser.add_argument(
        "-A", required=True, help="Macierz A w formacie elem1,elem2;row2"
    )
    parser.add_argument("-b", required=True, help="Macierz b w formacie elem1;elem2")
    return parser.parse_args()


def parse_matrix(matrix_str):
    """
    Parsuje string w formacie:
    "1,2,3;4,5,6;7,8,9" -> tf.Tensor([[1,2,3],[4,5,6],[7,8,9]], dtype=tf.float32)
    """
    rows = matrix_str.strip().split(";")
    matrix = []
    for row in rows:
        elements = row.strip().split(",")
        float_row = [float(el) for el in elements]
        matrix.append(float_row)
    return tf.constant(matrix, dtype=tf.float32)


@tf.function
def solve_linear_system(A_matrix, b_matrix):
    x = tf.linalg.solve(A_matrix, b_matrix)
    return tf.squeeze(x)


def is_solvable(A_matrix):
    det = tf.linalg.det(A_matrix)

    # Jeśli det != 0, układ jest rozwiązywalny
    return tf.not_equal(det, 0.0)


def main():
    args = getArgs()

    A = parse_matrix(args.A)
    b = parse_matrix(args.b)

    if not is_solvable(A):
        print("Niemożliwe do rozwiązania!")
        exit(1)

    solved = solve_linear_system(A, b)
    print(f"Rozwiązanie {solved}")


if __name__ == "__main__":
    sys.exit(main())
