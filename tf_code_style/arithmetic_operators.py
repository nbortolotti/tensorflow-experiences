__author__ = 'nbortolotti'

import tensorflow as tf
import argparse


def main():
    pa = argparse.ArgumentParser(description='example to use arithmetic_operators')
    pa.add_argument('--operator', dest='operator', help='add,subtract', type=str, default='add')
    pa.add_argument('--value_a', dest='value_a', help='value a', type=int, default=1)
    pa.add_argument('--value_b', dest='value_b', help='value b', type=int, default=1)

    args = pa.parse_args()

    a = tf.constant(int(args.value_a))
    b = tf.constant(int(args.value_b))

    operation = None
    if args.operator == 'add':
        operation = add_operation(a, b)
    elif args.operator == 'subtract':
        operation = subtract_operation(a, b)

    with tf.Session() as ses:
        print ses.run(operation)


def add_operation(a, b):
    operation = tf.add(a, b)
    return operation


def subtract_operation(a, b):
    operation = tf.subtract(a, b)
    return operation


if __name__ == '__main__':
    main()
