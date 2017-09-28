__author__ = 'nbortolotti'

import tensorflow as tf
import argparse

if __name__ == '__main__':
    pa = argparse.ArgumentParser(description='example to use arithmetic_operators')
    pa.add_argument('--operator', dest='operator', required=True, help='add,subtract')
    pa.add_argument('--value_a', dest='value_a', required=True, help='value a')
    pa.add_argument('--value_b', dest='value_b', required=True, help='value b')


args = pa.parse_args()

a = tf.constant(int(args.value_a))
b = tf.constant(int(args.value_b))

operation = None
if args.operator == 'add':
    operation = tf.add(a, b)
elif args.operator == 'subtract':
    operation = tf.subtract(a, b)

with tf.Session() as ses:
    print ses.run(operation)