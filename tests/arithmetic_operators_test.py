__author__ = 'nbortolotti'

import tensorflow as tf
from unittest import TestCase
from arithmetic_operators import add_operation, subtract_operation


class GeneralMain(TestCase):
    def test_main(self):
        pass

    def test_add(self):
        a = tf.constant(int(3))
        b = tf.constant(int(4))
        c = tf.constant(int(7))

        result = add_operation(a, b)
        self.assertTrue(tf.assert_equal(result, c))

    def test_subtract(self):
        a = tf.constant(int(10))
        b = tf.constant(int(3))
        c = tf.constant(int(7))

        result = subtract_operation(a, b)
        self.assertTrue(tf.assert_equal(result, c))