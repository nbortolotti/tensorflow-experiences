__author__ = 'nbortolotti'

import tensorflow as tf
import arithmetic_operators

from unittest import TestCase


class GeneralMain(TestCase):
    def test_main(self):
        pass

    def test_add(self):
        a = tf.constant(int(3))
        b = tf.constant(int(4))
        c = tf.constant(int(7))

        result = arithmetic_operators.add_operation(a, b)
        self.assertTrue(tf.assert_equal(result, c))

    def test_subtract(self):
        a = tf.constant(int(10))
        b = tf.constant(int(3))
        c = tf.constant(int(7))

        result = arithmetic_operators.subtract_operation(a, b)
        self.assertTrue(tf.assert_equal(result, c))