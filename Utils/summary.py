from __future__ import print_function
import tensorflow as tf

class Summary:

    def __init__(self, session):
        self._sess = session
        self._vars = {}
        self._ops = None
        self._writers = {}

    def add_writer(self, dir_sum, name):
        assert isinstance(name, str), "name must be str"
        self._writers[name] = tf.summary.FileWriter(dir_sum, self._sess.graph, filename_suffix=name)

    def add_variable(self, name="name"):
        assert name not in self._vars, "Already has " + name
        var = tf.Variable(0.)
        tf.summary.scalar(name, var)
        self._vars[name] = var

    def build(self):
        self._ops = tf.summary.merge_all()

    def run(self, feed_dict, name, step):
        feed_dict_final = {}
        for key, val in feed_dict.items():
            feed_dict_final[self._vars[key]] = val
        str_summary = self._sess.run(self._ops, feed_dict_final)
        self._writers[name].add_summary(str_summary, step)
        self._writers[name].flush()
