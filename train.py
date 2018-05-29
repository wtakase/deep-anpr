#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2016 Matthew Earl
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
#     The above copyright notice and this permission notice shall be included
#     in all copies or substantial portions of the Software.
# 
#     THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
#     OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
#     MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN
#     NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
#     DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
#     OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE
#     USE OR OTHER DEALINGS IN THE SOFTWARE.


"""
Routines for training the network.

"""


__all__ = (
    'train',
)


import argparse
import functools
import glob
import itertools
import multiprocessing
import os
import random
import sys
import time

import cv2
import numpy
import tensorflow as tf

import common
import gen
import model


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '2'


def code_to_vec(p, code):
    def char_to_vec(c):
        y = numpy.zeros((len(common.CHARS),))
        y[common.CHARS.index(c.replace("-", "^").replace("_", " "))] = 1.0
        return y

    c = numpy.vstack([char_to_vec(c) for c in code])

    return numpy.concatenate([[1. if p else 0], c.flatten()])


def read_data(img_glob):
    for fname in sorted(glob.glob(img_glob)):
        im = cv2.imread(fname)[:, :, 0].astype(numpy.float32) / 255.
        code = fname.split("/")[1][9:9+common.CODE_LEN]
        p = fname.split("/")[1][9+common.CODE_LEN+1] == '1'
        yield im, code_to_vec(p, code)


def unzip(b):
    xs, ys = zip(*b)
    xs = numpy.array(xs)
    ys = numpy.array(ys)
    return xs, ys


def batch(it, batch_size):
    out = []
    for x in it:
        out.append(x)
        if len(out) == batch_size:
            yield out
            out = []
    if out:
        yield out


def mpgen(f):
    def main(q, args, kwargs):
        try:
            for item in f(*args, **kwargs):
                q.put(item)
        finally:
            q.close()

    @functools.wraps(f)
    def wrapped(*args, **kwargs):
        q = multiprocessing.Queue(3) 
        proc = multiprocessing.Process(target=main,
                                       args=(q, args, kwargs))
        proc.start()
        try:
            while True:
                item = q.get()
                yield item
        finally:
            proc.terminate()
            proc.join()

    return wrapped
        

@mpgen
def read_batches(batch_size):
    g = gen.generate_ims()
    def gen_vecs():
        for im, c, p in itertools.islice(g, batch_size):
            yield im, code_to_vec(p, c)

    while True:
        yield unzip(gen_vecs())


def get_loss(y, y_):
    # Calculate the loss from digits being incorrect.  Don't count loss from
    # digits that are in non-present plates.
    digits_loss = tf.nn.softmax_cross_entropy_with_logits(
                                          logits=tf.reshape(y[:, 1:],
                                                     [-1, len(common.CHARS)]),
                                          labels=tf.reshape(y_[:, 1:],
                                                     [-1, len(common.CHARS)]))
    digits_loss = tf.reshape(digits_loss, [-1, common.CODE_LEN])
    digits_loss = tf.reduce_sum(digits_loss, 1)
    digits_loss *= (y_[:, 0] != 0)
    digits_loss = tf.reduce_sum(digits_loss)

    # Calculate the loss from presence indicator being wrong.
    presence_loss = tf.nn.sigmoid_cross_entropy_with_logits(
        logits=y[:, :1], labels=y_[:, :1])
    presence_loss = common.CODE_LEN * tf.reduce_sum(presence_loss)

    return digits_loss, presence_loss, digits_loss + presence_loss


def train(learn_rate, report_steps, batch_size, initial_weights=None, max_steps=0, dropout_ratio=0, output_file="weights.npz"):
    """
    Train the network.

    The function operates interactively: Progress is reported on stdout, and
    training ceases upon `KeyboardInterrupt` at which point the learned weights
    are saved to `weights.npz`, and also returned.

    :param learn_rate:
        Learning rate to use.

    :param report_steps:
        Every `report_steps` batches a progress report is printed.

    :param batch_size:
        The size of the batches used for training.

    :param initial_weights:
        (Optional.) Weights to initialize the network with.

    :param max_steps:
        (Optional.) Max steps to train.

    :param dropout_ratio:
        (Optional.) Dropout ratio.

    :return:
        The learned network weights.

    """
    keep_prob = tf.placeholder(tf.float32)
    x, y, params = model.get_training_model(keep_prob=keep_prob)

    y_ = tf.placeholder(tf.float32, [None, common.CODE_LEN * len(common.CHARS) + 1])

    digits_loss, presence_loss, loss = get_loss(y, y_)
    train_step = tf.train.AdamOptimizer(learn_rate).minimize(loss)

    best = tf.argmax(tf.reshape(y[:, 1:], [-1, common.CODE_LEN, len(common.CHARS)]), 2)
    correct = tf.argmax(tf.reshape(y_[:, 1:], [-1, common.CODE_LEN, len(common.CHARS)]), 2)

    test_xs_all, test_ys_all = unzip(list(read_data("test/*.png")))

    if initial_weights is not None:
        assert len(params) == len(initial_weights)
        assign_ops = [w.assign(v) for w, v in zip(params, initial_weights)]

    init = tf.global_variables_initializer()

    def vec_to_plate(v):
        return "".join(common.CHARS[i] for i in v)

    def do_report(batch_time):
        batch_mask = numpy.random.choice(test_xs_all.shape[0], batch_size)
        test_xs = test_xs_all[batch_mask]
        test_ys = test_ys_all[batch_mask]

        r = sess.run([best,
                      correct,
                      tf.greater(y[:, 0], 0),
                      y_[:, 0],
                      digits_loss,
                      presence_loss,
                      loss],
                     feed_dict={x: test_xs, y_: test_ys, keep_prob: (1 - dropout_ratio)})
        num_correct = numpy.sum(
                        numpy.logical_or(
                            numpy.all(r[0] == r[1], axis=1),
                            numpy.logical_and(r[2] < 0.5,
                                              r[3] < 0.5)))
        r_short = (r[0][:190], r[1][:190], r[2][:190], r[3][:190])
        sample_out = ""
        for b, c, pb, pc in zip(*r_short):
            #print("{} {} <-> {} {}".format(vec_to_plate(c), pc,
            #                               vec_to_plate(b), float(pb)))
            sample_out = "{} {} <-> {} {}".format(vec_to_plate(c).replace(" ", "_").replace("^", "-"), pc,
                                                  vec_to_plate(b).replace(" ", "_").replace("^", "-"), float(pb))
            break
        num_p_correct = numpy.sum(r[2] == r[3])

        print("step: {:d}, num_images: {:d}, number_plate_acc: {:2.02f}%, presence_acc: {:02.02f}%, loss: {}, time: {:02.02f}, {}".format(
            batch_idx,
            batch_idx * batch_size,
            100. * num_correct / (len(r[0])),
            100. * num_p_correct / len(r[2]),
            r[6],
            batch_time,
            sample_out))

    def do_batch(last_time_batch):
        sess.run(train_step,
                 feed_dict={x: batch_xs, y_: batch_ys, keep_prob: (1 - dropout_ratio)})
        if batch_idx != 0 and batch_idx % report_steps == 0:
            batch_time = time.time() - last_time_batch
            do_report(batch_time)
            last_time_batch = time.time()
        return last_time_batch

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.95)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        sess.run(init)
        if initial_weights is not None:
            sess.run(assign_ops)

        try:
            last_batch_idx = 0
            last_batch_time = time.time()
            batch_iter = enumerate(read_batches(batch_size))
            for batch_idx, (batch_xs, batch_ys) in batch_iter:
                last_batch_time = do_batch(last_batch_time)

                if max_steps != 0 and batch_idx >= max_steps:
                    raise KeyboardInterrupt

        except KeyboardInterrupt:
            last_weights = [p.eval() for p in params]
            numpy.savez(output_file, *last_weights)
            return last_weights


if __name__ == "__main__":

    argparser = argparse.ArgumentParser(description="Training script")
    argparser.add_argument("-i", "--input-file", default=None)
    argparser.add_argument("-o", "--output-file", default="weights.npz")
    argparser.add_argument("-b", "--batch-size", default=100, type=int, help="[100]")
    argparser.add_argument("-r", "--report-steps", default=10, type=int, help="[10]")
    argparser.add_argument("-m", "--max-steps", default=100, type=int, help="[100]")
    argparser.add_argument("-l", "--learn-rate", default=0.001, type=float, help="[0.001]")
    argparser.add_argument("-d", "--dropout-ratio", default=0.0, type=float, help="[0.0]")
    args = argparser.parse_args()

    if args.input_file:
        f = numpy.load(args.input_file)
        initial_weights = [f[n] for n in sorted(f.files,
                                                key=lambda s: int(s[4:]))]
    else:
        initial_weights = None

    print("#####")
    print("input_file: %s" % args.input_file)
    print("output_file: %s" % args.output_file)
    print("batch_size: %s" % args.batch_size)
    print("report_steps: %s" % args.report_steps)
    print("max_steps: %s" % args.max_steps)
    print("learn_rate: %s" % args.learn_rate)
    print("dropout_ratio: %s (keep_prob: %s)" % (args.dropout_ratio, 1 - args.dropout_ratio))
    print("#####")

    start_time = time.time()
    """
    train(learn_rate=0.001,
          report_steps=10,
          batch_size=100,
          initial_weights=initial_weights,
          max_steps=5000)
    """
    """
    train(learn_rate=0.001,
          report_steps=500,
          batch_size=100,
          initial_weights=initial_weights,
          max_steps=200000)
    """
    train(learn_rate=args.learn_rate,
          report_steps=args.report_steps,
          batch_size=args.batch_size,
          initial_weights=initial_weights,
          max_steps=args.max_steps,
          dropout_ratio=args.dropout_ratio,
          output_file=args.output_file)
    end_time = time.time()
    print("Elapsed time: %.2f" % (end_time - start_time))
