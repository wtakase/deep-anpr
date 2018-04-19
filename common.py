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
Definitions that don't fit elsewhere.

"""

__all__ = (
    'NP_FONT',
    'REGIONS',
    'DIGITS',
    'HYPHEN',
    'DOT',
    'HIRAGANAS'
    'REGION_NUM'
    'REGIONS_SLICE',
    'CHARS',
    'CODE_LEN',
    'sigmoid',
    'softmax',
)

import numpy


# number_plate_font.otf
NP_FONT = "number_plate_font.otf"
REGIONS = u"北九州金沢高知山口富士山徳島富山奈良ツクバ"
DIGITS = "0123456789"
HYPHEN = "^"
DOT = "."
HIRAGANAS = u"あいうえかきくけこさすそたちつてとなにぬねのはひふほまみむめもやゆよらりるれろわを"
REGION_NUM = 9
REGIONS_SLICE = [
    {"start": 0, "end": 3},
    {"start": 3, "end": 5},
    {"start": 5, "end": 7},
    {"start": 7, "end": 9},
    {"start": 9, "end": 12},
    {"start": 12, "end": 14},
    {"start": 14, "end": 16},
    {"start": 16, "end": 18},
    {"start": 18, "end": 21}
]

# NOTE(wtakase): Currently we don't care about REGION
CHARS = DIGITS + HIRAGANAS + DOT

CODE_LEN = 8

def softmax(a):
    exps = numpy.exp(a.astype(numpy.float64))
    return exps / numpy.sum(exps, axis=-1)[:, numpy.newaxis]

def sigmoid(a):
  return 1. / (1. + numpy.exp(-a))

