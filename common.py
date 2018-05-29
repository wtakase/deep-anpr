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
    'HIRAGANAS',
    'REGION_NUM',
    'REGION_LIST',
    'REGION_MAP',
    'CHARS',
    'CODE_LEN',
    'sigmoid',
    'softmax',
)

import numpy


# number_plate_font.otf
NP_FONT = "number_plate_font.otf"
DIGITS = "0123456789"
HYPHEN = "^"
DOT = "."
SPACE = " "
HIRAGANAS = u"あいうえかきくけこさすそたちつてとなにぬねのはひふほまみむめもやゆよらりるれろわを"

REGION_NUM = 118
REGIONS = ""
for i in range(REGION_NUM):
    REGIONS += chr(ord(u"\u00c0") + i)

REGION_LIST = [
"adachi",     "ibaraki",     "kurashiki",  "noda",         "shonan",
"aizu",       "ishikawa",    "kurume",     "numazu",       "sodegaura",
"akita",      "itinomiya",   "kushiro",    "obihiro",      "suginami",
"amami",      "iwaki",       "kyoto",      "oita",         "suwa",
"aomori",     "iwate",       "maebashi",   "okayama",      "suzuka",
"asahikawa",  "izumi",       "matsumoto",  "okazaki",      "takasaki",
"chiba",      "izu",         "mie",        "okinawa",      "tama",
"chikuho",    "kagawa",      "mikawa",     "oomiya",       "tochigi_hiragana",
"ehime",      "kagoshima",   "mito",       "osaka",        "tokorozawa",
"fujisan",    "kanazawa",    "miyagi",     "owarikomaki",  "tokushima",
"fukui",      "kashiwa",     "miyazaki",   "sagami",       "tottori",
"fukuoka",    "kasugai",     "morioka",    "saga",         "toyama",
"fukushima",  "kasukabe",    "muroran",    "saitama",      "toyohashi",
"fukuyama",   "kawagoe",     "nagano",     "sakai",        "toyota",
"gifu",       "kawaguchi",   "nagaoka",    "sapporo",      "tsuchiura",
"gunma",      "kawasaki",    "nagasaki",   "sasebo",       "tsukuba",
"hachinohe",  "kitakyushu",  "nagoya",     "sendai",       "utsunomiya",
"hachioji",   "kitami",      "naniwa",     "setagaya",     "wakayama",
"hakodate",   "kobe",        "nara",       "shiga",        "yamagata",
"hamamatsu",  "kochi",       "narashino",  "shimane",      "yamaguchi",
"hida",       "kooriyama",   "narita",     "shimonoseki",  "yamanashi",
"himeji",     "koshigaya",   "nasu",       "shinagawa",    "yokohama",
"hiraizumi",  "kumagaya",    "nerima",     "shizuoka",
"hiroshima",  "kumamoto",    "niigata",    "shonai"
]

REGION_MAP = {}
for i in range(REGION_NUM):
    REGION_MAP[REGIONS[i]] = REGION_LIST[i]

CHARS = DIGITS + HIRAGANAS + DOT + HYPHEN + SPACE + REGIONS

CODE_LEN = 10

def softmax(a):
    exps = numpy.exp(a.astype(numpy.float64))
    return exps / numpy.sum(exps, axis=-1)[:, numpy.newaxis]

def sigmoid(a):
  return 1. / (1. + numpy.exp(-a))
