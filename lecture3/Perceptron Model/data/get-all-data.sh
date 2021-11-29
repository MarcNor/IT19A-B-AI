#! /bin/bash

# For each file, add a download.py line
# Any additional processing on the downloaded file

HERE="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Yelp Reviews dataset
mkdir -p yelp
if [ ! -f yelp/raw_train.csv ]; then
    python download.py 1xeUnqkhuzGGzZKThzPeXe2Vf6Uu_g_xM yelp/raw_train.csv # 12536
fi
if [ ! -f yelp/raw_test.csv ]; then
    python download.py 1G42LXv72DrhK4QKJoFhabVL4IU6v2ZvB yelp/raw_test.csv # 4
fi
if [ ! -f yelp/reviews_with_splits_lite.csv ]; then
    python download.py 1Lmv4rsJiCWVs1nzs4ywA9YI-ADsTf6WB yelp/reviews_with_splits_lite.csv # 1217
fi

