#!/bin/sh

scp hduser@slave-1:/home/hduser/qai/main.py /Users/dunkyfool/QAI/
scp -r hduser@slave-1:/home/hduser/qai/model/basicCNN*.py /Users/dunkyfool/QAI/model/
#scp hduser@slave-1:/home/hduser/qai/model/utils/initParams.py /Users/dunkyfool/QAI/model/utils/
#scp hduser@slave-1:/home/hduser/qai/model/utils/layers.py /Users/dunkyfool/QAI/model/utils/
scp hduser@slave-1:/home/hduser/qai/model/utils/*.py /Users/dunkyfool/QAI/model/utils/
