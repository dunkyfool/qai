#!/bin/sh

scp /Users/dunkyfool/QAI/main.py hduser@slave-1:/home/hduser/qai/
scp /Users/dunkyfool/QAI/model/basicCNN.py hduser@slave-1:/home/hduser/qai/model/ 
scp /Users/dunkyfool/QAI/model/utils/initParams.py /hduser@slave-1:/home/hduser/qai/model/utils/ 
scp /Users/dunkyfool/QAI/model/utils/layers.py /hduser@slave-1:/home/hduser/qai/model/utils/ 
