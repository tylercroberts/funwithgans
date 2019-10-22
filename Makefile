STORAGE_DIR=..
MODEL_DIR=../models
REPRODUCIBLE=''
EPOCHS=1
LR=0.0002
BETA=0.999

dcgan-example: dcgan/src/__init__.py
	make clean
	python dcgan/src/__init__.py --storage-dir='$(STORAGE_DIR)' --model-dir='$(MODEL_DIR)' \
	 --reproducible='$(REPRODUCIBLE)' --epochs='$(EPOCHS)' --lr='$(LR)' --beta='$(BETA)'


clean: dcgan/out
	rm -f dcgan/out/*.png
	rm -f dcgan/out/*.gif