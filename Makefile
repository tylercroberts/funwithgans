STORAGE_DIR=..
MODEL_DIR=../models

dcgan-example: dcgan/src/__init__.py
	python dcgan/src/__init__.py --storage-dir='$(STORAGE_DIR)' --model-dir='$(MODEL_DIR)' \
	 --reproducible=$(REPRODUCIBLE) --epochs=1


clean: dcgan/out
	rm -f dcgan/out/*.png
	rm -f dcgan/out/*.gif