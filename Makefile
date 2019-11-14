dcgan-example: dcgan/train.py, dcgan/__init__.py  dcgan/src/networks.py dcgan/src/utils.py
	make clean
	python dcgan/train.py --config='dcgan\\config.json'

cyclegan-example: cyclegan/src/__init__.py cyclegan/src/networks.py cyclegan/src/utils.py cyclegan/src/models.py
	make clean
	python cyclegan/src/__init__.py --config='cyclegan\\config.json'

clean: dcgan/out
	rm -f dcgan/out/*.png
	rm -f dcgan/out/*.gif
	rm -f dcgan/out/fakes/*.png
	rm -f logs/*.log