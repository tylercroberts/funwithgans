dcgan-example: dcgan/src/__init__.py
	make clean
	python dcgan/src/__init__.py --config='dcgan\\config.json'


clean: dcgan/out
	rm -f dcgan/out/*.png
	rm -f dcgan/out/*.gif
	rm -f logs/*.log