docker build -t rid_fnf .
nvidia-docker run \
	-v ${PWD}/data:/data \
	-v ${PWD}/codes:/codes \
	-it rid_fnf;
