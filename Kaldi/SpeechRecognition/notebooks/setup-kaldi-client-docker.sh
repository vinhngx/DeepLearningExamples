apt-get update && apt-get install -y jupyter
apt-get install -y python3-pyaudio \
                   python-pyaudio \
                   libasound-dev \
                   portaudio19-dev \
                   libportaudio2 \
                   libportaudiocpp0 \
                   libsndfile1 \
                   alsa-base \
                   alsa-utils vim 

python3 -m pip uninstall -y pip
apt install python3-pip --reinstall
pip3 install matplotlib soundfile librosa sounddevice
jupyter notebook --allow-root --ip 0.0.0.0 --notebook-dir=/Kaldi/notebooks

