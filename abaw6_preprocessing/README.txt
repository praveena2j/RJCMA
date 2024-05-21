# Create environment
conda create -n pre python=3.11
conda activate pre

# Use pip to install any missing libs
pip install opencv-python tqdm opensmile resampy vosk
pip install --upgrade nltk
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install deepmultilingualpunctuation
# For ffmpeg, don't use pip, as it won't install the executable.
conda install -c conda-forge ffmpeg

# Also download https://alphacephei.com/vosk/models/vosk-model-en-us-0.22.zip for transcribing the video with timestamp
# Put it in folder load, together with "base" and "project", eventually u will have "base", "project", and "load" under the root dir.

# To set up nltk, type "python" to go into python console, then execute
# import nltk
# nltk.download('punkt')

# The time-cost bottleneck for the preprocessing is the speech transcript generation (using vosk without GPU support) and text embedding alignment.
# vosk is free and provides timestamp. Google also provide timestamp, but it is not free.

# Now the environment is ready. Go to config.py, edit all paths to fit your system. Note that I wrote comments to help understand the config.
# Then go to `main.py`, edit the path accordingly.
# I strongly recommend to run the code line by line with an IDE.
