# cs267project
aw hell nah

Instructions to run:
```
git clone https://github.com/arvindkalyan/cs267project.git              
# Download facebook.tar.gz from http://snap.stanford.edu/data/facebook.html and place it in the data directory
cd cs267project
python3.10 -m venv ./venv
source venv/bin/activate
pip3 install -r requirements.txt
# This file extracts the facebook data, examines the ego ids and checks for required file types
python3 setup_data.py
# From there, main.py has the ego network loader implementation
python3 main.py
```