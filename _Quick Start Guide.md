📋 Quick Start Guide
1
Setup Environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
2
Train the Agent
python train.py --episodes 2000 --save-freq 500
3
Test and Visualize
python test.py --model models/best_model.pth --render