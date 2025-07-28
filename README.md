## 🎨 Doodle Classifier
---
A neural net built from scratch to classify QuickDraw sketches using NumPy.  
Realtime prediction with custom UI — no TensorFlow, just pure NumPy & passion.

## 🧠 About
---
Doodle Classifier is a fully hand-coded neural network trained to recognize doodles from Google’s QuickDraw dataset.  
Designed with simplicity, accuracy, and educational value in mind.

## ✨ Highlights
---
- 🚀 Built using **only NumPy** (no ML frameworks)
- 🎮 **Realtime prediction** using Pygame
- 🧠 Manual backpropagation & forward pass
- 📈 Achieved **85%+ accuracy** on 80,000+ samples
- 💾 Load & save models using `.npy` files
- 🎥 [Demo Video on LinkedIn](https://www.linkedin.com/feed/update/urn:li:activity:7354247671492091904/)

## 👩‍💻 Author Info
---
- 👩‍💻 Name: Rea Sinha  
- 🎓 College: IIIT Guwahati (2023–2027)  
- 💼 Role: Full-Stack Dev | ML/DL Explorer | Blockchain Curious  
- 📧 Email: rea.sinha23b@iiitg.ac.in  
- 📱 Phone: +91 8787230742  
- 🌐 GitHub: [sinharea](https://github.com/sinharea)  
- 🔗 LinkedIn: [Rea Sinha](https://linkedin.com/in/rea-sinha-a33a18356)  
- 🧩 LeetCode: [sinharea](https://leetcode.com/u/sinharea1008/)

## 🧩 Features
---
- Trained on QuickDraw’s **10-class dataset**
- Manual **sigmoid activation & binary cross-entropy**
- Live drawing & prediction using **Pygame canvas**
- Reusable **neural network module** (`neural_net.py`)
- **Real-time feedback** after every sketch

## 🗂️ Project Structure
---
- `train.py` → Trains the neural net using `.npy` data
- `predict.py` → Runs real-time prediction GUI
- `neural_net.py` → Contains neural net logic (forward + backward pass)
- `utils.py` → Helper functions for data preprocessing
- `data/` → Stores input datasets
- `models/` → Stores trained weights
- `ui/` → Optional Pygame UI assets

## ⚙️ Architecture
---
- Input Layer: 784 nodes (28x28 images)
- Hidden Layer: 128 neurons
- Output Layer: 10 neurons
- Activation: Sigmoid
- Loss: Binary Cross Entropy
- Optimizer: Manual Gradient Descent
- Epochs: 100

## 📊 Results
---
- ✅ Accuracy: 85%+
- 🚀 Realtime Latency: < 25ms
- 🧠 Supported Classes:
  - cat, dog, tree, rocket, bird
  - cloud, boat, bicycle, airplane, lion

## 🧰 Tech Stack
---
- Languages: Python
- Libraries: NumPy, Pygame, Matplotlib
- Tools: Jupyter, VSCode, GitHub

## 💻 Usage
---
### 🧱 Install
```bash
git clone https://github.com/sinharea/DoodleClassifier.git
cd DoodleClassifier
pip install numpy pygame matplotlib
```
## 🏁 Run
---
commands:
  - "python train.py      # To train model"
  - "python predict.py    # To launch GUI and predict"

## 📜 License
---
type: MIT
url: "https://opensource.org/licenses/MIT"

## ☎️ Contact
---
email: "rea.sinha23b@iiitg.ac.in"
github: "https://github.com/sinharea"
linkedin: "https://linkedin.com/in/rea-sinha-a33a18356"
leetcode: "https://leetcode.com/u/sinharea1008/"
phone: "+91 8787230742"

## 💬 Final Words
---
quote: "Code with logic. Design with love. Build like it changes lives."
author: "Rea Sinha ✨"
