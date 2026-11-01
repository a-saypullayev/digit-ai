HANDWRITTEN DIGIT NEURAL NETWORK (DIGITS 1-9)

Architecture:
  324 input neurons (18 x 18 grayscale pixels)
  128 hidden neurons (Linear + ReLU)
  9 output neurons (digits 1 through 9)

RUN ON WINDOWS:
1. Install Python. Python 3.11 or newer is recommended.
2. Extract this folder.
3. Double-click start_windows.bat.
4. Wait for installation, then open http://127.0.0.1:8000
5. Draw a digit and press Recognize.
6. Choose the correct digit and press Teach / Train.
7. Repeat with MANY different handwriting samples for every digit.

The model weights are saved in digit_model.pt.
Labeled examples are saved in training_data.jsonl.
Both files are created automatically beside main.py.

API:
POST /predict
JSON: {"pixels":[324 numbers between 0 and 1]}
Returns: digit, confidence, probabilities.

POST /train
JSON: {"pixels":[324 numbers between 0 and 1],"label":1,"epochs":5}
Returns training example count and loss.

GET /health
Shows server status and number of saved training examples.

ROBLOX:
- Deploy this server to a Python hosting service with persistent disk/storage.
- In Roblox Studio, enable Game Settings > Security > Allow HTTP Requests.
- Put roblox_server_script.lua in ServerScriptService.
- Replace API_URL with your HTTPS server URL.
- Send the 324 pixel values from your Roblox drawing GUI to /predict.
- Make API calls from a server Script, NOT a LocalScript.

IMPORTANT:
- This is a small learning/demo model, not a pre-trained digit recognizer.
- It starts with random weights. It will initially make poor guesses.
- Train with varied examples for every digit. A single drawing is not enough.
- The output has 9 classes (1-9); digit 0 is intentionally not included.
- For a public deployment, protect /train with a secret TRAINING_KEY and add
  the X-Training-Key header to training requests. Do not put that secret in
  a Roblox LocalScript or in a public client.
- Use hosting with persistent storage; otherwise model/data files may be lost
  when the host restarts or redeploys.
