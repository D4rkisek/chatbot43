from flask import Flask, render_template, request, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Initialize model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")

app = Flask(__name__)

@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/get", methods=["GET", "POST"])
def chat():
    if request.method == "POST":
        msg = request.form["msg"]
    else:  # For GET request, just to make sure it doesn't break
        msg = request.args.get("msg", "")

    return jsonify({"response": get_Chat_response(msg)})

def get_Chat_response(text):
    chat_history_ids = None
    response = ""
    for step in range(1):  # Adjust the range if you want more interactions
        new_user_input_ids = tokenizer.encode(text + tokenizer.eos_token, return_tensors='pt')
        bot_input_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1) if chat_history_ids is not None else new_user_input_ids
        chat_history_ids = model.generate(bot_input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)
        text = tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
        response += text
    return response

if __name__ == '__main__':
    app.run(debug=True)
