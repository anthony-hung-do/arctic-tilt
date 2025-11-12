import os
from dotenv import load_dotenv
import wandb
from huggingface_hub import login, whoami

def setup_authentication():

    load_dotenv()

    WANDB_API_KEY = os.getenv('WANDB_API_KEY')
    HF_TOKEN = os.getenv('HF_TOKEN')

    wandb.login(key=WANDB_API_KEY)
    try:
        user = wandb.api.viewer()
        print(" Đã đăng nhập thành công với Weights & Biases!")
        print("Tên tài khoản:", user.get('username', 'Không xác định'))
    except Exception as e:
        print(" Chưa đăng nhập hoặc token không hợp lệ.")
        print("Lỗi:", e)

    login(HF_TOKEN)
    try:
        info = whoami()
        print(" Đã đăng nhập thành công với Hugging Face!")
        print("Tên tài khoản:", info["name"])
        print("Vai trò:", info.get("type", "Không xác định"))
    except Exception as e:
        print(" Chưa đăng nhập hoặc token không hợp lệ.")
        print("Lỗi:", e)