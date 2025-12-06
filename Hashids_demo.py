from hashids import Hashids
from dotenv import load_dotenv
import os

# โหลดค่าจาก .env
load_dotenv()

# ดึงค่า HASH_SALT และ MIN_LENGTH จาก .env
salt = os.getenv("HASH_SALT", "default_salt")
min_length = int(os.getenv("MIN_LENGTH", 8))

# สร้าง Hashids instance
hashids = Hashids(salt=salt, min_length=min_length)

# Demo: Encode และ Decode
print(f"Salt: {salt}")
print(f"Min Length: {min_length}")
print("-" * 40)

user_id = 11
encoded_user_id = hashids.encode(user_id)
print(f"User ID: {user_id} -> Encoded: {encoded_user_id}")
print(f"Encoded: {encoded_user_id} -> Decoded User ID: {hashids.decode(encoded_user_id)[0]}")
