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

id = 8
encoded_id = hashids.encode(id)
print(f"User ID: {id} -> Encoded: {encoded_id}")
print(f"Encoded: {encoded_id} -> Decoded User ID: {hashids.decode(encoded_id)[0]}")
