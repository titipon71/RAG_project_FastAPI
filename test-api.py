#!/usr/bin/env python3
import json
import ssl
import sys
import uuid
from urllib import error, request


API_URL = "http://thinkhub.kmutnb.ac.th/fastapi/api/v1/chat/completions"
API_KEY = "sk-RxixLIhxKlOlAC9_5m-PuhL8mF8xhXjpD2vB70a4w_g"
CHANNEL_ID = "vL1Xox3dAjq54yQW8YGk"
MESSAGE = "ช่วยสรุปข้อมูลในเอกสารนี้ให้หน่อย"
CONVERSATION_ID = "123"  # Optional, can be auto-generated if not provided
TIMEOUT_SECONDS = 90
VERIFY_TLS = True
STREAM_MODE = True


def main() -> int:
	if not API_KEY or API_KEY == "PASTE_YOUR_API_KEY":
		print("Error: set API_KEY in this file before running.", file=sys.stderr)
		return 1

	if not CHANNEL_ID or CHANNEL_ID == "PASTE_HASHED_CHANNEL_ID":
		print("Error: set CHANNEL_ID in this file before running.", file=sys.stderr)
		return 1

	conversation_id = CONVERSATION_ID or f"test-{uuid.uuid4().hex[:12]}"

	payload = {
		"channel_id": CHANNEL_ID,
		"conversation_id": conversation_id,
		"messages": [
			{
				"role": "user",
				"content": MESSAGE,
			}
		],
	}
	if STREAM_MODE:
		payload["stream"] = True

	headers = {
		"Authorization": f"Bearer {API_KEY}",
		"Content-Type": "application/json",
	}

	print("=== Request ===")
	print(f"URL: {API_URL}")
	print(json.dumps(payload, ensure_ascii=False, indent=2))

	body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
	req = request.Request(
		API_URL,
		data=body,
		headers=headers,
		method="POST",
	)

	context = None
	if not VERIFY_TLS:
		context = ssl._create_unverified_context()

	try:
		with request.urlopen(
			req,
			timeout=TIMEOUT_SECONDS,
			context=context,
		) as response:
			status_code = response.getcode()

			if STREAM_MODE:
				content_type = response.headers.get("Content-Type", "")
				print("\n=== Response ===")
				print(f"Status: {status_code}")
				print(f"Content-Type: {content_type}")
				print("\n=== Stream Output ===")

				full_text = ""
				for raw_line in response:
					line = raw_line.decode("utf-8", errors="replace").strip()
					if not line:
						continue

					if line.startswith("data:"):
						data_str = line[5:].strip()
						if data_str == "[DONE]":
							print("\n[STREAM DONE]")
							break

						try:
							event = json.loads(data_str)
						except ValueError:
							print(data_str)
							continue

						token = event.get("token") if isinstance(event, dict) else None
						if token:
							print(token, end="", flush=True)
							full_text += token
						elif isinstance(event, dict) and event.get("error"):
							print(f"\n[STREAM ERROR] {event.get('error')}")
						else:
							print(f"\n[EVENT] {json.dumps(event, ensure_ascii=False)}")
					else:
						print(line)

				if full_text:
					print("\n\n=== Full Text ===")
					print(full_text)

				return 0 if 200 <= status_code < 300 else 3

			response_body = response.read().decode("utf-8", errors="replace")
	except error.HTTPError as exc:
		status_code = exc.code
		response_body = exc.read().decode("utf-8", errors="replace")
	except error.URLError as exc:
		print(f"Request failed: {exc}", file=sys.stderr)
		return 2
	except TimeoutError:
		print("Request failed: timeout", file=sys.stderr)
		return 2
	except Exception as exc:
		print(f"Request failed: {exc}", file=sys.stderr)
		return 2

	print("\n=== Response ===")
	print(f"Status: {status_code}")

	try:
		data = json.loads(response_body)
		print(json.dumps(data, ensure_ascii=False, indent=2))
	except ValueError:
		print(response_body)

	return 0 if 200 <= status_code < 300 else 3


if __name__ == "__main__":
	raise SystemExit(main())
