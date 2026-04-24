from ollama import Client

# Initialize the client with your remote IP
client = Client(host='http://172.16.212.100:11434')

response = client.chat(model='ministral-3:3b', messages=[
  {
    'role': 'user',
    'content': 'เรื่องน่ารู้เกี่ยวกับแมว',
  },
])

print(response['message']['content'])
