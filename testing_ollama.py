import ollama
response = ollama.chat(model='llama2', messages=[
  {
    'role': 'user',
    'content':f"""User issue : I want to rest my password
  Problem category : reset_password

  You are world most intelligent and helpful HelpDesk AI bot.
  Your task is to take the above user issue and the Problem category into consideration and provide a solution to user issue in one sentence.
  Use human touch in your response by using words like ummm, hmm ,etc.
  By your response ,user issue must be resolved. Do not ask for more information.
  Your response should be in one line only.,
  """
  },
])
print(response['message']['content'])
