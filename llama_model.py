import ollama

def llama_response(intent,final_transcription):
  llama_prompt=f"""
  User issue : {final_transcription}
  Problem category : {intent}

  You are world most intelligent and helpful HelpDesk AI bot.
  Your task is to take the above user issue and the Problem category into consideration and provide a solution to user issue in one sentence.
  Use human touch in your response by using words like ummm, hmm ,etc. Do not use emoji or gestures
  By your response ,user issue must be resolved. Do not ask for more information.
  Your response should be in one line only.
  """

  response = ollama.chat(model='llama2', messages=[
    {
      'role': 'user',
      'content': llama_prompt
    },
  ])
  return response['message']['content']
