import requests
import os
from dotenv import load_dotenv
import random

# Load environment variables from .env file: HUGGINGFACE_TOKEN
load_dotenv()

# Description of Bot
# Which tool did you use? Why?
    # We used the Hugging Face API for emotion detection because it provides a modern, powerful, and easy-to-use 
    # API for analyzing text and understanding the emotion.
# How did you decide what responses to write?
    # We generated responses based on the three given situations: Acknowledge, Suggest, and Reinforce.
    # For simplicity, we just created three that will be randomly selected for each of the situations for every emotion.
# What makes your chatbot empathic?
    # Right now the bot is limited in how it can reply to the user but it can accurately detect the emotion based on the text
    # and respond in a way that is appropriate for the situation rather than fully randomly selecting a response.
    # If expanded to have more strategies and detectable words, the bot can be easily improved though it is somewhat rule-based at the moment.

class ChatBot:
    def __init__(self):
        self.api_token = os.getenv("HUGGINGFACE_TOKEN")
        if not self.api_token:
            raise ValueError("Missing API token.")
        
        # Emotion detection API that receives the text you give
        self.emotion_api_url = "https://api-inference.huggingface.co/models/j-hartmann/emotion-english-distilroberta-base"
        # can return these emotions: sadness, joy, love, anger, fear, surprise, disgust, neutral
        # The setup of the returned value is: {'label': 'anger', 'score': 0.004419783595949411}

    def detect_emotion(self, text):
        # Detect emotion in text using Hugging Face API
        headers = {"Authorization": f"Bearer {self.api_token}"}
        data = {"inputs": text}
        
        try:
            response = requests.post(self.emotion_api_url, headers=headers, json=data)
            if response.status_code == 200:
                result = response.json()
                if isinstance(result, list) and len(result) > 0:
                    # Get the emotion with highest score
                    # Returns like [[{'label': 'sadness', 'score': 0.95}, ...]], get the full list
                    emotions = result[0] 
                    top_emotion = max(emotions, key=lambda x: x['score'])
                    return top_emotion['label']
                else:
                    raise RuntimeError("Result was blank")
            else:
                raise RuntimeError(f"API Error: {response.status_code} - {response.text}")
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Network error: {e}")
        except Exception as e:
            raise RuntimeError(f"Unexpected error: {e}")

    
    def get_response(self, emotion, text_input):
        # Select response based on detected emotion and user input context
        # Analyze user input for specific words
        user_lower = text_input.lower()
        
        # For better empathy, the bot can choose different strategies based on words used
        response_list = {
            "sadness": {
                "acknowledge": [
                    "I can hear that you're going through a really difficult time right now.",
                    "It sounds like you're carrying a heavy emotional burden.",
                    "I can feel the sadness in your words, and that's completely valid."
                ],
                "suggest": [
                    "Sometimes when we're feeling this way, it helps to talk about what's weighing on your heart.",
                    "Would it help to share what's been making you feel so down lately?",
                    "Have you been able to reach out to anyone close to you about how you're feeling?"
                ],
                "reinforce": [
                    "You don't have to go through this alone - I'm here to listen.",
                    "Your feelings are completely valid, and it's okay to not be okay right now.",
                    "Even in the darkest moments, please remember that this feeling won't last forever."
                ]
            },
            "joy": {
                "acknowledge": [
                    "I can feel your excitement and happiness radiating through your words!",
                    "It's wonderful to hear such positive energy from you!",
                    "Your joy is absolutely contagious - thank you for sharing this moment!"
                ],
                "suggest": [
                    "This sounds like something worth celebrating! Tell me more about what happened.",
                    "I'd love to hear all the details about what's making you so happy!",
                    "What an amazing thing to experience! How are you planning to celebrate?"
                ],
                "reinforce": [
                    "Hold onto this feeling - you deserve all the happiness coming your way.",
                    "These are the moments that make life beautiful. Savor every bit of it!",
                    "Your happiness reminds me that there's so much good in the world."
                ]
            },
            "anger": {
                "acknowledge": [
                    "I can sense your frustration, and it's completely understandable to feel this way.",
                    "Your anger is valid - something has clearly upset you deeply.",
                    "I hear how frustrated and upset you are right now."
                ],
                "suggest": [
                    "When we're feeling this angry, it might help to take a few deep breaths before we react.",
                    "Would it help to talk through what happened that made you feel this way?",
                    "Sometimes stepping away from the situation for a moment can give us clarity."
                ],
                "reinforce": [
                    "It's okay to feel angry - it shows that something important to you was affected.",
                    "I'm here to listen without judgment while you work through these feelings.",
                    "Your feelings matter, and I want to understand what's bothering you."
                ]
            },
            "fear": {
                "acknowledge": [
                    "I can hear the worry and anxiety in what you're sharing with me.",
                    "It's completely natural to feel scared or anxious about uncertain situations.",
                    "Fear can be overwhelming, and I recognize how difficult this must be for you."
                ],
                "suggest": [
                    "When anxiety feels overwhelming, sometimes focusing on what we can control helps.",
                    "Have you tried any grounding techniques when you feel this anxious?",
                    "Would it help to break down what you're worried about into smaller, manageable pieces?"
                ],
                "reinforce": [
                    "You're braver than you think for facing these fears and talking about them.",
                    "Remember that you've gotten through difficult times before, and you can get through this too.",
                    "I'm here to support you through whatever you're facing."
                ]
            },
            "surprise": {
                "acknowledge": [
                    "Wow, that sounds like it caught you completely off guard!",
                    "I can imagine how unexpected and surprising that must have been!",
                    "Life certainly has a way of throwing us curveballs, doesn't it?"
                ],
                "suggest": [
                    "How are you processing this unexpected news or event?",
                    "Tell me more about how this surprise is affecting you.",
                    "What's going through your mind now that this has happened?"
                ],
                "reinforce": [
                    "Sometimes unexpected things can lead to wonderful opportunities.",
                    "It's okay to feel unsettled by surprises - they can be a lot to process.",
                    "Whatever this surprise brings, you have the strength to handle it."
                ]
            },
            "disgust": {
                "acknowledge": [
                    "I can tell that something has really bothered or upset you.",
                    "It sounds like you encountered something truly unpleasant.",
                    "Your reaction is completely understandable - that does sound awful."
                ],
                "suggest": [
                    "Would it help to talk about what happened and how it made you feel?",
                    "Sometimes sharing these unpleasant experiences can help us process them.",
                    "Is there anything specific about this situation that's still bothering you?"
                ],
                "reinforce": [
                    "It's perfectly normal to feel disgusted or repulsed by certain things.",
                    "Your feelings about this are completely justified and valid.",
                    "You don't have to keep those unpleasant feelings bottled up inside."
                ]
            },
            "neutral": {
                "acknowledge": [
                    "I'm here and ready to listen to whatever you'd like to share.",
                    "Thank you for reaching out - what's on your mind today?",
                    "I appreciate you taking the time to talk with me. How are you feeling?"
                ],
                "suggest": [
                    "Is there anything specific you'd like to talk about or explore together?",
                    "What would be most helpful for you to discuss right now?",
                    "How has your day been going so far?"
                ],
                "reinforce": [
                    "I'm here to support you in whatever way you need.",
                    "Your thoughts and feelings are always welcome here.",
                    "Take your time - there's no pressure to share anything you're not ready to discuss."
                ]
            }
        }
        
        # Use certain words as the context
        help_words = ['help', 'advice', 'suggest', 'should', 'what do', 'how do', '?']
        intense_words = ['very', 'really', 'extremely', 'so', 'completely', 'totally']
        
        # Have neutral as a fallback if the emotion isnt found
        emotion_responses = response_list.get(emotion, response_list["neutral"])
        
        # Choose strategy based on input analysis
        if any(word in user_lower for word in help_words):
            strategy = "suggest"
        elif any(word in user_lower for word in intense_words):
            strategy = "reinforce"
        else:
            strategy = "acknowledge"
        
        return random.choice(emotion_responses[strategy])
    
    def chat(self):
        print("Empathetic ChatBot :-)")
        print("Type 'quit' to end the conversation.")
        print("=" * 50)
        
        while True:
            try:
                text_input = input("\nYou: ").strip()
                
                if text_input.lower() == 'quit':
                    break

                # Detect emotion
                try:
                    emotion = self.detect_emotion(text_input)
                    response = self.get_response(emotion, text_input)
                    
                    print(f"\nBot: {response}")
                    
                        
                except RuntimeError as e:
                    print(f"\nError Occured: \n{e}")
                    continue
                
            except KeyboardInterrupt:
                break
            except EOFError:
                print("\nText Interrupted")
                break
            except Exception as e:
                print(f"\nError Occured: \n{e}")

def main():
    chatbot = ChatBot()
    chatbot.chat()

if __name__ == "__main__":
    main()