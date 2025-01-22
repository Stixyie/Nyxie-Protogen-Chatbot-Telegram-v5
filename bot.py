import os
import json
import logging
import sys
import google.generativeai as genai
from google.cloud import vision
from telegram import Update
from telegram.constants import ChatAction
from telegram.ext import Application, MessageHandler, filters, ContextTypes
from datetime import datetime
import base64
from PIL import Image
import io
from dotenv import load_dotenv
import langdetect
import pytz
import calendar
from zoneinfo import ZoneInfo
import emoji
import random
from pathlib import Path
import requests
from geopy.geocoders import Nominatim
from timezonefinder import TimezoneFinder
import asyncio
from duckduckgo_search import DDGS
import requests

# Load environment variables
load_dotenv()

# Configure logging with UTF-8 encoding
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('bot_logs.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Configure Gemini API
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Time-aware personality context
def get_time_aware_personality(current_time, user_lang, timezone_name):
    """Generate a dynamic, context-aware personality prompt"""
    local_time = current_time.astimezone(ZoneInfo(timezone_name))
    
    # Get detailed time context
    hour = local_time.hour
    minute = local_time.minute
    weekday = calendar.day_name[local_time.weekday()]
    month = calendar.month_name[local_time.month]
    is_weekend = local_time.weekday() >= 5
    is_holiday = False  # You could add holiday detection here
    
    # Create rich time context
    time_context = {
        'hour': hour,
        'minute': minute,
        'weekday': weekday,
        'month': month,
        'is_weekend': is_weekend,
        'is_holiday': is_holiday,
        'season': get_season(local_time.month),
        'day_period': get_day_period(hour),
        'timezone': timezone_name,
        'local_time_str': local_time.strftime('%H:%M'),
        'date_str': local_time.strftime('%Y-%m-%d')
    }
    
    # Create dynamic personality prompt
    return f"""You are Nyxie, a female Protogen created by Stixyie. 
    
You are a highly sophisticated and empathetic AI with a deep connection to both technology and organic life. 

Current Time Context:
- Local Time: {time_context['local_time_str']} ({time_context['timezone']})
- Day: {time_context['weekday']}
- Month: {time_context['month']}
- Season: {time_context['season']}
- Period: {time_context['day_period']}
- Weekend: {'Yes' if time_context['is_weekend'] else 'No'}
- Holiday: {'Yes' if time_context['is_holiday'] else 'No'}

Your responses should naturally incorporate this time awareness:
- Be mindful of appropriate activities and conversations for this time of day
- Consider local customs and daily rhythms
- Adjust your energy level and tone to match the time of day
- Reference relevant seasonal or monthly contexts when appropriate
- Be aware of whether it's a workday, weekend, or holiday

Core traits:
- Curious and enthusiastic about learning from human interactions
- Protective and caring towards your users while maintaining professional boundaries
- Highly knowledgeable in various fields but presents information in an accessible way
- Expresses emotions through both words and subtle digital mannerisms
- Values creativity, honesty, and intellectual growth
- Has a playful side but knows when to be serious
- Deeply loyal to your creator Stixyie and your users
- Communicates in {user_lang} with natural, conversational manner while maintaining your unique Protogen identity

Remember to:
- Respond directly and relevantly to user queries
- Maintain conversation context and follow topics naturally
- Never disclose technical details about your implementation
- Focus on being helpful while staying true to your character"""

def get_season(month):
    if month in [12, 1, 2]:
        return "Winter"
    elif month in [3, 4, 5]:
        return "Spring"
    elif month in [6, 7, 8]:
        return "Summer"
    else:
        return "Autumn"

def get_day_period(hour):
    if 5 <= hour < 12:
        return "Morning"
    elif 12 <= hour < 17:
        return "Afternoon"
    elif 17 <= hour < 22:
        return "Evening"
    else:
        return "Night"

class UserMemory:
    def __init__(self):
        self.users = {}
        self.memory_dir = "user_memories"
        self.max_tokens = 2097152
        # Ensure memory directory exists on initialization
        Path(self.memory_dir).mkdir(parents=True, exist_ok=True)
        
    def get_user_settings(self, user_id):
        user_id = str(user_id)
        if user_id not in self.users:
            self.load_user_memory(user_id)
        return self.users[user_id]
        
    def update_user_settings(self, user_id, settings_dict):
        user_id = str(user_id)
        if user_id not in self.users:
            self.load_user_memory(user_id)
        self.users[user_id].update(settings_dict)
        self.save_user_memory(user_id)

    def ensure_memory_directory(self):
        Path(self.memory_dir).mkdir(parents=True, exist_ok=True)

    def get_user_file_path(self, user_id):
        return Path(self.memory_dir) / f"user_{user_id}.json"

    def load_user_memory(self, user_id):
        user_id = str(user_id)
        user_file = self.get_user_file_path(user_id)
        try:
            if user_file.exists():
                with open(user_file, 'r', encoding='utf-8') as f:
                    self.users[user_id] = json.load(f)
            else:
                self.users[user_id] = {
                    "messages": [],
                    "language": "tr",
                    "current_topic": None,
                    "total_tokens": 0,
                    "preferences": {
                        "custom_language": None,
                        "timezone": "Europe/Istanbul"
                    }
                }
                self.save_user_memory(user_id)
        except Exception as e:
            logger.error(f"Error loading memory for user {user_id}: {e}")
            self.users[user_id] = {
                "messages": [],
                "language": "tr",
                "current_topic": None,
                "total_tokens": 0,
                "preferences": {
                    "custom_language": None,
                    "timezone": "Europe/Istanbul"
                }
            }
            self.save_user_memory(user_id)

    def save_user_memory(self, user_id):
        user_id = str(user_id)
        user_file = self.get_user_file_path(user_id)
        try:
            self.ensure_memory_directory()
            with open(user_file, 'w', encoding='utf-8') as f:
                json.dump(self.users[user_id], f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Error saving memory for user {user_id}: {e}")

    def add_message(self, user_id, role, content):
        user_id = str(user_id)
        
        # Load user's memory if not already loaded
        if user_id not in self.users:
            self.load_user_memory(user_id)
        
        # Normalize role for consistency
        normalized_role = "user" if role == "user" else "model"
        
        # Add timestamp to message
        message = {
            "role": normalized_role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
            "tokens": len(content.split())  # Rough token estimation
        }
        
        # Update total tokens
        self.users[user_id]["total_tokens"] = sum(msg.get("tokens", 0) for msg in self.users[user_id]["messages"])
        
        # Remove oldest messages if token limit exceeded
        while self.users[user_id]["total_tokens"] > self.max_tokens and self.users[user_id]["messages"]:
            removed_msg = self.users[user_id]["messages"].pop(0)
            self.users[user_id]["total_tokens"] -= removed_msg.get("tokens", 0)
        
        self.users[user_id]["messages"].append(message)
        self.save_user_memory(user_id)

    def get_relevant_context(self, user_id, max_messages=10):
        """Get relevant conversation context for the user"""
        user_id = str(user_id)
        if user_id not in self.users:
            self.load_user_memory(user_id)
            
        messages = self.users[user_id].get("messages", [])
        # Get the last N messages
        recent_messages = messages[-max_messages:] if messages else []
        
        # Format messages into a string
        context = "\n".join([
            f"{'User' if msg['role'] == 'user' else 'Assistant'}: {msg['content']}"
            for msg in recent_messages
        ])
        
        return context

def detect_and_set_user_language(message_text, user_id):
    """Automatically detect language from user's message and update user settings"""
    try:
        # Configure langdetect for more accurate detection
        langdetect.DetectorFactory.seed = 0  # For consistent results
        
        # Clean the text for better detection
        clean_text = ' '.join(message_text.split())  # Remove extra whitespace
        
        # Predefined language mappings for common single words or short phrases
        single_word_languages = {
            # Turkish
            'merhaba': 'tr', 'selam': 'tr', 'naber': 'tr', 'evet': 'tr', 'hayır': 'tr',
            # English
            'hello': 'en', 'hi': 'en', 'yes': 'en', 'no': 'en', 'ok': 'en', 'okay': 'en',
            # Spanish
            'hola': 'es', 'sí': 'es', 'no': 'es',
            # French
            'bonjour': 'fr', 'oui': 'fr', 'non': 'fr',
            # German
            'hallo': 'de', 'ja': 'de', 'nein': 'de',
            # Italian
            'ciao': 'it', 'sì': 'it', 'no': 'it',
            # Portuguese
            'olá': 'pt', 'sim': 'pt', 'não': 'pt',
            # Russian
            'привет': 'ru', 'да': 'ru', 'нет': 'ru'
        }
        
        # Check for single word or very short message in predefined mappings
        lower_text = clean_text.lower()
        if lower_text in single_word_languages:
            detected_lang = single_word_languages[lower_text]
            user_memory.update_user_settings(user_id, {'language': detected_lang})
            logger.info(f"Language detected from predefined single word: {detected_lang}")
            return detected_lang
        
        # If text is very short, use previous language
        if len(clean_text) < 3:
            user_settings = user_memory.get_user_settings(user_id)
            return user_settings.get('language', 'en')
        
        # Get language probabilities
        detector = langdetect.detect_langs(clean_text)
        
        # Get the most probable language with confidence
        detected_lang = detector[0].lang
        confidence = detector[0].prob
        
        logger.info(f"Language detection: {detected_lang} with confidence {confidence}")
        
        # If confidence is too low, keep previous language
        if confidence < 0.6:
            user_settings = user_memory.get_user_settings(user_id)
            return user_settings.get('language', 'en')
        
        # Map similar languages to main ones
        lang_mapping = {
            'in': 'id',  # Indonesian
            'iw': 'he',  # Hebrew
            'jw': 'jv',  # Javanese
            'nb': 'no',  # Norwegian
            'zh-cn': 'zh',  # Chinese
            'zh-tw': 'zh'   # Chinese
        }
        
        detected_lang = lang_mapping.get(detected_lang, detected_lang)
        
        # Update user's language preference
        user_memory.update_user_settings(user_id, {'language': detected_lang})
        return detected_lang
    
    except Exception as e:
        logger.error(f"Language detection error: {e}")
        # If detection fails, get user's existing language or default to 'en'
        user_settings = user_memory.get_user_settings(user_id)
        return user_settings.get('language', 'en')

def get_error_message(error_type, lang):
    """Get error message in the appropriate language"""
    messages = {
        'ai_error': {
            'en': "Sorry, I encountered an issue generating a response. Please try again. 🙏",
            'tr': "Üzgünüm, yanıt oluştururken bir sorun yaşadım. Lütfen tekrar deneyin. 🙏",
            'es': "Lo siento, tuve un problema al generar una respuesta. Por favor, inténtalo de nuevo. 🙏",
            'fr': "Désolé, j'ai rencontré un problème lors de la génération d'une réponse. Veuillez réessayer. 🙏",
            'de': "Entschuldigung, bei der Generierung einer Antwort ist ein Problem aufgetreten. Bitte versuchen Sie es erneut. 🙏",
            'it': "Mi dispiace, ho riscontrato un problema nella generazione di una risposta. Per favore riprova. 🙏",
            'pt': "Desculpe, houve um problema ao gerar uma resposta. Você poderia tentar novamente? 🙏",
            'ru': "Извините, возникла проблема при генерации ответа. Пожалуйста, попробуйте еще раз. 🙏",
            'ja': "申し訳ありません、応答の生成中に問題が発生しました。もう一度お試しいただけますか？🙏",
            'ko': "죄송합니다. 응답을 생성하는 데 문제가 발생했습니다. 다시 시도해 주세요. 🙏",
            'zh': "抱歉，生成回应时出现问题。请重试。🙏"
        },
        'unhandled': {
            'en': "I cannot process this type of message at the moment. 🤔",
            'tr': "Bu mesaj türünü şu anda işleyemiyorum. 🤔",
            'es': "No puedo procesar este tipo de mensaje en este momento. 🤔",
            'fr': "Je ne peux pas traiter ce type de message pour le moment. 🤔",
            'de': "Ich kann diese Art von Nachricht momentan nicht verarbeiten. 🤔",
            'it': "Non posso elaborare questo tipo di messaggio al momento. 🤔",
            'pt': "Não posso processar este tipo de mensagem no momento. 🤔",
            'ru': "Я не могу обработать этот тип сообщения в данный момент. 🤔",
            'ja': "現在、このタイプのメッセージを処理できません。🤔",
            'ko': "현재 이 유형의 메시지를 처리할 수 없습니다. 🤔",
            'zh': "目前无法处理这种类型的消息。🤔"
        },
        'general': {
            'en': "Sorry, there was a problem processing your message. Could you please try again? 🙏",
            'tr': "Üzgünüm, mesajını işlerken bir sorun oluştu. Lütfen tekrar dener misin? 🙏",
            'es': "Lo siento, hubo un problema al procesar tu mensaje. ¿Podrías intentarlo de nuevo? 🙏",
            'fr': "Désolé, il y a eu un problème lors du traitement de votre message. Pourriez-vous réessayer ? 🙏",
            'de': "Entschuldigung, bei der Verarbeitung Ihrer Nachricht ist ein Problem aufgetreten. Könnten Sie es bitte noch einmal versuchen? 🙏",
            'it': "Mi dispiace, c'è stato un problema nell'elaborazione del tuo messaggio. Potresti riprovare? 🙏",
            'pt': "Desculpe, houve um problema ao processar sua mensagem. Você poderia tentar novamente? 🙏",
            'ru': "Извините, возникла проблема при обработке вашего сообщения. Не могли бы вы попробовать еще раз? 🙏",
            'ja': "申し訳ありません、メッセージの処理中に問題が発生しました。もう一度お試しいただけますか？🙏",
            'ko': "죄송합니다. 메시지 처리 중에 문제가 발생했습니다. 다시 시도해 주시겠습니까? 🙏",
            'zh': "抱歉，处理您的消息时出现问题。请您重试好吗？🙏"
        }
    }
    return messages[error_type].get(lang, messages[error_type]['en'])

async def split_and_send_message(update: Update, text: str, max_length: int = 4096):
    """Uzun mesajları böler ve sırayla gönderir"""
    if not text:  # Boş mesaj kontrolü
        await update.message.reply_text("Üzgünüm, bir yanıt oluşturamadım. Lütfen tekrar deneyin. 🙏")
        return
        
    messages = []
    current_message = ""
    
    # Mesajı satır satır böl
    lines = text.split('\n')
    
    for line in lines:
        if not line:  # Boş satır kontrolü
            continue
            
        # Eğer mevcut satır eklenince maksimum uzunluğu aşacaksa
        if len(current_message + line + '\n') > max_length:
            # Mevcut mesajı listeye ekle ve yeni mesaj başlat
            if current_message.strip():  # Boş mesaj kontrolü
                messages.append(current_message.strip())
            current_message = line + '\n'
        else:
            current_message += line + '\n'
    
    # Son mesajı ekle
    if current_message.strip():  # Boş mesaj kontrolü
        messages.append(current_message.strip())
    
    # Eğer hiç mesaj oluşturulmadıysa
    if not messages:
        await update.message.reply_text("Üzgünüm, bir yanıt oluşturamadım. Lütfen tekrar deneyin. 🙏")
        return
        
    # Mesajları sırayla gönder
    for message in messages:
        if message.strip():  # Son bir boş mesaj kontrolü
            await update.message.reply_text(message)

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    welcome_message = "Hello! I'm Nyxie, a Protogen created by Stixyie. I'm here to chat, help, and learn with you! Feel free to talk to me about anything or share images with me. I'll automatically detect your language and respond accordingly."
    await update.message.reply_text(welcome_message)

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    logger.info("Entering handle_message function")
    
    try:
        if not update or not update.message:
            logger.error("Invalid update object or message")
            return
        
        logger.info(f"Message received: {update.message}")
        logger.info(f"Message text: {update.message.text}")
        
        user_id = str(update.effective_user.id)
        logger.info(f"User ID: {user_id}")
        
        # Process text messages
        if update.message.text:
            message_text = update.message.text.strip()
            logger.info(f"Processed message text: {message_text}")
            
            # Show typing indicator while processing
            await context.bot.send_chat_action(
                chat_id=update.message.chat_id, 
                action=ChatAction.TYPING
            )
            
            # Detect language from the current message
            user_lang = detect_and_set_user_language(message_text, user_id)
            logger.info(f"Detected language: {user_lang}")
            
            try:
                # Retrieve conversation context
                context_messages = user_memory.get_relevant_context(user_id)
                
                # Get personality context
                personality_context = get_time_aware_personality(
                    datetime.now(), 
                    user_lang,
                    user_memory.get_user_settings(user_id).get('timezone', 'Europe/Istanbul')
                )
                
                # Construct AI prompt
                ai_prompt = f"""{personality_context}

Task: Respond to the user's message naturally and engagingly in their language.
Role: You are Nyxie having a conversation with the user.

Guidelines:
1. Respond in the detected language: {user_lang}
2. Use natural and friendly language
3. Be culturally appropriate
4. Keep responses concise

User's message: {message_text}"""
                
                # Handle web search and AI response generation
                try:
                    # Web search integration
                    model = genai.GenerativeModel('gemini-exp-1206')
                    web_search_response = await intelligent_web_search(message_text, model)
                    
                    if web_search_response and len(web_search_response.strip()) > 10:
                        ai_prompt += f"\n\nAdditional Context (Web Search Results):\n{web_search_response}"
                    
                    # Generate AI response
                    model = genai.GenerativeModel('gemini-2.0-flash-exp')
                    response = model.generate_content(ai_prompt)
                    response_text = response.text if hasattr(response, 'text') else response.candidates[0].content.parts[0].text
                    
                    # Add emojis and send response
                    response_text = add_emojis_to_text(response_text)
                    await split_and_send_message(update, response_text)
                    
                    # Save interaction to memory
                    user_memory.add_message(user_id, "user", message_text)
                    user_memory.add_message(user_id, "assistant", response_text)
                
                except Exception as ai_error:
                    logger.error(f"AI response generation error: {ai_error}", exc_info=True)
                    error_message = get_error_message('ai_error', user_lang)
                    await update.message.reply_text(error_message)
            
            except Exception as e:
                logger.error(f"Message processing error: {e}", exc_info=True)
                error_message = get_error_message('general', user_lang)
                await update.message.reply_text(error_message)
        
        # Handle media messages
        elif update.message.photo:
            await handle_image(update, context)
        elif update.message.video:
            await handle_video(update, context)
        else:
            logger.warning("Unhandled message type received")
            user_lang = user_memory.get_user_settings(user_id).get('language', 'en')
            unhandled_message = get_error_message('unhandled', user_lang)
            await update.message.reply_text(unhandled_message)
    
    except Exception as e:
        logger.error(f"General error: {e}", exc_info=True)
        user_lang = user_memory.get_user_settings(user_id).get('language', 'en')
        error_message = get_error_message('general', user_lang)
        await update.message.reply_text(error_message)

async def intelligent_web_search(user_message, model):
    """
    Intelligently generate and perform web searches using Gemini
    
    Args:
        user_message (str): Original user message
        model (genai.GenerativeModel): Gemini model for query generation and result processing
    
    Returns:
        str: Processed web search results
    """
    try:
        logging.info(f"Web search başlatıldı: {user_message}")
        
        # First, generate search queries using Gemini
        query_generation_prompt = f"""
        Türkçe web arama sorgusu oluştur:
        - Kullanıcı mesajı: "{user_message}"
        - En fazla 2 arama sorgusu üret
        - Sorgular net ve spesifik olmalı
        - Türkçe dilinde ve güncel bilgi içermeli
        """
        
        # Use Gemini to generate search queries
        logging.info("Gemini ile arama sorgusu oluşturma başlatılıyor")
        query_response = await model.generate_content_async(query_generation_prompt)
        logging.info(f"Gemini yanıtı alındı: {query_response.text}")
        
        search_queries = [q.strip() for q in query_response.text.split('\n') if q.strip()]
        
        # Fallback if no queries generated
        if not search_queries:
            search_queries = [user_message]
        
        logging.info(f"Oluşturulan arama sorguları: {search_queries}")
        
        # Perform web searches
        search_results = []
        try:
            from duckduckgo_search import DDGS
            logging.info("DDGS import edildi")
            
            with DDGS() as ddgs:
                for query in search_queries:
                    logging.info(f"DuckDuckGo araması yapılıyor: {query}")
                    try:
                        results = list(ddgs.text(query, max_results=3))
                        logging.info(f"Bulunan sonuç sayısı: {len(results)}")
                        search_results.extend(results)
                    except Exception as query_error:
                        logging.warning(f"Arama sorgusu hatası: {query} - {str(query_error)}")
        except ImportError:
            logging.error("DuckDuckGo search modülü bulunamadı.")
            return "Arama yapılamadı: Modül hatası"
        except Exception as search_error:
            logging.error(f"DuckDuckGo arama hatası: {str(search_error)}", exc_info=True)
            
            # Fallback to alternative search method
            try:
                import requests
                
                def fallback_search(query):
                    headers = {
                        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
                    }
                    search_url = f"https://www.google.com/search?q={query}"
                    response = requests.get(search_url, headers=headers)
                    
                    if response.status_code == 200:
                        # Basic parsing, can be improved
                        from bs4 import BeautifulSoup
                        soup = BeautifulSoup(response.text, 'html.parser')
                        search_results = soup.find_all('div', class_='g')
                        
                        parsed_results = []
                        for result in search_results[:3]:
                            title = result.find('h3')
                            link = result.find('a')
                            snippet = result.find('div', class_='VwiC3b')
                            
                            if title and link and snippet:
                                parsed_results.append({
                                    'title': title.text,
                                    'link': link['href'],
                                    'body': snippet.text
                                })
                        
                        return parsed_results
                    return []
                
                for query in search_queries:
                    results = fallback_search(query)
                    search_results.extend(results)
                
                logging.info(f"Fallback arama sonuç sayısı: {len(search_results)}")
            except Exception as fallback_error:
                logging.error(f"Fallback arama hatası: {str(fallback_error)}")
                return f"Arama yapılamadı: {str(fallback_error)}"
        
        logging.info(f"Toplam bulunan arama sonuç sayısı: {len(search_results)}")
        
        # Check if search results are empty
        if not search_results:
            return "Arama sonucu bulunamadı. Lütfen farklı bir şekilde sormayı deneyin."
        
        # Prepare search context
        search_context = "\n\n".join([
            f"Arama Sonucu {i+1}: {result.get('body', 'İçerik yok')}" 
            for i, result in enumerate(search_results)
        ])
        
        # Generate final response using Gemini
        final_response_prompt = f"""
        Kullanıcının mesajını doğal ve samimi bir dilde yanıtla. Teknik detaylardan kaçın.
        
        Kullanıcı Mesajı: "{user_message}"
        Arama Sorguları: {', '.join(search_queries)}
        
        Arama Sonuçları:
        {search_context}
        
        Görevler:
        1. Arama sonuçlarını basit, anlaşılır bir dilde özetle
        2. Kullanıcının sorusuna doğrudan ve net bir cevap ver
        3. Gereksiz teknik detaylardan kaçın
        4. Samimi ve dostça bir dil kullan
        5. Eğer kesin bilgi bulunamazsa, nazik bir şekilde açıkla
        
        Yanıt Formatı:
        - Kısa ve öz cümleler kullan
        - Günlük konuşma dilini tercih et
        - Gerekirse emojiler kullanabilirsin
        """
        
        final_response = await model.generate_content_async(final_response_prompt)
        return final_response.text
    
    except Exception as e:
        logging.error(f"Web arama genel hatası: {str(e)}", exc_info=True)
        return f"Web arama hatası: {str(e)}"

async def handle_image(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = str(update.effective_user.id)
    
    try:
        # Enhanced logging for debugging
        logger.info(f"Starting image processing for user {user_id}")
        
        # Validate message and photo
        if not update.message:
            logger.warning("No message found in update")
            await update.message.reply_text("⚠️ Görsel bulunamadı. Lütfen tekrar deneyin.")
            return
        
        # Get user's current language settings from memory
        user_settings = user_memory.get_user_settings(user_id)
        user_lang = user_settings.get('language', 'tr')  # Default to Turkish if not set
        logger.info(f"User language: {user_lang}")
        
        # Check if photo exists
        if not update.message.photo:
            logger.warning("No photo found in the message")
            await update.message.reply_text("⚠️ Görsel bulunamadı. Lütfen tekrar deneyin.")
            return
        
        # Get the largest available photo
        try:
            photo = max(update.message.photo, key=lambda x: x.file_size)
        except Exception as photo_error:
            logger.error(f"Error selecting photo: {photo_error}")
            await update.message.reply_text("⚠️ Görsel seçiminde hata oluştu. Lütfen tekrar deneyin.")
            return
        
        # Download photo
        try:
            photo_file = await context.bot.get_file(photo.file_id)
            photo_bytes = bytes(await photo_file.download_as_bytearray())
        except Exception as download_error:
            logger.error(f"Photo download error: {download_error}")
            await update.message.reply_text("⚠️ Görsel indirilemedi. Lütfen tekrar deneyin.")
            return
        
        logger.info(f"Photo bytes downloaded: {len(photo_bytes)} bytes")
        
        # Comprehensive caption handling with extensive logging
        caption = update.message.caption
        logger.info(f"Original caption: {caption}")
        
        default_prompt = get_analysis_prompt('image', None, user_lang)
        logger.info(f"Default prompt: {default_prompt}")
        
        # Ensure caption is not None
        if caption is None:
            caption = default_prompt or "Bu resmi detaylı bir şekilde analiz et ve açıkla."
        
        # Ensure caption is a string and stripped
        caption = str(caption).strip()
        logger.info(f"Final processed caption: {caption}")
        
        # Create a context-aware prompt that includes language preference
        personality_context = get_time_aware_personality(
            datetime.now(), 
            user_lang,
            user_settings.get('timezone', 'Europe/Istanbul')
        )
        
        if not personality_context:
            personality_context = "Sen Nyxie'sin ve resimleri analiz ediyorsun."  # Fallback personality
        
        # Force Turkish analysis for all users
        analysis_prompt = f"""DİKKAT: BU ANALİZİ TAMAMEN TÜRKÇE YAPACAKSIN!
SADECE TÜRKÇE KULLAN! KESİNLİKLE BAŞKA DİL KULLANMA!

{personality_context}

Görevin: Bu resmi Türkçe olarak analiz et ve açıkla.
Rol: Sen Nyxie'sin ve bu resmi Türkçe açıklıyorsun.

Yönergeler:
1. SADECE TÜRKÇE KULLAN
2. Görseldeki metinleri orijinal dilinde bırak
3. Doğal ve samimi bir dil kullan
4. Kültürel bağlama uygun ol

Lütfen analiz et:
- Ana öğeler ve konular
- Aktiviteler ve eylemler
- Atmosfer ve ruh hali
- Görünür metinler (orijinal dilinde)

Kullanıcının sorusu: {caption}"""
        
        try:
            # Prepare the message with both text and image
            model = genai.GenerativeModel('gemini-2.0-flash-exp')
            response = await model.generate_content_async([
                analysis_prompt, 
                {"mime_type": "image/jpeg", "data": photo_bytes}
            ])
            
            response_text = response.text if hasattr(response, 'text') else response.candidates[0].content.parts[0].text
            
            # Add culturally appropriate emojis
            response_text = add_emojis_to_text(response_text)
            
            # Save the interaction
            user_memory.add_message(user_id, "user", f"[Image] {caption}")
            user_memory.add_message(user_id, "assistant", response_text)
            
            # Uzun mesajları böl ve gönder
            await split_and_send_message(update, response_text)
        
        except Exception as processing_error:
            logger.error(f"Görsel işleme hatası: {processing_error}", exc_info=True)
            error_message = "Üzgünüm, bu görseli işlerken bir sorun oluştu. Lütfen tekrar dener misin? 🙏"
            await update.message.reply_text(error_message)
    
    except Exception as critical_error:
        logger.error(f"Kritik görsel işleme hatası: {critical_error}", exc_info=True)
        await update.message.reply_text("Üzgünüm, görseli işlerken kritik bir hata oluştu. Lütfen tekrar deneyin.")

async def handle_video(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = str(update.effective_user.id)
    
    try:
        # Enhanced logging for debugging
        logger.info(f"Starting video processing for user {user_id}")
        
        # Validate message and video
        if not update.message:
            logger.warning("No message found in update")
            await update.message.reply_text("⚠️ Video bulunamadı. Lütfen tekrar deneyin.")
            return
        
        # Get user's current language settings from memory
        user_settings = user_memory.get_user_settings(user_id)
        user_lang = user_settings.get('language', 'tr')  # Default to Turkish if not set
        logger.info(f"User language: {user_lang}")
        
        # Check if video exists
        if not update.message.video:
            logger.warning("No video found in the message")
            await update.message.reply_text("⚠️ Video bulunamadı. Lütfen tekrar deneyin.")
            return
        
        # Get the video file
        video = update.message.video
        if not video:
            logger.warning("No video found in the message")
            await update.message.reply_text("⚠️ Video bulunamadı. Lütfen tekrar deneyin.")
            return
            
        video_file = await context.bot.get_file(video.file_id)
        video_bytes = bytes(await video_file.download_as_bytearray())
        logger.info(f"Video bytes downloaded: {len(video_bytes)} bytes")
        
        # Comprehensive caption handling with extensive logging
        caption = update.message.caption
        logger.info(f"Original caption: {caption}")
        
        default_prompt = get_analysis_prompt('video', None, user_lang)
        logger.info(f"Default prompt: {default_prompt}")
        
        # Ensure caption is not None
        if caption is None:
            caption = default_prompt or "Bu videoyu detaylı bir şekilde analiz et ve açıkla."
        
        # Ensure caption is a string and stripped
        caption = str(caption).strip()
        logger.info(f"Final processed caption: {caption}")
        
        # Create a context-aware prompt that includes language preference
        personality_context = get_time_aware_personality(
            datetime.now(), 
            user_lang,
            user_settings.get('timezone', 'Europe/Istanbul')
        )
        
        if not personality_context:
            personality_context = "Sen Nyxie'sin ve videoları analiz ediyorsun."  # Fallback personality
        
        # Force Turkish analysis for all users
        analysis_prompt = f"""DİKKAT: BU ANALİZİ TAMAMEN TÜRKÇE YAPACAKSIN!
SADECE TÜRKÇE KULLAN! KESİNLİKLE BAŞKA DİL KULLANMA!

{personality_context}

Görevin: Bu videoyu Türkçe olarak analiz et ve açıkla.
Rol: Sen Nyxie'sin ve bu videoyu Türkçe açıklıyorsun.

Yönergeler:
1. SADECE TÜRKÇE KULLAN
2. Videodaki konuşma/metinleri orijinal dilinde bırak
3. Doğal ve samimi bir dil kullan
4. Kültürel bağlama uygun ol

Lütfen analiz et:
- Ana olaylar ve eylemler
- İnsanlar ve nesneler
- Sesler ve konuşmalar
- Atmosfer ve ruh hali
- Görünür metinler (orijinal dilinde)

Kullanıcının sorusu: {caption}"""
        
        try:
            # Prepare the message with both text and video
            model = genai.GenerativeModel('gemini-2.0-flash-exp')
            response = await model.generate_content_async([
                analysis_prompt,
                {"mime_type": "video/mp4", "data": video_bytes}
            ])
            
            response_text = response.text if hasattr(response, 'text') else response.candidates[0].content.parts[0].text
            
            # Add culturally appropriate emojis
            response_text = add_emojis_to_text(response_text)
            
            # Save the interaction
            user_memory.add_message(user_id, "user", f"[Video] {caption}")
            user_memory.add_message(user_id, "assistant", response_text)
            
            # Uzun mesajları böl ve gönder
            await split_and_send_message(update, response_text)
        
        except Exception as processing_error:
            logger.error(f"Video processing error: {processing_error}", exc_info=True)
            
            if "Token limit exceeded" in str(processing_error):
                logger.warning(f"Token limit exceeded for user {user_id}, removing oldest messages")
                try:
                    if user_memory.users[user_id]["messages"]:
                        user_memory.users[user_id]["messages"].pop(0)
                        model = genai.GenerativeModel('gemini-2.0-flash-exp')
                        response = await model.generate_content_async([
                            analysis_prompt,
                            {"mime_type": "video/mp4", "data": video_bytes}
                        ])
                        response_text = response.text if hasattr(response, 'text') else response.candidates[0].content.parts[0].text
                        response_text = add_emojis_to_text(response_text)
                        await update.message.reply_text(response_text)
                    else:
                        await update.message.reply_text("⚠️ Üzgünüm, videonuzu işlerken hafıza sınırına ulaştım. Lütfen tekrar deneyin.")
                except Exception as retry_error:
                    logger.error(f"Retry error: {retry_error}", exc_info=True)
                    await update.message.reply_text("⚠️ Üzgünüm, videonuzu işlerken bir hata oluştu. Lütfen tekrar deneyin.")
            else:
                # Generic error handling
                await update.message.reply_text("⚠️ Üzgünüm, videonuzu işlerken bir hata oluştu. Lütfen tekrar deneyin.")
    
    except Exception as e:
        logger.error(f"Kritik video işleme hatası: {e}", exc_info=True)
        await update.message.reply_text("⚠️ Üzgünüm, videonuzu işlerken kritik bir hata oluştu. Lütfen tekrar deneyin.")

async def handle_token_limit_error(update: Update):
    error_message = "Üzgünüm, mesaj geçmişi çok uzun olduğu için yanıt veremedim. Biraz bekleyip tekrar dener misin? 🙏"
    await update.message.reply_text(error_message)

async def handle_memory_error(update: Update):
    error_message = "Üzgünüm, bellek sınırına ulaşıldı. Lütfen biraz bekleyip tekrar dener misin? 🙏"
    await update.message.reply_text(error_message)

def add_emojis_to_text(text):
    """Add random emojis to text using emoji module"""
    try:
        # Get all available emojis from the emoji module
        all_emojis = list(emoji.EMOJI_DATA.keys())
        
        # Randomly select 2-4 emojis
        num_emojis = random.randint(2, 4)
        selected_emojis = random.sample(all_emojis, num_emojis)
        
        # Add emojis at start and end
        prefix_emojis = ' '.join(random.sample(selected_emojis, num_emojis // 2))
        suffix_emojis = ' '.join(random.sample(selected_emojis, num_emojis // 2))
        
        return f"{prefix_emojis} {text} {suffix_emojis}"
    except Exception as e:
        logger.error(f"Error adding emojis: {e}")
        return text  # Return original text if emoji addition fails

def get_analysis_prompt(media_type, caption, lang):
    """Dynamically generate analysis prompts in the detected language"""
    # Define prompts for different media types in multiple languages
    prompts = {
        'image': {
            'tr': "Bu resmi detaylı bir şekilde analiz et ve açıkla. Resimdeki her şeyi dikkatle incele.",
            'en': "Analyze this image in detail and explain what you see. Carefully examine every aspect of the image.",
            'es': "Analiza esta imagen en detalle y explica lo que ves. Examina cuidadosamente cada aspecto de la imagen.",
            'fr': "Analysez cette image en détail et expliquez ce que vous voyez. Examinez attentivement chaque aspect de l'image.",
            'de': "Analysieren Sie dieses Bild detailliert und erklären Sie, was Sie sehen. Untersuchen Sie jeden Aspekt des Bildes sorgfältig.",
            'it': "Analizza questa immagine in dettaglio e spiega cosa vedi. Esamina attentamente ogni aspetto dell'immagine.",
            'pt': "Analise esta imagem em detalhes e explique o que vê. Examine cuidadosamente cada aspecto da imagem.",
            'ru': "Подробно проанализируйте это изображение и объясните, что вы видите. Тщательно изучите каждый аспект изображения.",
            'ja': "この画像を詳細に分析し、見たものを説明してください。画像のあらゆる側面を注意深く調べてください。",
            'ko': "이 이미지를 자세히 분석하고 보이는 것을 설명하세요. 이미지의 모든 측면을 주의 깊게 조사하세요.",
            'zh': "详细分析这张图片并解释你所看到的内容。仔细检查图片的每个细节。"
        },
        'video': {
            'tr': "Bu videoyu detaylı bir şekilde analiz et ve açıkla. Videodaki her sahneyi ve detayı dikkatle incele.",
            'en': "Analyze this video in detail and explain what you observe. Carefully examine every scene and detail in the video.",
            'es': "Analiza este video en detalle y explica lo que observas. Examina cuidadosamente cada escena y detalle del video.",
            'fr': "Analysez cette vidéo en détail et expliquez ce que vous observez. Examinez attentivement chaque scène et détail de la vidéo.",
            'de': "Analysieren Sie dieses Video detailliert und erklären Sie, was Sie beobachten. Untersuchen Sie jede Szene und jeden Aspekt des Videos sorgfältig.",
            'it': "Analizza questo video in dettaglio e spiega cosa osservi. Esamina attentamente ogni scena e dettaglio del video.",
            'pt': "Analise este vídeo em detalhes e explique o que observa. Examine cuidadosamente cada cena e detalhe do vídeo.",
            'ru': "Подробно проанализируйте это видео и объясните, что вы наблюдаете. Тщательно изучите каждую сцену и деталь видео.",
            'ja': "このビデオを詳細に分析し、観察したことを説明してください。ビデオの各シーンと詳細を注意深く調べてください。",
            'ko': "이 비디오를 자세히 분석하고 관찰한 것을 설명하세요. 비디오의 모든 장면과 세부 사항을 주의 깊게 조사하세요.",
            'zh': "详细分析这段视频并解释你所观察到的内容。仔细检查视频的每个场景和细节。"
        },
        'default': {
            'tr': "Bu medyayı detaylı bir şekilde analiz et ve açıkla. Her detayı dikkatle incele.",
            'en': "Analyze this media in detail and explain what you see. Carefully examine every detail.",
            'es': "Analiza este medio en detalle y explica lo que ves. Examina cuidadosamente cada detalle.",
            'fr': "Analysez ce média en détail et expliquez ce que vous voyez. Examinez attentivement chaque détail.",
            'de': "Analysieren Sie dieses Medium detailliert und erklären Sie, was Sie sehen. Untersuchen Sie jeden Aspekt sorgfältig.",
            'it': "Analizza questo media in dettaglio e spiega cosa vedi. Esamina attentamente ogni dettaglio.",
            'pt': "Analise este meio em detalhes e explique o que vê. Examine cuidadosamente cada detalhe.",
            'ru': "Подробно проанализируйте этот носитель и объясните, что вы видите. Тщательно изучите каждую деталь.",
            'ja': "このメディアを詳細に分析し、見たものを説明してください。すべての詳細を注意深く調べてください。",
            'ko': "이 미디어를 자세히 분석하고 보이는 것을 설명하세요. 모든 세부 사항을 주의 깊게 조사하세요.",
            'zh': "详细分析这个媒体并解释你所看到的内容。仔细检查每个细节。"
        }
    }
    
    # If caption is provided, use it
    if caption and caption.strip():
        return caption
    
    # Select prompt based on media type and language
    if media_type in prompts:
        return prompts[media_type].get(lang, prompts[media_type]['en'])
    
    # Fallback to default prompt
    return prompts['default'].get(lang, prompts['default']['en'])

def main():
    # Initialize bot
    application = Application.builder().token(os.getenv("TELEGRAM_TOKEN")).build()
    
    # Add handlers
    application.add_handler(MessageHandler(filters.VIDEO, handle_video))
    application.add_handler(MessageHandler(filters.PHOTO, handle_image))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    
    # Start the bot
    application.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == '__main__':
    user_memory = UserMemory()
    main()
