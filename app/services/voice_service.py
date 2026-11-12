"""
Voice Service for Tamil Language Support
Provides speech recognition and text-to-speech capabilities for Tamil language.
Integrates with the chatbot for voice-based interactions.
"""

import logging
import tempfile
import os
from typing import Optional, Tuple
from threading import Thread

logger = logging.getLogger(__name__)


class TamilVoiceService:
    """
    Service for Tamil voice recognition and synthesis.
    Supports both English and Tamil languages for voice interactions.
    """
    
    def __init__(self):
        """Initialize the voice service"""
        self.is_listening = False
        self.speech_recognizer = None
        self.tts_engine = None
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize speech recognition and TTS components"""
        try:
            import speech_recognition as sr
            self.speech_recognizer = sr.Recognizer()
            logger.info("Speech recognition initialized successfully")
        except ImportError:
            logger.warning("Speech recognition library not installed. Install with: pip install SpeechRecognition")
        except Exception as e:
            logger.warning(f"Failed to initialize speech recognition: {e}")
        
        try:
            import pyttsx3
            self.tts_engine = pyttsx3.init()
            self._configure_tts_engine()
            logger.info("Text-to-speech engine initialized successfully")
        except ImportError:
            logger.warning("pyttsx3 library not installed. Install with: pip install pyttsx3")
        except Exception as e:
            logger.warning(f"Failed to initialize TTS engine: {e}")
    
    def _configure_tts_engine(self):
        """Configure TTS engine for better Tamil pronunciation"""
        if self.tts_engine:
            try:
                # Set properties for better voice quality
                self.tts_engine.setProperty('rate', 150)  # Speed of speech
                self.tts_engine.setProperty('volume', 0.9)  # Volume level
                
                # Try to set a voice that might support Tamil better
                voices = self.tts_engine.getProperty('voices')
                if voices:
                    # Prefer female voice for healthcare context
                    for voice in voices:
                        if 'female' in voice.name.lower() or 'zira' in voice.name.lower():
                            self.tts_engine.setProperty('voice', voice.id)
                            break
            except Exception as e:
                logger.warning(f"Failed to configure TTS engine: {e}")
    
    def listen_for_speech(self, language: str = 'ta-IN', timeout: int = 5) -> Optional[str]:
        """
        Listen for speech input and convert to text.
        
        Args:
            language: Language code (default: 'ta-IN' for Tamil)
            timeout: Maximum time to listen in seconds
            
        Returns:
            Recognized text or None if recognition failed
        """
        if not self.speech_recognizer:
            logger.error("Speech recognition not available")
            return None
        
        try:
            import speech_recognition as sr
            
            with sr.Microphone() as source:
                logger.info(f"Listening for speech in {language}...")
                self.speech_recognizer.adjust_for_ambient_noise(source, duration=1)
                
                try:
                    audio = self.speech_recognizer.listen(source, timeout=timeout)
                    logger.info("Processing speech...")
                    
                    # Try Google Speech Recognition first (better Tamil support)
                    try:
                        text = self.speech_recognizer.recognize_google(audio, language=language)
                        logger.info(f"Recognized text: {text}")
                        return text
                    except Exception as google_error:
                        logger.warning(f"Google recognition failed: {google_error}")
                        # Fallback to Sphinx (offline)
                        try:
                            text = self.speech_recognizer.recognize_sphinx(audio)
                            logger.info(f"Recognized text (Sphinx): {text}")
                            return text
                        except Exception as sphinx_error:
                            logger.warning(f"Sphinx recognition failed: {sphinx_error}")
                            return None
                        
                except sr.WaitTimeoutError:
                    logger.warning("No speech detected within timeout")
                    return None
                    
        except Exception as e:
            logger.error(f"Speech recognition error: {e}")
            return None
    
    def speak_text(self, text: str, language: str = 'en') -> bool:
        """
        Convert text to speech and play it.
        
        Args:
            text: Text to speak
            language: Language of the text (default: 'en' for English)
            
        Returns:
            True if successful, False otherwise
        """
        if not self.tts_engine:
            logger.error("TTS engine not available")
            return False
        
        try:
            logger.info(f"Speaking text: {text[:50]}...")
            
            # For Tamil text, we might need special handling
            if language == 'ta':
                # Try to use a Tamil-specific TTS approach
                # For now, we'll use the default engine which may have limited Tamil support
                pass
            
            # Speak the text in a separate thread to avoid blocking
            def speak_thread():
                try:
                    self.tts_engine.say(text)
                    self.tts_engine.runAndWait()
                except Exception as e:
                    logger.error(f"TTS playback error: {e}")
            
            thread = Thread(target=speak_thread)
            thread.daemon = True
            thread.start()
            
            return True
            
        except Exception as e:
            logger.error(f"TTS error: {e}")
            return False
    
    def translate_to_tamil(self, english_text: str) -> str:
        """
        Translate English text to Tamil.
        This is a simplified translation for demonstration.
        In production, use a proper translation API.
        
        Args:
            english_text: English text to translate
            
        Returns:
            Tamil text (simplified mapping)
        """
        # Basic medical terminology mapping for demonstration
        translations = {
            # Common medical terms
            'cancer': 'புற்றுநோய்',
            'tumor': 'கட்டி',
            'diagnosis': 'நோயறிதல்',
            'treatment': 'சிகிச்சை',
            'doctor': 'மருத்துவர்',
            'hospital': 'மருத்துவமனை',
            'medicine': 'மருந்து',
            'symptoms': 'அறிகுறிகள்',
            'pain': 'வலி',
            'fever': 'காய்ச்சல்',
            'headache': 'தலைவலி',
            'risk': 'ஆபத்து',
            'high risk': 'அதிக ஆபத்து',
            'low risk': 'குறைந்த ஆபத்து',
            'test': 'பரிசோதனை',
            'scan': 'ஸ்கேன்',
            'report': 'அறிக்கை',
            'appointment': 'நியமனம்',
            'emergency': 'அவசரம்',
            'help': 'உதவி',
            'thank you': 'நன்றி',
            'hello': 'வணக்கம்',
            'goodbye': 'வணக்கம்',
            'please': 'தயவு செய்து',
            'sorry': 'மன்னிக்கவும்',
            
            # AI-specific terms
            'artificial intelligence': 'செயற்கை நுண்ணறிவு',
            'prediction': 'கணிப்பு',
            'analysis': 'பகுப்பாய்வு',
            'confidence': 'நம்பிக்கை',
            'result': 'முடிவு',
        }
        
        tamil_text = english_text.lower()
        for eng, tam in translations.items():
            tamil_text = tamil_text.replace(eng, tam)
        
        return tamil_text
    
    def get_supported_languages(self) -> list:
        """Get list of supported languages for voice recognition"""
        return [
            {'code': 'ta-IN', 'name': 'Tamil (India)', 'native': 'தமிழ்'},
            {'code': 'en-US', 'name': 'English (US)', 'native': 'English'},
            {'code': 'en-GB', 'name': 'English (UK)', 'native': 'English'},
        ]
    
    def is_available(self) -> Tuple[bool, bool]:
        """
        Check availability of voice components.
        
        Returns:
            Tuple of (speech_recognition_available, tts_available)
        """
        return (self.speech_recognizer is not None, self.tts_engine is not None)


# Singleton instance
_voice_service = None

def get_voice_service() -> TamilVoiceService:
    """Get or create the voice service singleton"""
    global _voice_service
    if _voice_service is None:
        _voice_service = TamilVoiceService()
    return _voice_service