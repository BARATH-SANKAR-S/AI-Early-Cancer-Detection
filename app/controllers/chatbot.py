"""
Chatbot Controller
API endpoints for AI cancer information chatbot with Tamil voice support
"""

from flask import Blueprint, request, jsonify, render_template
from flask_login import current_user
from app import db
from app.models import ChatMessage
from app.services.chatbot_service import ChatbotService
from app.services.voice_service import get_voice_service
from datetime import datetime
import logging
import uuid

logger = logging.getLogger(__name__)

chatbot_bp = Blueprint('chatbot', __name__)

# Lazy initialization helpers to avoid heavy imports/downloads at server startup
_chatbot_service = None
_voice_service = None

def get_active_chatbot_service():
    global _chatbot_service
    if _chatbot_service is None:
        _chatbot_service = ChatbotService()
    return _chatbot_service

def get_active_voice_service():
    global _voice_service
    if _voice_service is None:
        _voice_service = get_voice_service()
    return _voice_service


@chatbot_bp.route('/api/chatbot/query', methods=['POST'])
def chat_query():
    """
    Process user query and return chatbot response.
    Accessible to both authenticated users and guest visitors.
    """
    try:
        data = request.get_json(silent=True) or {}
        
        if 'query' not in data:
            return jsonify({'error': 'Query is required'}), 400
        
        query = data['query'].strip()
        session_id = (data.get('session_id') or '').strip() or str(uuid.uuid4())
        
        if not query:
            return jsonify({'error': 'Query cannot be empty'}), 400
        
        # User ID if authenticated, otherwise None
        uid = current_user.id if getattr(current_user, 'is_authenticated', False) else None
        
        # Process query
        service = get_active_chatbot_service()
        result = service.process_query(query, uid)
        
        # Store chat message in database safely
        try:
            chat_message = ChatMessage(
                user_id=uid,
                session_id=session_id,
                query=query,
                response=result.get('response', ''),
                sources=result.get('sources', []),
                citations=result.get('citations', ''),
                is_cancer_related=result.get('is_cancer_related', True),
                has_sources=result.get('has_sources', False)
            )
            db.session.add(chat_message)
            db.session.commit()
        except Exception as db_err:
            logger.warning(f"Note: Could not save chat message to database: {db_err}")
            db.session.rollback()
        
        # Add session ID to response
        result['session_id'] = session_id
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error processing chat query: {e}", exc_info=True)
        db.session.rollback()
        return jsonify({'error': 'Failed to process query'}), 500


@chatbot_bp.route('/api/chatbot/history', methods=['GET'])
def chat_history():
    """
    Get chat history for current user or current session.
    """
    try:
        limit = request.args.get('limit', 20, type=int)
        session_id = (request.args.get('session_id') or '').strip()
        
        stmt = db.session.query(ChatMessage)
        
        if getattr(current_user, 'is_authenticated', False):
            stmt = stmt.filter(ChatMessage.user_id == current_user.id)
            if session_id:
                stmt = stmt.filter(ChatMessage.session_id == session_id)
        elif session_id:
            stmt = stmt.filter(ChatMessage.session_id == session_id)
        else:
            return jsonify({'history': []})
        
        messages = stmt.order_by(ChatMessage.created_at.desc()).limit(limit).all()
        
        history = []
        for msg in messages:
            history.append({
                'id': msg.id,
                'session_id': msg.session_id,
                'query': msg.query,
                'response': msg.response,
                'sources': msg.sources or [],
                'citations': msg.citations,
                'is_cancer_related': msg.is_cancer_related,
                'has_sources': msg.has_sources,
                'created_at': msg.created_at.isoformat() + 'Z' if msg.created_at else None
            })
        
        return jsonify({'history': history})
        
    except Exception as e:
        logger.error(f"Error getting chat history: {e}")
        return jsonify({'error': 'Failed to retrieve chat history'}), 500


@chatbot_bp.route('/api/chatbot/history/clear', methods=['DELETE'])
def clear_chat_history():
    """
    Clear chat history for current user or session.
    """
    try:
        session_id = (request.args.get('session_id') or '').strip()
        stmt = db.session.query(ChatMessage)
        
        if getattr(current_user, 'is_authenticated', False):
            stmt = stmt.filter(ChatMessage.user_id == current_user.id)
            if session_id:
                stmt = stmt.filter(ChatMessage.session_id == session_id)
        elif session_id:
            stmt = stmt.filter(ChatMessage.session_id == session_id)
        else:
            return jsonify({'message': 'No chat history to clear'})
        
        deleted_count = stmt.delete(synchronize_session=False)
        db.session.commit()
        
        return jsonify({
            'message': f'Chat history cleared successfully ({deleted_count} messages deleted)'
        })
        
    except Exception as e:
        logger.error(f"Error clearing chat history: {e}")
        db.session.rollback()
        return jsonify({'error': 'Failed to clear chat history'}), 500


@chatbot_bp.route('/api/chatbot/health', methods=['GET'])
def chatbot_health():
    """
    Get chatbot health status
    """
    try:
        service = get_active_chatbot_service()
        health_status = service.get_health_status()
        return jsonify(health_status)
        
    except Exception as e:
        logger.error(f"Error getting chatbot health: {e}")
        return jsonify({
            'status': 'unhealthy',
            'error': str(e),
            'timestamp': datetime.utcnow().isoformat() + 'Z'
        }), 500


@chatbot_bp.route('/api/chatbot/sessions', methods=['GET'])
def chat_sessions():
    """
    Get list of chat sessions for current authenticated user
    """
    try:
        if not getattr(current_user, 'is_authenticated', False):
            return jsonify({'sessions': []})
            
        from sqlalchemy import func
        
        sessions = db.session.query(
            ChatMessage.session_id,
            func.count(ChatMessage.id).label('message_count'),
            func.max(ChatMessage.created_at).label('last_activity')
        ).filter(
            ChatMessage.user_id == current_user.id
        ).group_by(
            ChatMessage.session_id
        ).order_by(
            func.max(ChatMessage.created_at).desc()
        ).all()
        
        session_list = []
        for session in sessions:
            session_list.append({
                'session_id': session.session_id,
                'message_count': session.message_count,
                'last_activity': session.last_activity.isoformat() + 'Z' if session.last_activity else None
            })
        
        return jsonify({'sessions': session_list})
        
    except Exception as e:
        logger.error(f"Error getting chat sessions: {e}")
        return jsonify({'error': 'Failed to retrieve chat sessions'}), 500


@chatbot_bp.route('/chatbot')
def chatbot_page():
    """Chatbot interface page - accessible to all users"""
    return render_template('chatbot.html')


@chatbot_bp.route('/api/chatbot/voice/status', methods=['GET'])
def voice_status():
    """
    Get voice service availability status.
    """
    try:
        vservice = get_active_voice_service()
        speech_available, tts_available = vservice.is_available()
        supported_languages = vservice.get_supported_languages()
        
        return jsonify({
            'speech_recognition_available': speech_available,
            'tts_available': tts_available,
            'supported_languages': supported_languages,
            'voice_service_enabled': speech_available or tts_available
        })
        
    except Exception as e:
        logger.error(f"Error getting voice status: {e}")
        return jsonify({
            'speech_recognition_available': False,
            'tts_available': False,
            'supported_languages': [],
            'voice_service_enabled': False,
            'error': str(e)
        }), 500


@chatbot_bp.route('/api/chatbot/voice/listen', methods=['POST'])
def voice_listen():
    """
    Listen for voice input and convert to text.
    """
    try:
        data = request.get_json(silent=True) or {}
        language = data.get('language', 'ta-IN')  # Default to Tamil
        timeout = data.get('timeout', 5)
        
        vservice = get_active_voice_service()
        recognized_text = vservice.listen_for_speech(language, timeout)
        
        if recognized_text:
            return jsonify({
                'success': True,
                'text': recognized_text,
                'language': language
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Speech recognition failed or no speech detected'
            }), 400
            
    except Exception as e:
        logger.error(f"Error in voice listen: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@chatbot_bp.route('/api/chatbot/voice/speak', methods=['POST'])
def voice_speak():
    """
    Convert text to speech and play it.
    """
    try:
        data = request.get_json(silent=True) or {}
        text = data.get('text', '')
        language = data.get('language', 'en')
        
        if not text:
            return jsonify({'error': 'Text is required'}), 400
        
        vservice = get_active_voice_service()
        # Translate to Tamil if requested
        if language == 'ta':
            text = vservice.translate_to_tamil(text)
        
        success = vservice.speak_text(text, language)
        
        if success:
            return jsonify({
                'success': True,
                'message': 'Text converted to speech successfully'
            })
        else:
            return jsonify({
                'success': False,
                'error': 'TTS conversion failed'
            }), 500
            
    except Exception as e:
        logger.error(f"Error in voice speak: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@chatbot_bp.route('/api/chatbot/voice/translate', methods=['POST'])
def voice_translate():
    """
    Translate text to Tamil for voice output.
    """
    try:
        data = request.get_json(silent=True) or {}
        english_text = data.get('text', '')
        
        if not english_text:
            return jsonify({'error': 'Text is required'}), 400
        
        vservice = get_active_voice_service()
        tamil_text = vservice.translate_to_tamil(english_text)
        
        return jsonify({
            'success': True,
            'original': english_text,
            'translated': tamil_text,
            'language': 'ta'
        })
        
    except Exception as e:
        logger.error(f"Error in voice translate: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500