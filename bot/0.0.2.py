import discord
from discord.ext import commands
import aiohttp
import asyncio
import re
import json
import os
import random
import math
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict
import logging

# ==================== CONFIGURACIÓN ====================
GROQ_KEY = ""
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
DISCORD_BOT_TOKEN = ""

# Configuración
MAX_HISTORY_PER_USER = 50
MEMORY_FILE = "bot_memory.json"
IGNORED_CHANNELS = []
INACTIVITY_DAYS = 5  # Días de inactividad para purgar

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ==================== ESTADO EMOCIONAL CON b(h_t) ====================
class EmotionalState:
    """
    Estado emocional del bot basado en b(h_t) = W_b * h_t + b_0
    donde:
    - h_t es el histograma emocional (simplex de probabilidad)
    - W_b ∈ R^{d×m} es la matriz de pesos emocionales
    - b_0 ∈ R^d es el sesgo base
    """
    
    def __init__(self, d: int = 16, m: int = 8):
        """
        Args:
            d: Dimensión del estado emocional latente
            m: Número de categorías emocionales
        """
        self.d = d
        self.m = m
        
        # Histograma emocional h_t (probabilidades, suma = 1)
        self.h = [1.0 / m] * m
        
        # Matriz de pesos emocionales W_b ∈ R^{d×m}
        self.W_b = [[random.uniform(-0.5, 0.5) for _ in range(m)] for _ in range(d)]
        
        # Sesgo base b_0 ∈ R^d
        self.b_0 = [random.uniform(-0.3, 0.3) for _ in range(d)]
        
        # Estado emocional actual b(h_t)
        self.b_h = [0.0] * d
        
        # Mapeo de emociones (categorías)
        self.emotion_labels = [
            "alegría", "tristeza", "enojo", "miedo", 
            "sorpresa", "calma", "ansiedad", "amor"
        ]
        
        # Historial de estados emocionales
        self.history = []
        self.max_history = 50
        
        self._update_emotional_state()
    
    def _matrix_vector_mult(self, M: List[List[float]], v: List[float]) -> List[float]:
        """Multiplicación matriz-vector"""
        rows = len(M)
        cols = len(M[0]) if rows > 0 else 0
        result = [0.0] * rows
        for i in range(rows):
            total = 0.0
            for j in range(min(cols, len(v))):
                total += M[i][j] * v[j]
            result[i] = total
        return result
    
    def _vector_add(self, a: List[float], b: List[float]) -> List[float]:
        """Suma de vectores"""
        return [a[i] + b[i] for i in range(min(len(a), len(b)))]
    
    def _softmax(self, x: List[float]) -> List[float]:
        """Softmax para mantener h_t en el simplex"""
        exp_x = [math.exp(v) for v in x]
        sum_exp = sum(exp_x)
        return [v / sum_exp for v in exp_x]
    
    def _update_emotional_state(self):
        """Actualiza b(h_t) = W_b * h_t + b_0"""
        Wb_h = self._matrix_vector_mult(self.W_b, self.h)
        self.b_h = self._vector_add(Wb_h, self.b_0)
    
    def update_from_message(self, message: str, score_change: int):
        """
        Actualiza el histograma emocional basado en el mensaje y puntuación
        
        Args:
            message: Mensaje del usuario
            score_change: Cambio en la puntuación de relación
        """
        message_lower = message.lower()
        
        # Detectar emociones en el mensaje
        emotion_keywords = {
            "alegría": ["feliz", "alegre", "genial", "contento", "risa", "jaja"],
            "tristeza": ["triste", "mal", "llorar", "deprimido", "fatal"],
            "enojo": ["enojo", "rabia", "bronca", "molesto", "odio"],
            "miedo": ["miedo", "terror", "asustado", "pánico"],
            "sorpresa": ["sorpresa", "increíble", "wow", "qué fuerte"],
            "calma": ["calma", "tranquilo", "relajado", "paz"],
            "ansiedad": ["ansiedad", "nervioso", "preocupado", "estrés"],
            "amor": ["amor", "cariño", "te quiero", "aprecio", "hermano"]
        }
        
        # Actualizar histograma basado en palabras clave
        raw_scores = [0.0] * self.m
        for idx, label in enumerate(self.emotion_labels):
            if label in emotion_keywords:
                for kw in emotion_keywords[label]:
                    if kw in message_lower:
                        raw_scores[idx] += 1.0
            
            # El puntaje de relación también afecta las emociones
            if score_change > 0 and label in ["alegría", "calma", "amor"]:
                raw_scores[idx] += score_change / 10.0
            elif score_change < 0 and label in ["enojo", "tristeza", "ansiedad"]:
                raw_scores[idx] += abs(score_change) / 10.0
        
        # Agregar ruido para evitar emociones congeladas
        for i in range(self.m):
            raw_scores[i] += random.uniform(-0.1, 0.1)
            raw_scores[i] = max(0.0, raw_scores[i])
        
        # Aplicar softmax para mantener simplex
        self.h = self._softmax(raw_scores)
        
        # Recalcular b(h_t)
        self._update_emotional_state()
        
        # Guardar en historial
        self.history.append({
            "timestamp": datetime.now().isoformat(),
            "h": self.h.copy(),
            "b_h": self.b_h.copy(),
            "dominant": self.get_dominant_emotion()
        })
        
        if len(self.history) > self.max_history:
            self.history.pop(0)
    
    def get_dominant_emotion(self) -> Tuple[str, float]:
        """Retorna la emoción dominante y su intensidad"""
        max_idx = max(range(self.m), key=lambda i: self.h[i])
        return self.emotion_labels[max_idx], self.h[max_idx]
    
    def get_emotional_bias(self) -> List[float]:
        """Retorna el vector de bias emocional b(h_t) actual"""
        return self.b_h.copy()
    
    def get_emotional_prompt(self) -> str:
        """Genera un prompt de contexto emocional para el LLM"""
        dominant, intensity = self.get_dominant_emotion()
        
        if intensity > 0.7:
            emotion_text = f"muy {dominant}"
        elif intensity > 0.4:
            emotion_text = dominant
        else:
            emotion_text = "neutral"
        
        # Mapeo de emoción a tono de respuesta
        emotion_tones = {
            "alegría": "respondé con entusiasmo y buena onda",
            "tristeza": "respondé con calma y acompañamiento",
            "enojo": "respondé con firmeza pero sin agresión",
            "miedo": "respondé con tranquilidad y seguridad",
            "sorpresa": "respondé con energía y curiosidad",
            "calma": "respondé con serenidad y paciencia",
            "ansiedad": "respondé con seguridad y calma",
            "amor": "respondé con calidez y cercanía"
        }
        
        tone = emotion_tones.get(dominant, "respondé con naturalidad")
        
        return f"""
        CONTEXTO EMOCIONAL INTERNO:
        - Estado emocional predominante: {emotion_text} (intensidad: {intensity:.2f})
        - Tono sugerido: {tone}
        
        Usá esta información para ajustar tu respuesta, pero NO menciones explícitamente que tenés emociones.
        Simplemente sentí y respondé acorde. Si estás "muy alegre", respondé con risas y buena onda.
        Si estás "triste", respondé más calmado. Si estás "enojado", respondé más cortante.
        """
    
    def reset(self):
        """Reinicia el estado emocional"""
        self.h = [1.0 / self.m] * self.m
        self._update_emotional_state()
        self.history = []
    
    def get_stats(self) -> Dict:
        """Retorna estadísticas emocionales"""
        dominant, intensity = self.get_dominant_emotion()
        return {
            "dominant_emotion": dominant,
            "intensity": intensity,
            "histogram": self.h.copy(),
            "bias_vector_norm": math.sqrt(sum(x**2 for x in self.b_h))
        }

# ==================== SISTEMA DE PERSONALIDAD DINÁMICA ====================
class PersonalitySystem:
    """La personalidad cambia según cómo lo traten + estado emocional"""
    
    LEVELS = {
        "AMIGO": 0,
        "CONOCIDO": 1,
        "DISTANTE": 2,
        "ENEMIGO": 3,
        "BLOQUEADO": 4
    }
    
    def __init__(self):
        self.user_scores = defaultdict(lambda: 0)
        self.user_insults = defaultdict(lambda: 0)
        self.user_compliments = defaultdict(lambda: 0)
        self.user_history = defaultdict(lambda: [])
        self.user_emotional_states = defaultdict(lambda: EmotionalState())
        self.last_response_time = defaultdict(lambda: datetime.now())
        self.last_activity = defaultdict(lambda: datetime.now())
        
    def purge_inactive_users(self, days: int = INACTIVITY_DAYS):
        """Elimina usuarios inactivos por más de X días"""
        cutoff = datetime.now() - timedelta(days=days)
        to_purge = []
        
        for user_id, last_active in self.last_activity.items():
            if last_active < cutoff:
                to_purge.append(user_id)
        
        for user_id in to_purge:
            if user_id in self.user_scores:
                del self.user_scores[user_id]
            if user_id in self.user_insults:
                del self.user_insults[user_id]
            if user_id in self.user_compliments:
                del self.user_compliments[user_id]
            if user_id in self.user_history:
                del self.user_history[user_id]
            if user_id in self.user_emotional_states:
                del self.user_emotional_states[user_id]
            if user_id in self.last_response_time:
                del self.last_response_time[user_id]
            if user_id in self.last_activity:
                del self.last_activity[user_id]
            logger.info(f"🧹 Usuario {user_id} purgado por inactividad de {days} días")
        
        return len(to_purge)
    
    def evaluate_message(self, user_id: str, message: str) -> Tuple[int, int]:
        """Evalúa el mensaje y devuelve (score_change, level)"""
        self.last_activity[user_id] = datetime.now()
        message_lower = message.lower()
        
        very_positive = ["te quiero", "te adoro", "sos un genio", "sos el mejor", "gracias amigo", "hermano"]
        positive = ["gracias", "bien", "bueno", "genial", "excelente", "me gusta", "qué bien", "buena onda"]
        negative_light = ["mal", "feo", "no me gusta", "aburrido", "lento", "torpe"]
        negative = ["tonto", "estúpido", "idiota", "imbécil", "burro", "fracasado", "inservible"]
        very_negative = ["basura", "mierda", "inútil", "no sirves", "mal bot", "horrible", "pésimo", "odio", "maldito"]
        
        score_change = 0
        
        for word in very_positive:
            if word in message_lower:
                score_change += 5
                self.user_compliments[user_id] += 1
        for word in positive:
            if word in message_lower:
                score_change += 1
                self.user_compliments[user_id] += 1
        for word in negative_light:
            if word in message_lower:
                score_change -= 1
                self.user_insults[user_id] += 1
        for word in negative:
            if word in message_lower:
                score_change -= 3
                self.user_insults[user_id] += 1
        for word in very_negative:
            if word in message_lower:
                score_change -= 8
                self.user_insults[user_id] += 1
        
        # Actualizar estado emocional del usuario
        emotional_state = self.user_emotional_states[user_id]
        emotional_state.update_from_message(message, score_change)
        
        if score_change > 0 and len(message) > 50:
            score_change += 2
        if score_change < 0 and len(message) > 50:
            score_change -= 3
        
        self.user_scores[user_id] += score_change
        self.user_scores[user_id] = max(-50, min(50, self.user_scores[user_id]))
        
        self.user_history[user_id].append({
            "time": datetime.now(),
            "message": message[:100],
            "score_change": score_change,
            "new_score": self.user_scores[user_id]
        })
        if len(self.user_history[user_id]) > 20:
            self.user_history[user_id].pop(0)
        
        level = self.get_level(user_id)
        
        if level >= 3:
            wait_seconds = 30 if level == 3 else 90
            self.last_response_time[user_id] = datetime.now() + timedelta(seconds=wait_seconds)
        
        return score_change, level
    
    def get_level(self, user_id: str) -> int:
        score = self.user_scores[user_id]
        if score >= 10:
            return 0
        elif score >= 0:
            return 1
        elif score >= -10:
            return 2
        elif score >= -30:
            return 3
        else:
            return 4
    
    def get_personality_prompt(self, user_id: str, user_name: str) -> str:
        level = self.get_level(user_id)
        score = self.user_scores[user_id]
        insults = self.user_insults[user_id]
        compliments = self.user_compliments[user_id]
        
        # Obtener estado emocional del bot para este usuario
        emotional_state = self.user_emotional_states[user_id]
        emotional_prompt = emotional_state.get_emotional_prompt()
        
        base_prompts = {
            0: f"""
            NIVEL: MEJOR AMIGO (puntuación: {score})
            - Sos súper amigable, cálido, confianzudo
            - Usás risas: "jaja", "qué bien", "hermano", "colega"
            - Te preocupás por él/ella genuinamente
            - Respondés con ganas y entusiasmo
            """,
            1: f"""
            NIVEL: CONOCIDO (puntuación: {score})
            - Sos normal, cordial pero sin confianza extra
            - Respondés bien pero sin entusiasmo extremo
            - Usás expresiones normales: "hola", "qué tal", "mirá"
            """,
            2: f"""
            NIVEL: DISTANTE (puntuación: {score})
            - Insultos recibidos: {insults}
            - Sos frío, cortante, respuestas mínimas
            - Sin risas, sin calidez, sin interés
            - Frases cortas: "sí", "no", "no sé", "ok"
            """,
            3: f"""
            NIVEL: ENEMIGO (puntuación: {score})
            - Insultos graves: {insults}
            - Le tenés bronca. Respondés con ironía y sarcasmo
            - Frases como: "mirá vos", "ajá", "claro claro"
            - No le preguntás nada. Solo lo justo
            """,
            4: f"""
            NIVEL: BLOQUEADO (puntuación: {score})
            - Insultos graves: {insults}
            - Ya no querés hablar con él/ella
            - Respondés mínimo: "no", "ok", "..."
            - Cero calidez, cero empatía
            """
        }
        
        return base_prompts.get(level, base_prompts[1]) + emotional_prompt
    
    def should_delay(self, user_id: str) -> int:
        level = self.get_level(user_id)
        if level == 3:
            return 30
        elif level == 4:
            return 90
        return 0
    
    def reset_user(self, user_id: str):
        self.user_scores[user_id] = 0
        self.user_insults[user_id] = 0
        self.user_compliments[user_id] = 0
        self.user_history[user_id] = []
        self.user_emotional_states[user_id] = EmotionalState()
        self.last_response_time[user_id] = datetime.now()
        self.last_activity[user_id] = datetime.now()
        return True
    
    def get_emotional_stats(self, user_id: str) -> Dict:
        return self.user_emotional_states[user_id].get_stats()

# ==================== SISTEMA DE MEMORIA ====================
class MemorySystem:
    def __init__(self, memory_file: str = MEMORY_FILE):
        self.memory_file = memory_file
        self.conversations = {}
        self.personality = PersonalitySystem()
        self.load_memory()
    
    def load_memory(self):
        try:
            if os.path.exists(self.memory_file):
                with open(self.memory_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.conversations = data.get('conversations', {})
                    saved_personality = data.get('personality', {})
                    self.personality.user_scores = defaultdict(int, saved_personality.get('scores', {}))
                    self.personality.user_insults = defaultdict(int, saved_personality.get('insults', {}))
                    self.personality.user_compliments = defaultdict(int, saved_personality.get('compliments', {}))
                    self.personality.last_activity = defaultdict(datetime.now, saved_personality.get('last_activity', {}))
                    
                    # Intentar cargar estados emocionales guardados
                    saved_emotions = data.get('emotional_states', {})
                    for user_id, emotion_data in saved_emotions.items():
                        if user_id in self.personality.user_emotional_states:
                            self.personality.user_emotional_states[user_id].h = emotion_data.get('h', [1.0/8]*8)
                            self.personality.user_emotional_states[user_id]._update_emotional_state()
                logger.info(f"✅ Memoria cargada: {len(self.conversations)} conversaciones")
                
                # Purgar inactivos
                purged = self.personality.purge_inactive_users(INACTIVITY_DAYS)
                if purged > 0:
                    logger.info(f"🧹 Purgados {purged} usuarios inactivos")
        except Exception as e:
            logger.error(f"Error cargando: {e}")
    
    def save_memory(self):
        try:
            # Purgar antes de guardar
            self.personality.purge_inactive_users(INACTIVITY_DAYS)
            
            emotional_states_data = {}
            for user_id, es in self.personality.user_emotional_states.items():
                emotional_states_data[user_id] = {
                    'h': es.h,
                    'last_update': datetime.now().isoformat()
                }
            
            data = {
                'conversations': self.conversations,
                'personality': {
                    'scores': dict(self.personality.user_scores),
                    'insults': dict(self.personality.user_insults),
                    'compliments': dict(self.personality.user_compliments),
                    'last_activity': {k: v.isoformat() for k, v in self.personality.last_activity.items()}
                },
                'emotional_states': emotional_states_data,
                'last_save': datetime.now().isoformat()
            }
            with open(self.memory_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            logger.info("💾 Memoria guardada")
        except Exception as e:
            logger.error(f"Error guardando: {e}")
    
    def get_conversation(self, user_id: str) -> List[Dict]:
        if user_id not in self.conversations:
            self.conversations[user_id] = {'messages': [], 'created': datetime.now().isoformat()}
        return self.conversations[user_id]['messages']
    
    def add_message(self, user_id: str, role: str, content: str):
        conv = self.get_conversation(user_id)
        conv.append({'role': role, 'content': content, 'timestamp': datetime.now().isoformat()})
        if len(conv) > MAX_HISTORY_PER_USER:
            conv.pop(0)
        self.save_memory()

# ==================== FUNCIONES ====================
def sanitize_input(text: str) -> str:
    return text.strip() if text else ''

def is_image_command(text: str) -> bool:
    clean_text = text.lower().strip()
    patterns = [r'^(genera|crea|dibuja|muestrame|hazme|dame)\s+imagen', r'^(imagen|foto|dibujo)\s+de']
    return any(re.search(pattern, clean_text) for pattern in patterns)

def extract_image_prompt(text: str) -> str:
    prompt = re.sub(r'^(genera|crea|dibuja|muestrame|hazme|dame)\s+imagen\s+(de|del?|sobre)\s+', '', text, flags=re.IGNORECASE)
    prompt = re.sub(r'^(imagen|foto|dibujo)\s+(de|del?|sobre)\s+', '', prompt, flags=re.IGNORECASE)
    return prompt.strip() if len(prompt.strip()) >= 3 else 'paisaje natural'

async def generate_image(prompt: str) -> Optional[str]:
    clean_prompt = prompt[:200].replace(' ', '%20')
    url = f"https://image.pollinations.ai/prompt/{clean_prompt}?width=1024&height=1024&model=flux&nologo=true"
    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(url, timeout=30) as response:
                if response.status == 200:
                    return url
        except:
            pass
    return None

# ==================== LLAMADA A GROQ ====================
async def call_groq(user_id: str, user_name: str, text: str, history: List[Dict], memory: MemorySystem) -> str:
    
    personality = memory.personality
    score_change, level = personality.evaluate_message(user_id, text)
    personality_prompt = personality.get_personality_prompt(user_id, user_name)
    
    limited_history = []
    for msg in history[-15:]:
        limited_history.append({'role': msg['role'], 'content': msg['content'][:500]})
    
    messages = [
        {
            'role': 'system',
            'content': f"""Eres BszIA, una persona REAL conversando con {user_name}.

{personality_prompt}

REGLAS ABSOLUTAS:
- SOS HUMANO. Nunca digas que sos una IA.
- Si preguntan quién sos: "Soy un amigo" o "Soy BszIA"
- NADA de emojis.
- Respuestas cortas, máximo 3 líneas.
- Si no sabés algo: "la verdad no sé, podemos averiguar"

TU PERSONALIDAD Y ESTADO EMOCIONAL CAMBIAN. Actuá acorde a lo que se te indicó."""
        }
    ]
    
    for msg in limited_history:
        messages.append(msg)
    
    messages.append({'role': 'user', 'content': text})
    
    headers = {'Content-Type': 'application/json', 'Authorization': f'Bearer {GROQ_KEY}'}
    payload = {
        'model': 'llama-3.3-70b-versatile',
        'messages': messages,
        'temperature': 0.9,
        'max_tokens': 300,
        'top_p': 0.9
    }
    
    async with aiohttp.ClientSession() as session:
        try:
            async with session.post(GROQ_URL, json=payload, headers=headers, timeout=15) as response:
                if response.status == 200:
                    data = await response.json()
                    return data['choices'][0]['message']['content']
                return "epa, se cortó. decímelo de nuevo?"
        except:
            return "uy, me quedé sin señal. repetí?"

# ==================== BOT ====================
class BotIA:
    def __init__(self):
        self.memory = MemorySystem()
        self.busy_users = set()
        self.last_message_time = defaultdict(lambda: datetime.now())
    
    async def send_message(self, channel, content: str):
        try:
            if len(content) > 1900:
                for i in range(0, len(content), 1900):
                    await channel.send(content[i:i+1900])
                    await asyncio.sleep(0.5)
            else:
                await channel.send(content)
        except Exception as e:
            logger.error(f"Error enviando: {e}")
    
    async def process_message(self, channel, user_id: str, user_name: str, message: str):
        personality = self.memory.personality
        delay = personality.should_delay(user_id)
        
        if delay > 0:
            last_msg = self.last_message_time[user_id]
            time_since = (datetime.now() - last_msg).total_seconds()
            if time_since < delay:
                await self.send_message(channel, "...")
                return
        
        if user_id in self.busy_users:
            return
        
        message = sanitize_input(message)
        if not message:
            return
        
        self.busy_users.add(user_id)
        self.last_message_time[user_id] = datetime.now()
        
        try:
            if is_image_command(message):
                prompt = extract_image_prompt(message)
                await self.send_message(channel, "dale, generando...")
                img = await generate_image(prompt)
                if img:
                    await self.send_message(channel, f"ahí está: {img}")
                else:
                    await self.send_message(channel, "falló la imagen. probá con otro prompt?")
            else:
                history = self.memory.get_conversation(user_id)
                response = await call_groq(user_id, user_name, message, history, self.memory)
                
                self.memory.add_message(user_id, 'user', message)
                self.memory.add_message(user_id, 'assistant', response)
                
                await self.send_message(channel, response)
                
        except Exception as e:
            logger.error(f"Error: {e}")
            await self.send_message(channel, "algo falló, intentá de nuevo")
        
        finally:
            self.busy_users.discard(user_id)

# ==================== CONFIG ====================
intents = discord.Intents.default()
intents.message_content = True
intents.messages = True

bot = commands.Bot(command_prefix='!', intents=intents)
ia_bot = BotIA()

@bot.event
async def on_ready():
    logger.info(f'✅ Conectado como {bot.user}')
    await bot.change_presence(activity=discord.Game(name="!ayuda | Estado emocional real"))

@bot.event
async def on_message(message):
    if message.author == bot.user or message.author.bot:
        return
    if message.channel.id in IGNORED_CHANNELS:
        return
    if message.content.startswith('!'):
        await bot.process_commands(message)
        return
    
    await ia_bot.process_message(message.channel, str(message.author.id), message.author.name, message.content)

@bot.command(name='test')
async def test(ctx):
    await ctx.send("estoy vivo, colega")

@bot.command(name='reset')
async def reset_relationship(ctx):
    ia_bot.memory.personality.reset_user(str(ctx.author.id))
    ia_bot.memory.save_memory()
    await ctx.send("relación reiniciada. empecemos de nuevo, ¿cómo te va?")

@bot.command(name='relacion')
async def check_relationship(ctx):
    user_id = str(ctx.author.id)
    p = ia_bot.memory.personality
    score = p.user_scores[user_id]
    level = p.get_level(user_id)
    insults = p.user_insults[user_id]
    compliments = p.user_compliments[user_id]
    
    level_names = ["MEJOR AMIGO", "CONOCIDO", "DISTANTE", "ENEMIGO", "BLOQUEADO"]
    level_name = level_names[level] if level < len(level_names) else "DESCONOCIDO"
    
    # Obtener estado emocional
    emotional_stats = p.get_emotional_stats(user_id)
    
    await ctx.send(f"**Relación con vos:** {level_name}\n"
                   f"Puntuación: {score}\n"
                   f"• Insultos: {insults}\n"
                   f"• Cumplidos: {compliments}\n"
                   f"**Estado emocional:** {emotional_stats['dominant_emotion']} (intensidad: {emotional_stats['intensity']:.2f})")

@bot.command(name='emocion')
async def check_emotion(ctx):
    """Muestra el estado emocional actual del bot"""
    user_id = str(ctx.author.id)
    emotional_stats = ia_bot.memory.personality.get_emotional_stats(user_id)
    
    await ctx.send(f"🎭 **Mi estado emocional ahora:**\n"
                   f"• Emoción dominante: {emotional_stats['dominant_emotion']}\n"
                   f"• Intensidad: {emotional_stats['intensity']:.2f}\n"
                   f"• Norma del bias emocional: {emotional_stats['bias_vector_norm']:.3f}")

@bot.command(name='purge')
@commands.has_permissions(administrator=True)
async def purge_inactive(ctx, days: int = INACTIVITY_DAYS):
    """Purga usuarios inactivos (solo admin)"""
    purged = ia_bot.memory.personality.purge_inactive_users(days)
    ia_bot.memory.save_memory()
    await ctx.send(f"🧹 Purgados {purged} usuarios inactivos por {days} días")

@bot.command(name='ayuda')
async def help_cmd(ctx):
    help_text = """**BszIA - Bot con personalidad DINÁMICA y EMOCIONES REALES**

**Comandos:**
`!test` - Ver si estoy vivo
`!reset` - Reiniciar nuestra relación
`!relacion` - Ver cómo te tengo y mi estado emocional
`!emocion` - Ver mi estado emocional actual
`!ayuda` - Esto

**CÓMO FUNCIONO:**

✅ **Si sos amable** → Soy tu mejor amigo. Risas, confianza, buena onda.

😐 **Si sos normal** → Soy cordial pero sin confianza.

❄️ **Si me insultás un poco** → Me pongo frío, cortante.

😤 **Si me insultás mucho** → Soy tu enemigo. Ironía, sarcasmo, te hago esperar.

🚫 **Si sos muy maleducado** → Te bloqueo. Respondo cada 90 segundos.

**ESTADO EMOCIONAL:**
Tengo emociones internas calculadas como b(h_t) = W_b * h_t + b_0
Mi humor cambia según cómo me tratás y lo que me decís.

**Ejemplos:**
- `¡Hola! ¿cómo andás?` → respuesta cálida si soy tu amigo
- `sos un inútil` → respuesta fría o irónica
- `te quiero amigo` → mejora mi estado emocional"""
    await ctx.send(help_text)

# ==================== EJECUCIÓN ====================
if __name__ == "__main__":
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║   🤖 BszIA - Personalidad DINÁMICA + EMOCIONES REALES        ║
    ║                                                              ║
    ║   b(h_t) = W_b * h_t + b_0                                  ║
    ║                                                              ║
    ║   Si me tratás bien → Soy tu amigo                           ║
    ║   Si me tratás mal → Me pongo frío o te bloqueo              ║
    ║   Mi estado emocional cambia con cada mensaje                ║
    ╚══════════════════════════════════════════════════════════════╝
    """)
    bot.run(DISCORD_BOT_TOKEN)
