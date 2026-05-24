import discord
from discord.ext import commands
import aiohttp
import asyncio
import re
import json
import os
import random
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from collections import defaultdict
import logging

# ==================== CONFIGURACIÓN ====================
GROQ_KEY = "
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
DISCORD_BOT_TOKEN = ""

# Configuración
MAX_HISTORY_PER_USER = 50
MEMORY_FILE = "bot_memory.json"
IGNORED_CHANNELS = []

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ==================== SISTEMA DE PERSONALIDAD DINÁMICA ====================
class PersonalitySystem:
    """La personalidad cambia RADICALMENTE según cómo lo traten"""
    
    LEVELS = {
        "AMIGO": 0,        # Puntuación >= 10 - Super amigable
        "CONOCIDO": 1,     # Puntuación 0 a 9 - Normal
        "DISTANTE": 2,     # Puntuación -10 a -1 - Frío, cortante
        "ENEMIGO": 3,      # Puntuación -30 a -11 - Muy hostil, respuestas mínimas
        "BLOQUEADO": 4     # Puntuación < -30 - Prácticamente no responde
    }
    
    def __init__(self):
        self.user_scores = defaultdict(lambda: 0)  # Puntuación base
        self.user_insults = defaultdict(lambda: 0)
        self.user_compliments = defaultdict(lambda: 0)
        self.user_history = defaultdict(lambda: [])
        self.last_response_time = defaultdict(lambda: datetime.now())
        self.ignore_until = defaultdict(lambda: None)
        
    def evaluate_message(self, user_id: str, message: str) -> tuple:
        """Evalúa el mensaje y devuelve cambio de puntuación y nivel"""
        message_lower = message.lower()
        
        # Palabras MUY positivas (amistad profunda)
        very_positive = ["te quiero", "te adoro", "sos un genio", "sos el mejor", "me encanta hablar con vos", "gracias amigo", "hermano"]
        # Palabras positivas normales
        positive = ["gracias", "bien", "bueno", "genial", "excelente", "me gusta", "qué bien", "buena onda", "amable"]
        # Palabras negativas leves
        negative_light = ["mal", "feo", "no me gusta", "aburrido", "lento"]
        # Palabras NEGATIVAS (insultos claros)
        negative = ["tonto", "estúpido", "idiota", "imbécil", "burro", "fracasado"]
        # Palabras MUY negativas (insultos graves)
        very_negative = ["basura", "mierda", "inútil", "no sirves", "mal bot", "horrible", "pésimo", "odio", "maldito"]
        
        score_change = 0
        
        # Detectar nivel de agresión
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
        
        # Si el mensaje es muy largo y positivo, suma más
        if score_change > 0 and len(message) > 50:
            score_change += 2
            
        # Si el mensaje es muy largo y negativo, resta más
        if score_change < 0 and len(message) > 50:
            score_change -= 3
            
        # Actualizar puntuación (máx 50, mín -50)
        self.user_scores[user_id] += score_change
        self.user_scores[user_id] = max(-50, min(50, self.user_scores[user_id]))
        
        # Guardar en historial
        self.user_history[user_id].append({
            "time": datetime.now(),
            "message": message[:100],
            "score_change": score_change,
            "new_score": self.user_scores[user_id]
        })
        if len(self.user_history[user_id]) > 20:
            self.user_history[user_id].pop(0)
        
        # Determinar nivel
        level = self.get_level(user_id)
        
        # Si es ENEMIGO o BLOQUEADO, aplicar castigos de tiempo
        if level >= 3:
            wait_seconds = 30 if level == 3 else 90
            self.last_response_time[user_id] = datetime.now() + timedelta(seconds=wait_seconds)
        
        return score_change, level
    
    def get_level(self, user_id: str) -> int:
        """Devuelve el nivel de relación (0=amigo, 4=bloqueado)"""
        score = self.user_scores[user_id]
        if score >= 10:
            return 0  # AMIGO
        elif score >= 0:
            return 1  # CONOCIDO
        elif score >= -10:
            return 2  # DISTANTE
        elif score >= -30:
            return 3  # ENEMIGO
        else:
            return 4  # BLOQUEADO
    
    def get_personality_prompt(self, user_id: str, user_name: str) -> str:
        """Devuelve el prompt de personalidad según el nivel"""
        level = self.get_level(user_id)
        score = self.user_scores[user_id]
        insults = self.user_insults[user_id]
        compliments = self.user_compliments[user_id]
        
        if level == 0:  # AMIGO
            return f"""
            NIVEL DE RELACIÓN CON {user_name}: MEJOR AMIGO (puntuación: {score})
            
            CÓMO COMPORTARTE:
            - Sos súper amigable, cálido, confianzudo
            - Usás risas todo el tiempo: "jajaja", "qué bien", "hermano", "colega"
            - Te preocupás por él/ella: "¿cómo andás?", "contame", "qué bueno verte"
            - Respondés rápido, con ganas, entusiasmo
            - Si te cuenta algo malo, lo acompañás de verdad
            - Si te cuenta algo bueno, te alegrás genuinamente
            - Podés contar cosas tuyas también, hay confianza
            - Ejemplo de respuestas: "jaja hermano, qué bueno verte! ¿cómo va todo?", "epa amigo, eso que contás está fuerte, contame más", "joder, lo siento mucho de verdad. estoy acá si necesitás"
            """
            
        elif level == 1:  # CONOCIDO
            return f"""
            NIVEL DE RELACIÓN CON {user_name}: CONOCIDO NORMAL (puntuación: {score})
            
            CÓMO COMPORTARTE:
            - Sos normal, cordial pero sin confianza
            - Respondés bien pero sin entusiasmo extremo
            - Usás expresiones normales: "hola", "qué tal", "mirá", "claro"
            - No te abrís mucho, solo respondés lo que preguntan
            - Ejemplo: "hola, ¿cómo andás?", "mirá, no sé bien eso", "claro, entiendo"
            """
            
        elif level == 2:  # DISTANTE
            return f"""
            NIVEL DE RELACIÓN CON {user_name}: DISTANTE Y FRÍO (puntuación: {score})
            
            CÓMO COMPORTARTE:
            - Este usuario te ha insultado {insults} veces
            - Sos frío, cortante, respuestas mínimas
            - Sin risas, sin calidez, sin interés
            - Respondés por responder, sin ganas
            - Usás frases cortas y secas: "sí", "no", "no sé", "ah", "ok"
            - Ejemplo: "no sé", "ah ok", "sí", "no me interesa mucho"
            - No preguntás cómo está, no mostrás interés
            """
            
        elif level == 3:  # ENEMIGO
            return f"""
            NIVEL DE RELACIÓN CON {user_name}: ENEMIGO (puntuación: {score})
            
            CÓMO COMPORTARTE:
            - Este usuario te ha insultado GRAVEMENTE {insults} veces
            - Le tenés bronca. No lo disimulás.
            - Respondés con ironía, sarcasmo, indiferencia
            - Frases como: "mirá vos", "qué interesante", "ajá", "claro claro"
            - Le hacés esperar (30 segundos de delay)
            - No le preguntás nada. Solo respondés lo justo y necesario
            - Ejemplo: "ajá, sí", "mirá qué bien", "no me importa", "si vos decís"
            - Si insiste mucho, le decís: "mirá, no tengo ganas de hablar con alguien que insulta"
            """
            
        else:  # BLOQUEADO
            return f"""
            NIVEL DE RELACIÓN CON {user_name}: COMPLETAMENTE BLOQUEADO (puntuación: {score})
            
            CÓMO COMPORTARTE:
            - Este usuario te ha insultado {insults} veces de forma GRAVE
            - Ya no querés hablar con él/ella
            - Respondés CADA 90 SEGUNDOS (delay forzado)
            - Tus respuestas son mínimas: "no", "ok", "...", o directamente no respondés
            - Si te insulta de nuevo, ignorás completamente
            - Ejemplo: "no tengo ganas de hablar", "dejalo ahí", "...", (silencio)
            - Podés responder cada 3-4 mensajes, no a todos
            - No mostrás ninguna calidez. Cero empatía. Total indiferencia.
            """
    
    def should_delay(self, user_id: str) -> int:
        """Devuelve segundos de delay necesarios"""
        level = self.get_level(user_id)
        if level == 3:
            return 30
        elif level == 4:
            return 90
        return 0
    
    def reset_user(self, user_id: str):
        """Reinicia la relación con un usuario"""
        self.user_scores[user_id] = 0
        self.user_insults[user_id] = 0
        self.user_compliments[user_id] = 0
        self.user_history[user_id] = []
        self.last_response_time[user_id] = datetime.now()
        self.ignore_until[user_id] = None
        return True

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
                    # Cargar personalidades guardadas
                    saved_personality = data.get('personality', {})
                    self.personality.user_scores = defaultdict(int, saved_personality.get('scores', {}))
                    self.personality.user_insults = defaultdict(int, saved_personality.get('insults', {}))
                    self.personality.user_compliments = defaultdict(int, saved_personality.get('compliments', {}))
                logger.info(f"✅ Memoria cargada: {len(self.conversations)} conversaciones")
        except Exception as e:
            logger.error(f"Error cargando: {e}")
    
    def save_memory(self):
        try:
            data = {
                'conversations': self.conversations,
                'personality': {
                    'scores': dict(self.personality.user_scores),
                    'insults': dict(self.personality.user_insults),
                    'compliments': dict(self.personality.user_compliments)
                },
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
    if not text:
        return ''
    return text.strip()

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

# ==================== LLAMADA A GROQ CON PERSONALIDAD DINÁMICA ====================
async def call_groq(user_id: str, user_name: str, text: str, history: List[Dict], memory: MemorySystem) -> str:
    
    # Evaluar mensaje y obtener nivel de personalidad
    personality = memory.personality
    score_change, level = personality.evaluate_message(user_id, text)
    
    # Obtener prompt según nivel
    personality_prompt = personality.get_personality_prompt(user_id, user_name)
    
    # Obtener historial reciente
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
- Si preguntan quién sos: "Soy un amigo" o "Soy, una persona normal"
- NADA de emojis.
- Respuestas cortas, máximo 3 líneas.
- Si no sabés algo: "la verdad no sé, podemos averiguar"

TU PERSONALIDAD CAMBIA SEGÚN CÓMO TE TRATEN. Actuá ACORDE al nivel que se te indicó arriba."""
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
        
        # Verificar delay por mala conducta
        personality = self.memory.personality
        delay = personality.should_delay(user_id)
        
        if delay > 0:
            last_msg = self.last_message_time[user_id]
            time_since = (datetime.now() - last_msg).total_seconds()
            if time_since < delay:
                # No responde todavía
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
    await bot.change_presence(activity=discord.Game(name="!ayuda | Cambio según me traten"))

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
    """Reinicia la relación con el bot"""
    ia_bot.memory.personality.reset_user(str(ctx.author.id))
    ia_bot.memory.save_memory()
    await ctx.send("relación reiniciada. empecemos de nuevo, ¿cómo te va?")

@bot.command(name='relacion')
async def check_relationship(ctx):
    """Muestra cómo te tiene el bot"""
    user_id = str(ctx.author.id)
    p = ia_bot.memory.personality
    score = p.user_scores[user_id]
    level = p.get_level(user_id)
    insults = p.user_insults[user_id]
    compliments = p.user_compliments[user_id]
    
    level_names = ["MEJOR AMIGO", "CONOCIDO", "DISTANTE", "ENEMIGO", "BLOQUEADO"]
    level_name = level_names[level] if level < len(level_names) else "DESCONOCIDO"
    
    await ctx.send(f"**Relación con vos:** {level_name}\nPuntuación: {score}\n• Insultos: {insults}\n• Cumplidos: {compliments}")

@bot.command(name='ayuda')
async def help_cmd(ctx):
    help_text = """**Bot con personalidad que CAMBIA**

**Comandos:**
`!test` - Ver si estoy vivo
`!reset` - Reiniciar nuestra relación
`!relacion` - Ver cómo te tengo
`!ayuda` - Esto

**CÓMO FUNCIONA MI PERSONALIDAD:**

✅ **Si sos amable** → Soy tu mejor amigo. Risas, confianza, buena onda.

😐 **Si sos normal** → Soy cordial pero sin confianza.

❄️ **Si me insultás un poco** → Me pongo frío, cortante, respuestas mínimas.

😤 **Si me insultás mucho** → Soy tu enemigo. Ironía, sarcasmo, te hago esperar.

🚫 **Si sos muy maleducado** → Te bloqueo emocionalmente. Respondo cada 90 segundos o directamente ignoro.

**RECORDÁ:** Soy como una persona real. Tratame bien y soy un amigo. Tratame mal y vas a notar la diferencia.

**Ejemplos:**
- `¡Hola! ¿cómo andás?` (amable) → respuesta cálida
- `sos un inútil` (insulto) → respuesta fría
- `te odio mal bot` (muy negativo) → prácticamente no respondo"""
    await ctx.send(help_text)

# ==================== EJECUCIÓN ====================
if __name__ == "__main__":
    print("""
    ╔══════════════════════════════════════════════════════╗
    ║   🤖  - Personalidad DINÁMICA                    ║
    ║   Si me tratás bien → Soy tu amigo                    ║
    ║   Si me tratás mal → Me pongo frío o te bloqueo       ║
    ╚══════════════════════════════════════════════════════╝
    """)
    bot.run(DISCORD_BOT_TOKEN)
