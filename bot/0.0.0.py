import discord
from discord.ext import commands
import aiohttp
import asyncio
import re
from datetime import datetime
from typing import List, Dict, Any

# ==================== CONFIGURACIÓN ====================
GROQ_KEY = ""
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
DISCORD_BOT_TOKEN = ""

# IDs de canales donde NO quieres que responda (ignorar)
IGNORED_CHANNELS = []  # Puedes añadir IDs aquí si quieres ignorar algún canal

# ==================== FUNCIONES DE UTILIDAD ====================
def sanitize_input(text: str) -> str:
    if not text:
        return ''
    dangerous_patterns = [
        r'<script\b[^<]*(?:(?!<\/script>)<[^<]*)*<\/script>',
        r'javascript:', r'on\w+\s*=', r'<iframe', r'<object', r'<embed'
    ]
    for pattern in dangerous_patterns:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)
    return text

def is_image_command(text: str) -> bool:
    clean_text = text.lower().strip()
    patterns = [
        r'^(genera|crea|dibuja|muestrame|hazme|dame)\s+(una?|un)?\s*imagen\s+(de|del?|sobre)',
        r'^(imagen|foto|dibujo)\s+(de|del?|sobre)'
    ]
    return any(re.search(pattern, clean_text) for pattern in patterns)

def extract_image_prompt(text: str) -> str:
    prompt = text
    prefixes = [
        r'^(genera|crea|dibuja|muestrame|hazme|dame)\s+(una?|un)?\s*imagen\s+(de|del?|sobre)\s+',
        r'^(imagen|foto|dibujo)\s+(de|del?|sobre)\s+'
    ]
    for pattern in prefixes:
        prompt = re.sub(pattern, '', prompt, flags=re.IGNORECASE)
    prompt = prompt.strip()
    return prompt if prompt and len(prompt) >= 3 else 'paisaje natural'

async def generate_image(prompt: str, retry: int = 0) -> str:
    clean_prompt = prompt[:200].replace(' ', '%20')
    image_url = f"https://image.pollinations.ai/prompt/{clean_prompt}?width=1024&height=1024&model=flux&nologo=true&seed={int(datetime.now().timestamp() * 1000) + retry}"
    
    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(image_url, timeout=aiohttp.ClientTimeout(total=30)) as response:
                if response.status == 200:
                    return image_url
                raise Exception(f"Error HTTP: {response.status}")
        except asyncio.TimeoutError:
            if retry < 2:
                return await generate_image(prompt, retry + 1)
            raise Exception("Timeout al generar imagen")
        except Exception as e:
            if retry < 2:
                return await generate_image(prompt, retry + 1)
            raise Exception(f"Error: {str(e)}")

async def call_groq(user_name: str, text: str, history: List[Dict[str, Any]]) -> str:
    limited_history = []
    for msg in history[-10:]:
        if msg.get('type') == 'text' and msg.get('content'):
            content = msg['content'][:2000] if len(msg['content']) > 2000 else msg['content']
            limited_history.append({'role': msg['role'], 'content': content})
    
    messages = [
        {
            'role': 'system',
            'content': f"""Eres una persona normal llamada BszIA, estás conversando con {user_name}. 

INSTRUCCIONES:
- Responde como un humano normal, natural y espontáneo
- Usa lenguaje conversacional, como si hablaras con un amigo
- Expresa emociones: "jaja", "uy", "epa", "oye", "mira"
- Sé amigable pero no demasiado formal
- No menciones que eres una IA
- NO uses emojis
- NO des respuestas largas, sé directo pero natural

RESPONDE EN ESPAÑOL, de forma natural y conversacional."""
        },
        *limited_history,
        {'role': 'user', 'content': text}
    ]
    
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {GROQ_KEY}'
    }
    
    payload = {
        'model': 'llama-3.3-70b-versatile',
        'messages': messages,
        'temperature': 0.95,
        'max_tokens': 2000,
        'top_p': 0.9
    }
    
    async with aiohttp.ClientSession() as session:
        async with session.post(GROQ_URL, json=payload, headers=headers) as response:
            if response.status != 200:
                error_text = await response.text()
                raise Exception(f"Error en Groq: {response.status}")
            data = await response.json()
            return data['choices'][0]['message']['content']

# ==================== CLASE DEL BOT ====================
class BotIA:
    def __init__(self):
        self.busy_per_user = {}
        self.conversations = {}
    
    async def send_message(self, channel, content: str):
        try:
            # Dividir mensajes largos en partes
            if len(content) > 1900:
                parts = [content[i:i+1900] for i in range(0, len(content), 1900)]
                for part in parts:
                    await channel.send(part)
            else:
                await channel.send(content)
        except Exception as e:
            print(f"Error enviando mensaje: {e}")
    
    async def process_message(self, channel, user_id: str, user_name: str, message: str):
        # Verificar si el usuario está ocupado
        if self.busy_per_user.get(user_id, False):
            await self.send_message(channel, "epa un momento que estoy pensando...")
            return
        
        message = sanitize_input(message)
        
        # Inicializar conversación
        if user_id not in self.conversations:
            self.conversations[user_id] = []
        
        self.busy_per_user[user_id] = True
        
        try:
            # Comando de imagen
            if is_image_command(message):
                prompt = extract_image_prompt(message)
                await self.send_message(channel, f"dale, déjame generarte esa imagen de {prompt}...")
                
                try:
                    image_url = await generate_image(prompt)
                    await self.send_message(channel, f"ahí está {user_name}, mirá: {image_url}")
                except Exception as e:
                    await self.send_message(channel, f"uy no, falló la imagen: {str(e)}")
            else:
                # Respuesta normal
                response = await call_groq(user_name, message, self.conversations[user_id])
                
                # Guardar historial
                self.conversations[user_id].append({'role': 'user', 'content': message, 'type': 'text'})
                self.conversations[user_id].append({'role': 'assistant', 'content': response, 'type': 'text'})
                
                # Mantener historial corto
                if len(self.conversations[user_id]) > 20:
                    self.conversations[user_id] = self.conversations[user_id][-20:]
                
                # Enviar respuesta
                await self.send_message(channel, response)
                
        except Exception as e:
            await self.send_message(channel, f"uy se me fue la luz, {str(e)[:100]}")
            print(f"Error: {e}")
        
        finally:
            self.busy_per_user[user_id] = False

# ==================== CONFIGURACIÓN DEL BOT ====================
intents = discord.Intents.default()
intents.message_content = True
intents.messages = True

bot = commands.Bot(command_prefix='!', intents=intents)
ia_bot = BotIA()

# ==================== EVENTOS ====================
@bot.event
async def on_ready():
    print(f'✅ Conectado como {bot.user}')
    print(f'📡 Servidores donde estoy:')
    
    for guild in bot.guilds:
        print(f'   📌 Servidor: {guild.name} (ID: {guild.id})')
        channels = [f'#{channel.name}' for channel in guild.text_channels]
        print(f'      Canales: {", ".join(channels[:5])}')
    
    print(f'\n🎯 El bot responderá en TODOS los canales de texto')
    print(f'   (a menos que especifiques IDs en IGNORED_CHANNELS)')

@bot.event
async def on_message(message):
    # Ignorar mensajes del propio bot
    if message.author == bot.user:
        return
    
    # Ignorar mensajes de otros bots (opcional)
    if message.author.bot:
        return
    
    # Ignorar canales en la lista negra
    if message.channel.id in IGNORED_CHANNELS:
        return
    
    # Ignorar mensajes vacíos
    if not message.content.strip():
        return
    
    # Mostrar en consola
    print(f"📨 [{message.channel.name}] {message.author.name}: {message.content[:50]}")
    
    # Procesar mensaje (responde en el mismo canal)
    await ia_bot.process_message(
        message.channel,
        str(message.author.id),
        message.author.name,
        message.content
    )
    
    # Procesar comandos (los que empiezan con !)
    await bot.process_commands(message)

# ==================== COMANDOS ====================
@bot.command(name='test')
async def test_command(ctx):
    """Comando de prueba"""
    await ctx.send("✅ Estoy vivo y respondiendo!")

@bot.command(name='olvidar')
async def forget_conversation(ctx):
    """Olvida la conversación del usuario"""
    if ctx.author.id in ia_bot.conversations:
        del ia_bot.conversations[ctx.author.id]
        await ctx.send("epa, listo. Conversación nueva, ¿de qué querés hablar?")
    else:
        await ctx.send("no tenías conversación activa, empecemos de cero...")

@bot.command(name='quiensos')
async def who_am_i(ctx):
    """Responde quién es"""
    await ctx.send("soy BszIA, un amigo que anda por acá. nada más, nada menos.")

@bot.command(name='ayuda')
async def help_command(ctx):
    """Muestra los comandos disponibles"""
    help_text = """**Comandos disponibles:**
    `!test` - Prueba si el bot responde
    `!olvidar` - Reinicia la conversación
    `!quiensos` - Pregunta quién soy
    `!ayuda` - Muestra esta ayuda

**Generar imágenes:**
    `genera una imagen de [descripción]`
    `crea una imagen de [descripción]`
    
**Ejemplos:**
    ¡Hola! - Conversación normal
    genera una imagen de un gato en el espacio
    """
    await ctx.send(help_text)

# ==================== EJECUCIÓN ====================
if __name__ == "__main__":
    print("🚀 Iniciando bot...")
    print("📝 El bot responderá en TODOS los canales donde esté presente")
    bot.run(DISCORD_BOT_TOKEN)
