import discord
from discord.ext import commands

# 봇의 프리픽스 설정
bot = commands.Bot(command_prefix='!')

@bot.event
async def on_ready():
    print(f'봇이 준비되었습니다: {bot.user.name}')

@bot.command()
async def ping(ctx):
    await ctx.send('Pong!')

@bot.command()
async def hello(ctx):
    await ctx.send('안녕하세요!')

@bot.command()
async def echo(ctx, *, message: str):
    await ctx.send(message)

# 여기서 YOUR_BOT_TOKEN을 복사한 토큰으로 교체합니다.
bot.run('BOT_TOKEN')
