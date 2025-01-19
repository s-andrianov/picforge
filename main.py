import asyncio
from bot.bot import ImageGenerationBot
from utils.config import Config

async def main():
    config = Config('config.yaml')
    bot = ImageGenerationBot(config)
    await bot.run()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        print(f"Error: {e}")