import logging
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, MessageHandler, filters, CallbackQueryHandler
from wqueue.request_queue import RequestQueue
from generation.generator import ImageGenerator
from utils.config import Config
from utils.logger import Logger
from utils.resource_scanner import ResourceScanner
import asyncio
import json
import os
from telegram.ext import ContextTypes

class ImageGenerationBot:
    def __init__(self, config: Config):
        self.config = config
        self.queue = RequestQueue()
        self.generator = ImageGenerator(config)
        self.resource_scanner = ResourceScanner(config)
        self.application = Application.builder().token(config.get('bot_token')).build()
        self.user_settings = {}
        self.last_settings = {}

        # Настройка логирования
        logging.basicConfig(
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            level=logging.INFO
        )
        self.logger = logging.getLogger(__name__)

    async def start(self, update: Update, context):
        if update.effective_user.id != self.config.allowed_user_id:
            self.logger.warning(f"Unauthorized access attempt from user {update.effective_user.id}")
            await update.message.reply_text("Извините, у вас нет доступа к этому боту.")
            return
        await self.show_interactive_panel(update, context)

    async def show_interactive_panel(self, update: Update, context):
        user_id = update.effective_user.id
        settings = self.user_settings.get(user_id, self.config.default_settings)
        
        keyboard = [
            [InlineKeyboardButton("🚀 Начать генерацию", callback_data='start_generation')],
            [InlineKeyboardButton(f"Модель: {settings['model']}", callback_data='change_model')],
            [InlineKeyboardButton(f"VAE: {settings['vae']}", callback_data='change_vae')],
            [InlineKeyboardButton(f"LoRA: {settings['lora']}", callback_data='change_lora')],
            [InlineKeyboardButton(f"Семплер: {settings['sampler']}", callback_data='change_sampler')],
            [InlineKeyboardButton(f"CFG Scale: {settings['cfg_scale']}", callback_data='change_cfg')],
            [InlineKeyboardButton(f"Шаги: {settings['steps']}", callback_data='change_steps')],
            [InlineKeyboardButton(f"Размер: {settings['size']}", callback_data='change_size')],
            [InlineKeyboardButton("Изменить промпт", callback_data='change_prompt')],
            [InlineKeyboardButton("Изменить негативный промпт", callback_data='change_negative_prompt')],
            [InlineKeyboardButton("Применить стандартные", callback_data='apply_default')]
        ]
        
        if user_id in self.last_settings:
            keyboard.append([InlineKeyboardButton("Применить последние", callback_data='apply_last')])
        
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        message = "Настройте параметры генерации или начните генерацию с текущими настройками:"
        
        try:
            if isinstance(update, Update):
                if update.message:
                    await update.message.reply_text(message, reply_markup=reply_markup)
                elif update.callback_query:
                    await update.callback_query.message.edit_text(message, reply_markup=reply_markup)
            else:
                await context.bot.send_message(chat_id=user_id, text=message, reply_markup=reply_markup)
        except Exception as e:
            self.logger.error(f"Error in show_interactive_panel: {str(e)}")
            if isinstance(update, Update) and update.effective_message:
                await update.effective_message.reply_text("Произошла ошибка при отображении панели настроек. Пожалуйста, попробуйте еще раз.")

    async def handle_callback(self, update: Update, context):
        query = update.callback_query
        await query.answer()
        
        try:
            if query.data == 'start_generation':
                await self.start_generation(update, context)
            elif query.data.startswith('change_'):
                param = query.data[7:]  # Убираем 'change_' из начала
                await self.change_parameter(update, context, param)
            elif query.data == 'apply_default':
                await self.apply_default_settings(update, context)
            elif query.data == 'apply_last':
                await self.apply_last_settings(update, context)
            elif query.data == 'back_to_panel':
                await self.show_interactive_panel(update, context)
            elif query.data.startswith('set_'):
                _, param, value = query.data.split('_', 2)
                await self.set_parameter(update, context, param, value)
            elif query.data == 'repeat_generation':
                await self.repeat_generation(update, context)
            elif query.data == 'modify_settings':
                await self.modify_settings(update, context)
        except Exception as e:
            self.logger.error(f"Error in handle_callback: {str(e)}")
            await query.message.reply_text("Произошла ошибка при обработке вашего запроса. Пожалуйста, попробуйте еще раз.")

    async def change_parameter(self, update: Update, context, param):
        user_id = update.effective_user.id
        settings = self.user_settings.get(user_id, self.config.default_settings)
        
        if param in ['model', 'vae', 'lora', 'sampler']:
            options = getattr(self.resource_scanner, f'scan_{param}s')()
            keyboard = [[InlineKeyboardButton(option, callback_data=f'set_{param}_{option}')] for option in options]
        elif param in ['cfg', 'steps', 'size']:
            await update.callback_query.edit_message_text(f"Введите новое значение для {param}:")
            context.user_data['expect_input'] = param
            return
        elif param in ['prompt', 'negative_prompt']:
            current_value = settings.get(param, "Не задано")
            await update.callback_query.edit_message_text(
                f"Текущий {'промпт' if param == 'prompt' else 'негативный промпт'}:\n{current_value}\n\n"
                f"Введите новый {'промпт' if param == 'prompt' else 'негативный промпт'}:"
            )
            context.user_data['expect_input'] = param
            return
        else:
            await update.callback_query.edit_message_text(f"Неизвестный параметр: {param}")
            return
        
        keyboard.append([InlineKeyboardButton("Назад", callback_data='back_to_panel')])
        reply_markup = InlineKeyboardMarkup(keyboard)
        await update.callback_query.edit_message_text(f"Выберите {param}:", reply_markup=reply_markup)

    async def set_parameter(self, update: Update, context, param, value):
        user_id = update.effective_user.id
        self.user_settings.setdefault(user_id, self.config.default_settings.copy())
        self.user_settings[user_id][param] = value
        await self.show_interactive_panel(update, context)

    async def handle_text_input(self, update: Update, context):
        user_id = update.effective_user.id
        text = update.message.text
        
        if 'expect_input' in context.user_data:
            param = context.user_data['expect_input']
            del context.user_data['expect_input']
            
            if param in ['cfg_scale', 'steps']:
                try:
                    value = float(text) if param == 'cfg_scale' else int(text)
                    self.user_settings.setdefault(user_id, self.config.default_settings.copy())
                    self.user_settings[user_id][param] = value
                    await self.show_interactive_panel(update, context)
                except ValueError:
                    await update.message.reply_text(f"Неверный формат. Пожалуйста, введите число для {param}.")
            elif param == 'size':
                try:
                    width, height = map(int, text.split('x'))
                    self.user_settings.setdefault(user_id, self.config.default_settings.copy())
                    self.user_settings[user_id]['size'] = f"{width}x{height}"
                    await self.show_interactive_panel(update, context)
                except ValueError:
                    await update.message.reply_text("Неверный формат. Пожалуйста, введите размер в формате ШИРИНАxВЫСОТА (например, 512x512).")
            elif param in ['prompt', 'negative_prompt']:
                self.user_settings.setdefault(user_id, self.config.default_settings.copy())
                self.user_settings[user_id][param] = text
                await self.show_interactive_panel(update, context)
        else:
            # Если это не ожидаемый ввод, считаем текст промптом и показываем панель настроек
            self.user_settings.setdefault(user_id, self.config.default_settings.copy())
            self.user_settings[user_id]['prompt'] = text
            await self.show_interactive_panel(update, context)

    async def apply_default_settings(self, update: Update, context):
        user_id = update.effective_user.id
        self.user_settings[user_id] = self.config.default_settings.copy()
        await self.show_interactive_panel(update, context)

    async def apply_last_settings(self, update: Update, context):
        user_id = update.effective_user.id
        if user_id in self.last_settings:
            self.user_settings[user_id] = self.last_settings[user_id].copy()
        await self.show_interactive_panel(update, context)

    async def start_generation(self, update: Update, context):
        user_id = update.effective_user.id
        settings = self.user_settings.setdefault(user_id, self.config.default_settings.copy())
        self.last_settings[user_id] = settings.copy()
        
        await self.queue.add_task(self.generate_and_send, update, context, settings)
        
        if update.callback_query:
            status_message = await update.callback_query.edit_message_text("🔄 Задача добавлена в очередь. Ожидайте начала генерации.")
        else:
            status_message = await update.message.reply_text("🔄 Задача добавлена в очередь. Ожидайте начала генерации.")
        
        await self.update_queue_status(status_message)

    async def update_queue_status(self, message):
        try:
            while self.queue.queue_size > 0:
                await message.edit_text(f"🔄 Ваша задача в очереди.\nПозиция: {self.queue.queue_size}\nТекущая задача: {self.queue.current_task_name}\nВремя выполнения: {self.queue.elapsed_time:.2f}s")
                await asyncio.sleep(5)
        except Exception as e:
            self.logger.error(f"Error updating queue status: {str(e)}")

    async def update_status(self, message, text):
        try:
            await message.edit_text(f"{text}\nВ очереди: {self.queue.queue_size}\nВремя выполнения: {self.queue.elapsed_time:.2f}s")
        except Exception as e:
            self.logger.error(f"Error updating status: {str(e)}")

    async def generate_and_send(self, update: Update, context, settings):
        try:
            if update.callback_query:
                status_message = await update.callback_query.message.reply_text("🔄 Подготовка к генерации...")
            else:
                status_message = await update.message.reply_text("🔄 Подготовка к генерации...")

            model_name = settings.get('model', self.config.default_model)
            model_type = settings.get('model_type', self.config.default_model_type)
            vae_name = settings.get('vae', None)

            print(f"Загружается модель: {model_name} (Тип: {model_type})")

            if self.generator.model is None or (self.generator.model.config and self.generator.model.config.get('name') != model_name):
                await self.update_status(status_message, f"🔄 Загрузка модели: {model_name} (Тип: {model_type})")
                await self.generator.load_model(model_name, vae_name=vae_name)

            if settings.get('lora'):
                await self.update_status(status_message, "🔄 Загрузка LoRA...")
                await self.generator.load_lora(settings['lora'])

            await self.update_status(status_message, "🚀 Генерация началась...")

            # async def progress_callback(progress, elapsed_time, remaining_time, speed):
            #     try:
            #         progress = progress or 0
            #         elapsed_time = elapsed_time or 0
            #         remaining_time = remaining_time or 0
            #         speed = speed or 0

            #         await self.update_status(
            #             status_message,
            #             f"🚀 Генерация: {progress:.2f} %\n"
            #             f"⏱️ Прошло: {elapsed_time:.2f} s\n"
            #             f"⏳ Осталось: {remaining_time:.2f} s\n"
            #             f"⚡ Скорость: {speed:.2f} s/шаг"
            #         )
            #     except Exception as e:
            #         print(f"ERROR in progress_callback: {e}")

            # image = await self.generator.generate_image(settings, progress_callback=progress_callback)
            image = await self.generator.generate_image(settings)

            await self.update_status(status_message, "✅ Генерация завершена!")

            output_path = os.path.join(self.config.output_path, f"{update.effective_user.id}_{update.effective_message.message_id}.png")
            image.save(output_path)

            metadata = json.dumps(settings, indent=2, ensure_ascii=False)
            caption = f"🎉 Вот ваше изображение!\n\n📄 Метаданные:\n{metadata}"
            
            keyboard = [
                [InlineKeyboardButton("🔄 Повторить", callback_data='repeat_generation')],
                [InlineKeyboardButton("✏️ Изменить", callback_data='modify_settings')]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)
            
            with open(output_path, 'rb') as image_file:
                await update.effective_message.reply_photo(
                    photo=image_file,
                    caption=caption,
                    reply_markup=reply_markup
                )

            self.logger.info(f"Image generated successfully for user {update.effective_user.id}")

        except Exception as e:
            await update.effective_message.reply_text(f"Произошла ошибка: {str(e)}")
            self.logger.error(f"Error during image generation: {str(e)}")

    async def help_command(self, update: Update, context):
        help_text = """
        🤖 Добро пожаловать в бот генерации изображений!

        Доступные команды:
        /start или /s - Показать интерактивную панель настроек
        /generate или /g - Начать генерацию с текущими настройками
        /help или /h - Показать это сообщение

        Команды для быстрой настройки параметров:
        /set_model или /sm <модель> - Установить модель
        /set_vae или /sv <vae> - Установить VAE
        /set_lora или /sl <lora> - Установить LoRA
        /set_sampler или /ss <семплер> - Установить семплер
        /set_cfg_scale или /sc <значение> - Установить CFG Scale
        /set_steps или /st <количество> - Установить количество шагов
        /set_size или /sz <ширина>x<высота> - Установить размер изображения
        /set_prompt или /sp <промпт> - Установить промпт
        /set_negative_prompt или /sn <негативный промпт> - Установить негативный промпт

        Вы также можете просто отправить текстовое сообщение, и оно будет использовано как промпт для генерации.
        """
        await update.message.reply_text(help_text)

    async def run(self):
        self.application.add_handler(CommandHandler(["start", "s"], self.start))
        self.application.add_handler(CommandHandler(["generate", "g"], self.start_generation))
        self.application.add_handler(CommandHandler(["help", "h"], self.help_command))
        self.application.add_handler(CommandHandler("clear", self.clear))
        self.application.add_handler(CallbackQueryHandler(self.handle_callback))
        self.application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_text_input))
        
        command_handlers = {
            'model': self.set_model_command,
            'vae': self.set_vae_command,
            'lora': self.set_lora_command,
            'sampler': self.set_sampler_command,
            'cfg_scale': self.set_cfg_scale_command,
            'steps': self.set_steps_command,
            'size': self.set_size_command,
            'prompt': self.set_prompt_command,
            'negative_prompt': self.set_negative_prompt_command,
        }

        for param, handler in command_handlers.items():
            self.application.add_handler(CommandHandler([f"set_{param}", f"s{param[0]}"], handler))
        
        self.application.add_error_handler(self.handle_error)
    
        self.logger.info("Starting bot")

        async with self.application:
            await self.application.initialize()
            await self.application.start()
            await self.application.updater.start_polling()
            
            self.logger.info("Bot started, waiting for messages")
            queue_task = asyncio.create_task(self.queue.process_queue())
            
            # try:
            #     await self.application.updater.wait_closed()
            # finally:
            await asyncio.Event().wait() 
            await queue_task

    async def set_parameter_command(self, update: Update, context, param):
        user_id = update.effective_user.id
        if not context.args:
            await update.message.reply_text(f"Пожалуйста, укажите значение для {param}.")
            return
        
        value = ' '.join(context.args)
        self.user_settings.setdefault(user_id, self.config.default_settings.copy())
        
        if param in ['cfg_scale', 'steps', 'size']:
            try:
                if param == 'cfg_scale':
                    self.user_settings[user_id][param] = float(value)
                elif param == 'steps':
                    self.user_settings[user_id][param] = int(value)
                elif param == 'size':
                    width, height = map(int, value.split('x'))
                    self.user_settings[user_id][param] = f"{width}x{height}"
            except ValueError:
                await update.message.reply_text(f"Неверный формат для {param}. Попробуйте еще раз.")
                return
        else:
            self.user_settings[user_id][param] = value
        
        await update.message.reply_text(f"{param.capitalize()} установлен на {value}.")
        await self.show_interactive_panel(update, context)

    async def handle_error(self, update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
        self.logger.error(msg="Exception while handling an update:", exc_info=context.error)
        if update and update.effective_message:
            await update.effective_message.reply_text("Произошла ошибка при обработке вашего запроса. Пожалуйста, попробуйте еще раз или обратитесь к администратору.")

    async def set_model_command(self, update: Update, context):
        await self.set_parameter_command(update, context, 'model')

    async def set_vae_command(self, update: Update, context):
        await self.set_parameter_command(update, context, 'vae')

    async def set_lora_command(self, update: Update, context):
        await self.set_parameter_command(update, context, 'lora')

    async def set_sampler_command(self, update: Update, context):
        await self.set_parameter_command(update, context, 'sampler')

    async def set_cfg_scale_command(self, update: Update, context):
        await self.set_parameter_command(update, context, 'cfg_scale')

    async def set_steps_command(self, update: Update, context):
        await self.set_parameter_command(update, context, 'steps')

    async def set_size_command(self, update: Update, context):
        await self.set_parameter_command(update, context, 'size')

    async def set_prompt_command(self, update: Update, context):
        await self.set_parameter_command(update, context, 'prompt')

    async def set_negative_prompt_command(self, update: Update, context):
        await self.set_parameter_command(update, context, 'negative_prompt')

    async def clear(self, update: Update, context):
        self.generator.unload_model()
        await update.message.reply_text("Модель выгружена из памяти.")

    async def repeat_generation(self, update: Update, context):
        user_id = update.effective_user.id
        if user_id in self.last_settings:
            settings = self.last_settings[user_id].copy()
            await self.queue.add_task(self.generate_and_send, update, context, settings)
            await update.callback_query.message.reply_text("🔄 Задача добавлена в очередь. Ожидайте начала генерации.")
        else:
            await update.callback_query.message.reply_text("Нет доступных настроек для повтора генерации.")

    async def modify_settings(self, update: Update, context):
        await update.callback_query.message.reply_text("Настройте параметры генерации:")
        await self.show_interactive_panel(update, context)

