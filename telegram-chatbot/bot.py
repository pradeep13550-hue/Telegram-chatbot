import asyncio
import time
from typing import Optional, Dict, List, Tuple
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    CallbackQueryHandler,
    filters,
    ContextTypes,
    ConversationHandler
)
from telegram.constants import ParseMode, ChatAction
from telegram.error import TelegramError

from config import config
from utils import (
    DeepSeekClient,
    RedisConversationManager,
    MemoryConversationManager,
    RateLimiter,
    MessageProcessor,
    Analytics,
    RateLimitExceeded,
    APIError
)
from loguru import logger

# Initialize components
deepseek_client = DeepSeekClient(
    api_key=config.DEEPSEEK_API_KEY,
    base_url=config.DEEPSEEK_BASE_URL,
    model=config.MODEL_NAME
)

# Initialize conversation manager
if config.REDIS_ENABLED:
    conversation_manager = RedisConversationManager(
        redis_url=config.REDIS_URL,
        max_history=config.MAX_HISTORY_MESSAGES
    )
else:
    conversation_manager = MemoryConversationManager(
        max_history=config.MAX_HISTORY_MESSAGES
    )

rate_limiter = RateLimiter(redis_url=config.REDIS_URL)

# Bot statistics
bot_stats = {
    'total_requests': 0,
    'successful_requests': 0,
    'failed_requests': 0,
    'active_users': set(),
    'start_time': time.time()
}

class TelegramDeepSeekBot:
    def __init__(self):
        self.application = None
        self.user_settings: Dict[int, Dict] = {}
    
    async def initialize_components(self):
        """Initialize all components"""
        try:
            if hasattr(conversation_manager, 'initialize'):
                await conversation_manager.initialize()
            
            await rate_limiter.initialize()
            logger.info("All components initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize components: {e}")
    
    async def start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /start command"""
        user = update.effective_user
        bot_stats['active_users'].add(user.id)
        
        welcome_message = f"""
🤖 *Welcome to DeepSeek AI Assistant*, {user.first_name}!

I'm powered by DeepSeek's advanced AI model. Here's what I can do:

✨ *Available Commands:*
/start - Start the bot
/help - Show help message
/new - Start new conversation
/settings - Configure settings
/models - Available AI models
/status - Check bot status
/stats - Your usage statistics
/clear - Clear chat history

💡 *Pro Tips:*
• Use markdown for formatting
• Send /new to clear history
• Use /settings to customize

*Simply send me a message to start chatting!*
        """
        
        await update.message.reply_text(
            welcome_message,
            parse_mode=ParseMode.MARKDOWN,
            disable_web_page_preview=True
        )
    
    async def help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /help command"""
        help_text = """
📚 *Available Commands:*

*Basic Commands:*
/start - Welcome message
/help - Show this help
/new - Start fresh conversation
/clear - Clear chat history

*Settings & Info:*
/settings - Configure bot settings
/models - List available models
/status - Check bot status
/stats - Your usage statistics

*Admin Commands:*
/broadcast - Broadcast message (Admin only)
/stats_all - All users statistics (Admin only)

*Features:*
• Context-aware conversations
• Multiple AI models
• Markdown support
• Conversation history
• Rate limiting

Need more help? Contact admin.
        """
        
        await update.message.reply_text(
            help_text,
            parse_mode=ParseMode.MARKDOWN,
            disable_web_page_preview=True
        )
    
    async def models_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /models command"""
        models_text = """
🤖 *Available AI Models:*

*General Purpose:*
• `deepseek-chat` - General conversations (Default)
• `deepseek-reasoner` - Complex reasoning tasks

*Code Generation:*
• `deepseek-coder` - Code generation and review

*Features:*
• All models support 128K context
• Code completion
• Text generation
• Question answering

*Current Model:* `deepseek-chat`

Use /settings to change model (if implemented).
        """
        
        await update.message.reply_text(
            models_text,
            parse_mode=ParseMode.MARKDOWN
        )
    
    async def new_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /new command - clear conversation"""
        user_id = str(update.effective_user.id)
        
        await conversation_manager.clear_conversation(user_id)
        
        await update.message.reply_text(
            "🔄 *New conversation started!*\n"
            "Your chat history has been cleared.",
            parse_mode=ParseMode.MARKDOWN
        )
    
    async def clear_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /clear command (alias for /new)"""
        await self.new_command(update, context)
    
    async def settings_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /settings command"""
        keyboard = [
            [
                InlineKeyboardButton("📊 View Settings", callback_data="view_settings"),
            ],
            [
                InlineKeyboardButton("🔄 Reset Settings", callback_data="reset_settings"),
            ]
        ]
        
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await update.message.reply_text(
            "⚙️ *Bot Settings*\n\n"
            "Configure your preferences:",
            reply_markup=reply_markup,
            parse_mode=ParseMode.MARKDOWN
        )
    
    async def status_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /status command"""
        user_id = update.effective_user.id
        is_admin = user_id in config.ADMIN_IDS
        
        # Calculate uptime
        uptime_seconds = time.time() - bot_stats['start_time']
        uptime_str = self._format_uptime(uptime_seconds)
        
        status_text = f"""
📊 *Bot Status*

*User Info:*
• ID: `{user_id}`
• Name: {update.effective_user.first_name}
• Admin: {'✅' if is_admin else '❌'}

*Bot Stats:*
• Uptime: {uptime_str}
• Total Requests: {bot_stats['total_requests']}
• Successful: {bot_stats['successful_requests']}
• Failed: {bot_stats['failed_requests']}
• Active Users: {len(bot_stats['active_users'])}

*Configuration:*
• Model: `{config.MODEL_NAME}`
• Max Tokens: {config.MAX_TOKENS}
• Temperature: {config.TEMPERATURE}
• Rate Limit: {config.MESSAGES_PER_USER_PER_MINUTE}/min

*System Status:*
• DeepSeek API: {'✅ Connected' if config.DEEPSEEK_API_KEY else '❌ Disconnected'}
• Redis: {'✅ Enabled' if config.REDIS_ENABLED else '❌ Disabled'}
        """
        
        await update.message.reply_text(
            status_text,
            parse_mode=ParseMode.MARKDOWN
        )
    
    async def stats_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /stats command - user statistics"""
        user_id = str(update.effective_user.id)
        
        # Get user stats
        user_stats = await conversation_manager.get_user_stats(user_id)
        
        # Get conversation history
        conversation = await conversation_manager.get_conversation(user_id)
        user_message_count = len([m for m in conversation if m.get("role") == "user"])
        
        stats_text = f"""
📈 *Your Statistics*

*Usage:*
• Total Messages: {user_stats.get('total_messages', 0)}
• Current Session: {user_message_count} messages
• History Length: {len(conversation)} messages

*Limits:*
• Rate Limit: {config.MESSAGES_PER_USER_PER_MINUTE}/minute
• Max Context: {config.MAX_CONTEXT_LENGTH} tokens
• History Saved: {config.MAX_HISTORY_MESSAGES} messages

*Last Active:* {user_stats.get('last_active', 'Never')}

Use /new to clear your history.
        """
        
        await update.message.reply_text(
            stats_text,
            parse_mode=ParseMode.MARKDOWN
        )
    
    async def stats_all_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Admin: Show all users statistics"""
        user_id = update.effective_user.id
        
        if user_id not in config.ADMIN_IDS:
            await update.message.reply_text("❌ Admin only command.")
            return
        
        stats_text = f"""
📊 *All Users Statistics*

*Bot Statistics:*
• Total Requests: {bot_stats['total_requests']}
• Successful: {bot_stats['successful_requests']}
• Failed: {bot_stats['failed_requests']}
• Active Users: {len(bot_stats['active_users'])}
• Uptime: {self._format_uptime(time.time() - bot_stats['start_time'])}

*Configuration:*
• Model: {config.MODEL_NAME}
• Max Tokens: {config.MAX_TOKENS}
• Admin IDs: {len(config.ADMIN_IDS)}
        """
        
        await update.message.reply_text(
            stats_text,
            parse_mode=ParseMode.MARKDOWN
        )
    
    async def broadcast_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Admin: Broadcast message to all users"""
        user_id = update.effective_user.id
        
        if user_id not in config.ADMIN_IDS:
            await update.message.reply_text("❌ Admin only command.")
            return
        
        if not context.args:
            await update.message.reply_text(
                "Usage: /broadcast <message>\n"
                "Example: /broadcast Hello everyone!"
            )
            return
        
        message = " ".join(context.args)
        broadcast_text = f"""
📢 *Announcement*

{message}

*From:* Admin
        """
        
        # Note: In production, iterate through stored user IDs
        await update.message.reply_text(
            f"📢 *Broadcast Sent*\n\n"
            f"Message: {message}\n\n"
            f"*Note:* User database required for actual broadcasting.",
            parse_mode=ParseMode.MARKDOWN
        )
    
    async def handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle incoming text messages"""
        user_id = str(update.effective_user.id)
        message_text = update.message.text
        
        # Update active users
        bot_stats['active_users'].add(update.effective_user.id)
        bot_stats['total_requests'] += 1
        
        # Check rate limiting
        allowed, current_count = await rate_limiter.check_rate_limit(
            user_id=user_id,
            limit=config.MESSAGES_PER_USER_PER_MINUTE
        )
        
        if not allowed:
            await update.message.reply_text(
                f"⏳ *Rate limit exceeded*\n"
                f"You've sent {current_count} messages this minute.\n"
                f"Limit: {config.MESSAGES_PER_USER_PER_MINUTE} messages per minute.\n"
                f"Please wait a moment.",
                parse_mode=ParseMode.MARKDOWN
            )
            bot_stats['failed_requests'] += 1
            return
        
        # Show typing indicator
        try:
            await update.message.chat.send_action(action=ChatAction.TYPING)
        except:
            pass
        
        try:
            # Get conversation history
            conversation_history = await conversation_manager.get_conversation(user_id)
            
            # Format messages for API
            messages = MessageProcessor.format_messages_for_api(
                conversation_history,
                message_text
            )
            
            # Truncate if too long
            messages = MessageProcessor.truncate_conversation(
                messages,
                max_chars=config.MAX_CONTEXT_LENGTH * 4  # Rough char estimate
            )
            
            # Get response from DeepSeek
            response = await deepseek_client.generate_response(
                messages=messages,
                temperature=config.TEMPERATURE,
                max_tokens=config.MAX_TOKENS
            )
            
            if response is None:
                raise APIError("Empty response from API")
            
            # Update conversation history
            messages.append({
                "role": "assistant",
                "content": response
            })
            
            await conversation_manager.save_conversation(user_id, messages)
            
            # Track successful request
            bot_stats['successful_requests'] += 1
            Analytics.track_request(
                user_id=user_id,
                prompt_length=len(message_text),
                response_length=len(response),
                success=True
            )
            
            # Split and send response
            response_parts = MessageProcessor.split_long_response(response)
            
            for i, part in enumerate(response_parts):
                parse_mode = ParseMode.MARKDOWN if i == 0 else None
                await update.message.reply_text(
                    part,
                    parse_mode=parse_mode,
                    disable_web_page_preview=True
                )
                
        except RateLimitExceeded:
            await update.message.reply_text(
                "🚫 *API Rate Limit Exceeded*\n"
                "DeepSeek API rate limit reached. Please try again later.",
                parse_mode=ParseMode.MARKDOWN
            )
            bot_stats['failed_requests'] += 1
            
        except APIError as e:
            await update.message.reply_text(
                f"❌ *API Error*\n"
                f"Error: {str(e)}\n"
                f"Please try again in a moment.",
                parse_mode=ParseMode.MARKDOWN
            )
            bot_stats['failed_requests'] += 1
            
        except asyncio.TimeoutError:
            await update.message.reply_text(
                "⏱️ *Request Timeout*\n"
                "The request took too long. Please try again.",
                parse_mode=ParseMode.MARKDOWN
            )
            bot_stats['failed_requests'] += 1
            
        except Exception as e:
            logger.error(f"Unexpected error in handle_message: {e}")
            await update.message.reply_text(
                "❌ *Unexpected Error*\n"
                "An unexpected error occurred. Please try again.",
                parse_mode=ParseMode.MARKDOWN
            )
            bot_stats['failed_requests'] += 1
    
    async def handle_image(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle image messages"""
        photo = update.message.photo[-1] if update.message.photo else None
        caption = update.message.caption or ""
        
        if photo:
            await update.message.reply_text(
                "📷 *Image Received*\n\n"
                f"Caption: {caption}\n\n"
                "⚠️ *Note:* Full image analysis requires vision-capable model.\n"
                "Currently, I can only process text descriptions.",
                parse_mode=ParseMode.MARKDOWN
            )
        else:
            await update.message.reply_text(
                "Please send an image with a caption for description.",
                parse_mode=ParseMode.MARKDOWN
            )
    
    async def callback_handler(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle callback queries"""
        query = update.callback_query
        
        try:
            await query.answer()
            
            if query.data == "view_settings":
                settings_text = """
⚙️ *Current Settings*

*Model Settings:*
• Model: `deepseek-chat`
• Temperature: 0.7
• Max Tokens: 4096

*Bot Settings:*
• Max History: 20 messages
• Rate Limit: 10/min
• Context Length: 8000 tokens

*To change settings, contact admin.*
                """
                await query.edit_message_text(
                    settings_text,
                    parse_mode=ParseMode.MARKDOWN
                )
                
            elif query.data == "reset_settings":
                await query.edit_message_text(
                    "🔄 Settings reset to defaults.",
                    parse_mode=ParseMode.MARKDOWN
                )
                
        except Exception as e:
            logger.error(f"Error in callback handler: {e}")
            try:
                await query.edit_message_text(
                    "❌ Error processing request.",
                    parse_mode=ParseMode.MARKDOWN
                )
            except:
                pass
    
    async def error_handler(self, update: object, context: ContextTypes.DEFAULT_TYPE):
        """Handle errors"""
        logger.error(f"Update {update} caused error {context.error}")
        
        if isinstance(context.error, TelegramError):
            logger.error(f"Telegram API error: {context.error}")
        else:
            logger.error(f"Unexpected error: {context.error}", exc_info=True)
        
        # Try to send error message to user if possible
        if update and isinstance(update, Update) and update.effective_message:
            try:
                await update.effective_message.reply_text(
                    "❌ *An error occurred*\n"
                    "Please try again or contact admin if issue persists.",
                    parse_mode=ParseMode.MARKDOWN
                )
            except:
                pass
    
    def _format_uptime(self, seconds: float) -> str:
        """Format uptime to human readable string"""
        days, remainder = divmod(int(seconds), 86400)
        hours, remainder = divmod(remainder, 3600)
        minutes, seconds = divmod(remainder, 60)
        
        if days > 0:
            return f"{days}d {hours}h {minutes}m"
        elif hours > 0:
            return f"{hours}h {minutes}m"
        else:
            return f"{minutes}m {seconds}s"
    
    def setup_handlers(self):
        """Setup all command and message handlers"""
        # Command handlers
        self.application.add_handler(CommandHandler("start", self.start_command))
        self.application.add_handler(CommandHandler("help", self.help_command))
        self.application.add_handler(CommandHandler("new", self.new_command))
        self.application.add_handler(CommandHandler("clear", self.clear_command))
        self.application.add_handler(CommandHandler("settings", self.settings_command))
        self.application.add_handler(CommandHandler("models", self.models_command))
        self.application.add_handler(CommandHandler("status", self.status_command))
        self.application.add_handler(CommandHandler("stats", self.stats_command))
        self.application.add_handler(CommandHandler("stats_all", self.stats_all_command))
        self.application.add_handler(CommandHandler("broadcast", self.broadcast_command))
        
        # Callback query handler
        self.application.add_handler(CallbackQueryHandler(self.callback_handler))
        
        # Message handlers
        self.application.add_handler(
            MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_message)
        )
        self.application.add_handler(
            MessageHandler(filters.PHOTO, self.handle_image)
        )
        
        # Error handler
        self.application.add_error_handler(self.error_handler)
    
    async def shutdown(self):
        """Clean shutdown"""
        logger.info("Shutting down bot...")
        await deepseek_client.close()
        logger.info("Bot shutdown complete")
    
    async def run(self):
        """Run the bot"""
        try:
            # Initialize components
            await self.initialize_components()
            
            # Create Application
            self.application = (
                Application.builder()
                .token(config.TELEGRAM_TOKEN)
                .post_shutdown(self.shutdown)
                .build()
            )
            
            # Setup handlers
            self.setup_handlers()
            
            # Start the bot
            logger.info("🤖 Bot is starting...")
            await self.application.initialize()
            await self.application.start()
            
            # Start polling
            await self.application.updater.start_polling(
                allowed_updates=Update.ALL_TYPES,
                drop_pending_updates=True
            )
            
            logger.info("✅ Bot is running. Press Ctrl+C to stop.")
            
            # Keep running until interrupted
            while True:
                await asyncio.sleep(3600)  # Sleep for 1 hour
            
        except asyncio.CancelledError:
            logger.info("Bot stopped by user")
        except Exception as e:
            logger.error(f"Failed to run bot: {e}")
            raise
        finally:
            if self.application:
                await self.application.stop()

async def main():
    """Main function"""
    bot = TelegramDeepSeekBot()
    await bot.run()

if __name__ == "__main__":
    asyncio.run(main())
