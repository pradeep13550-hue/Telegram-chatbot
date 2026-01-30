#!/usr/bin/env python3
import subprocess
import sys
import os

def install_requirements():
    """Install required packages"""
    print("Installing requirements...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])

def check_env():
    """Check environment variables"""
    env_file = ".env"
    if not os.path.exists(env_file):
        print("Creating .env file...")
        with open(env_file, "w") as f:
            f.write("""# Telegram Bot Token (from @BotFather)
TELEGRAM_TOKEN=your_telegram_bot_token_here

# DeepSeek API Key
DEEPSEEK_API_KEY=your_deepseek_api_key_here

# Admin User IDs (comma-separated)
ADMIN_IDS=123456789,987654321

# Redis URL (optional, for conversation history)
REDIS_URL=redis://localhost:6379/0
REDIS_ENABLED=true
""")
        print(f"Please edit {env_file} with your actual tokens")
        return False
    return True

def setup_redis():
    """Setup Redis instructions"""
    print("\n=== Redis Setup ===")
    print("Redis is optional but recommended for conversation history.")
    print("\nInstall Redis:")
    print("  Ubuntu/Debian: sudo apt-get install redis-server")
    print("  macOS: brew install redis")
    print("  Windows: Download from https://github.com/microsoftarchive/redis/releases")
    print("\nStart Redis:")
    print("  Ubuntu/Debian: sudo systemctl start redis")
    print("  macOS: brew services start redis")
    print("\nDisable Redis in .env if not needed: REDIS_ENABLED=false")

def main():
    """Main setup function"""
    print("=== DeepSeek Telegram Bot Setup ===")
    
    # Install requirements
    install_requirements()
    
    # Check environment
    if check_env():
        print("\n✅ Environment check passed")
    else:
        print("\n⚠️  Please configure .env file")
    
    # Redis setup
    setup_redis()
    
    print("\n=== Setup Complete ===")
    print("\nTo run the bot:")
    print("1. Configure .env with your API keys")
    print("2. Start Redis (optional)")
    print("3. Run: python bot.py")
    print("\nFor Docker: docker-compose up --build")

if __name__ == "__main__":
    main()
