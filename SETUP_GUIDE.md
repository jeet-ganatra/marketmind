# MarketMind — Complete Setup Guide

## Machine: MacBook Air M4, 16GB RAM, macOS

This guide assumes a fresh machine with nothing installed. Follow every step in order.

---

## Step 1: Install Homebrew (macOS Package Manager)

Open Terminal (Cmd + Space → type "Terminal" → Enter) and run:

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

After installation, Homebrew will print instructions to add it to your PATH. It will look something like:

```bash
echo >> ~/.zprofile
echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> ~/.zprofile
eval "$(/opt/homebrew/bin/brew shellenv)"
```

**Run whatever it tells you.** Then verify:

```bash
brew --version
```

---

## Step 2: Install Core Tools

```bash
# Git (may already be installed via Xcode CLI tools, but let's be sure)
brew install git

# Python 3.12 via pyenv (so you can manage multiple Python versions later)
brew install pyenv

# Add pyenv to your shell
echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.zshrc
echo '[[ -d $PYENV_ROOT/bin ]] && export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.zshrc
echo 'eval "$(pyenv init -)"' >> ~/.zshrc

# Reload shell
source ~/.zshrc

# Install Python 3.12
pyenv install 3.12
pyenv global 3.12

# Verify
python --version
# Should show: Python 3.12.x
```

---

## Step 3: Install Poetry (Python Package Manager)

```bash
curl -sSL https://install.python-poetry.org | python3 -

# Add poetry to your PATH
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.zshrc
source ~/.zshrc

# Verify
poetry --version
```

### Why Poetry?
- Industry-standard dependency management with a lockfile for reproducible builds
- Handles virtual environments automatically
- Clean integration with pyproject.toml
- Widely used in production Python projects

---

## Step 4: Install VS Code

```bash
brew install --cask visual-studio-code
```

After installation, open VS Code and install these extensions (Cmd+Shift+X to open Extensions panel):

1. **Python** (by Microsoft) — Python language support
2. **Claude Code** (by Anthropic) — AI coding assistant
3. **Ruff** (by Astral) — Fast Python linter/formatter
4. **GitLens** (by GitKraken) — Git history visualization (optional but helpful)

### Claude Code Extension Setup

1. Open VS Code
2. Go to Extensions (Cmd+Shift+X)
3. Search "Claude Code" → Install
4. Open the Claude Code panel (should appear in the sidebar)
5. Sign in with your Anthropic account (same one as your work machine)
6. You should now be able to chat with Claude directly in VS Code

---

## Step 5: Configure Git & GitHub

```bash
# Set your identity
git config --global user.name "Your Name"
git config --global user.email "your-github-email@example.com"

# Generate SSH key for GitHub (press Enter for all prompts to accept defaults)
ssh-keygen -t ed25519 -C "your-github-email@example.com"

# Start SSH agent
eval "$(ssh-agent -s)"

# Add key to agent
ssh-add ~/.ssh/id_ed25519

# Copy public key to clipboard
pbcopy < ~/.ssh/id_ed25519.pub
```

Now go to GitHub.com → Settings → SSH and GPG keys → New SSH key → Paste → Save.

Test it:

```bash
ssh -T git@github.com
# Should say: Hi <username>! You've successfully authenticated
```

---

## Step 6: Create the GitHub Repository

```bash
# Go to your projects directory (create one if you don't have one)
mkdir -p ~/projects
cd ~/projects

# Clone the repo after creating it on GitHub:
# Go to github.com → New Repository
#   Name: marketmind
#   Description: "AI-powered stock market decision-support agent"
#   Visibility: Public
#   Initialize with: DO NOT add README, .gitignore, or license (we'll add our own)
#   Click "Create repository"

# Then clone it:
git clone git@github.com:<YOUR_GITHUB_USERNAME>/marketmind.git
cd marketmind
```

---

## Step 7: Install Dependencies & Verify

```bash
cd ~/projects/marketmind

# Copy environment template
cp .env.example .env
# Edit .env and add your API keys (see Step 8 below)

# Install all dependencies (Poetry creates the virtual environment automatically)
poetry install

# Verify everything works
poetry run marketmind check-setup
```

You should see green checkmarks for `.env`, API keys, yfinance, OpenAI, and Anthropic.

---

## Step 8: Get API Keys

### OpenAI (for GPT-4o-mini)
1. Go to https://platform.openai.com/signup
2. Create account → Go to API Keys → Create new secret key
3. Add payment method (pay-as-you-go, set a $10/month spending limit)
4. Save the key — you'll put it in `.env`

### Anthropic (for Claude Sonnet)
1. Go to https://console.anthropic.com
2. Create account (or use existing) → Go to API Keys → Create key
3. Add payment method (set a $20/month spending limit)
4. Save the key — you'll put it in `.env`

### Alpha Vantage (for market data)
1. Go to https://www.alphavantage.co/support/#api-key
2. Get free API key (no credit card needed)
3. Save the key

### Finnhub (for news and analyst ratings)
1. Go to https://finnhub.io/register
2. Get free API key
3. Save the key

**IMPORTANT: Never commit these keys to git. They go in `.env` only.**

---

## Troubleshooting

### "No module named 'marketmind'"
This usually means the virtual environment wasn't set up correctly. Fix it by reinstalling:
```bash
rm -rf .venv
poetry install
poetry run marketmind check-setup
```

### `rm -rf .venv` fails with "Directory not empty"
Something is holding a lock on the directory (e.g. an active shell or VS Code). Try:
```bash
deactivate 2>/dev/null; rm -rf .venv
```
If that still fails:
```bash
sudo rm -rf .venv
```

### "command not found: python"
Your shell didn't pick up pyenv. Run `source ~/.zshrc` and try again.

### "poetry: command not found"
The poetry installer adds to `~/.local/bin`. Make sure it's in your PATH:
```bash
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.zshrc
source ~/.zshrc
```

### VS Code can't find Python
Open VS Code → Cmd+Shift+P → "Python: Select Interpreter" → Choose the `.venv` Python 3.12 interpreter in the project directory.

### SSH key issues with GitHub
Make sure you copied the `.pub` (public) key, not the private key. Run `cat ~/.ssh/id_ed25519.pub` to see it.
