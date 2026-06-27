import re

with open('main.py', 'r', encoding='utf-8') as f:
    content = f.read()

old_bs = "bs_symbol = 'sh.' + symbol if symbol.startswith('0') else 'sz.' + symbol"
new_bs = "bs_symbol = 'sh.' + symbol[-6:] if 'sh' in symbol.lower() else 'sz.' + symbol[-6:]"

content = content.replace(old_bs, new_bs)

with open('main.py', 'w', encoding='utf-8') as f:
    f.write(content)
