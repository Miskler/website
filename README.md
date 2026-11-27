# website

Для запуска используется Python 3.12 и проверенный набор зависимостей:

```bash
python3.12 -m venv venv
./venv/bin/pip install -r requirements.lock
```

После изменения зависимостей обновите lock-файл командой `uv pip compile requirements.txt --python-version 3.12 --output-file requirements.lock` и проверьте сайт.
