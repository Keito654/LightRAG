import asyncio
from operation.operate import query, insert

# Windows 環境向けのハック
# 参考: https://stackoverflow.com/questions/31469707/changing-the-locale-preferred-encoding-in-python-3-in-windows
import os
if os.name == 'nt':
    import _locale
    _locale._getdefaultlocale_backup = _locale._getdefaultlocale
    _locale._getdefaultlocale = (lambda *args: (_locale._getdefaultlocale_backup()[0], 'UTF-8'))

async def main():
    await insert()
    await query()

if __name__ == "__main__":
    asyncio.run(main())