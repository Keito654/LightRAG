import asyncio
from operation.operate import query, insert

async def main():
    await insert()
    await query()

if __name__ == "__main__":
    asyncio.run(main())