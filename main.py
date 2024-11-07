import asyncio
from operation.operate import query 

async def main():
    await query()

if __name__ == "__main__":
    asyncio.run(main())