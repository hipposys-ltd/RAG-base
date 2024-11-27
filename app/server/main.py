import os

from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from starlette.middleware.sessions import SessionMiddleware

from app.server.general import general_router
from app.server.chat import chat_router
from app.server.embeddings import embeddings_router
from app.databases.postgres import Database
from app.utils.config import Config
from app.utils.logger import Logger


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Run the database setup and teardown."""
    
    await Database.setup()
    Logger().get_logger().info('Database setup complete')

    yield
    # Optionally add teardown code here


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    SessionMiddleware,
    secret_key=os.environ['SECRET_KEY'],
    https_only=Config.get_deploy_env() == 'PROD',
)

@app.middleware("http")
async def check_token_middleware(request: Request, call_next):
    """Allow only requests with the correct token."""
    token = request.headers.get("x-access-token")
    if (Config.get_deploy_env() != 'LOCAL') and (token != os.environ['FAST_API_ACCESS_SECRET_TOKEN']):
        return JSONResponse(status_code=403, content={'reason': 'Invalid or missing token'})
    
    response = await call_next(request)
    return response


app.include_router(chat_router, prefix='/chat')
app.include_router(embeddings_router, prefix='/embeddings')
app.include_router(general_router, prefix='')
