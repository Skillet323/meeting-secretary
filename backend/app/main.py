from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .api.routes import router
from .api.evaluation import router as evaluation_router
# Import models to ensure they are registered with SQLModel metadata
from .models import Meeting, GoldStandard, Task, EvaluationRun, EvaluationMetric  # noqa

app = FastAPI(title="Meeting Secretary")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # для прототипа; в проде ограничьте доменом фронтенда
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)
app.include_router(evaluation_router)