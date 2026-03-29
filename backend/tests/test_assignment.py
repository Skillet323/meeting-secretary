import pytest
from app.services.assignment_engine import assign_tasks_to_participants, _find_participant_by_name
from app.models import Participant, Rule
from sqlmodel import Session, create_engine
import json

def test_find_participant_by_name():
    participants = [
        Participant(name="Alice", email="alice@example.com"),
        Participant(name="Bob", email="bob@example.com"),
    ]
    assert _find_participant_by_name(participants, "Alice") is participants[0]
    assert _find_participant_by_name(participants, "alice") is participants[0]
    assert _find_participant_by_name(participants, "bob@") is participants[1]
    assert _find_participant_by_name(participants, "Unknown") is None

def test_assign_tasks_round_robin(tmp_path):
    # Создадим временную базу
    db_url = f"sqlite:///{tmp_path}/test.db"
    engine = create_engine(db_url)
    from app.models import SQLModel
    SQLModel.metadata.create_all(engine)
    with Session(engine) as session:
        # Добавим двух участников
        session.add(Participant(name="Alice", email="alice@example.com"))
        session.add(Participant(name="Bob", email="bob@example.com"))
        session.commit()
        tasks = [{"description": "Do something", "assignee_hint": None}]
        result = assign_tasks_to_participants(tasks, session)
        assignee = result[0]["assignee"]
        assert assignee in ["Alice", "Bob", "alice@example.com", "bob@example.com"]
